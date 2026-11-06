# -*- mode: python -*-
"""Prepare an open-ephys recording for spike sorting (prep-oephys)

Run on the recording computer, on a session directory that holds one
recording. The recording's metadata are confirmed with the user, starting
from what the recording says: the metadata the stimulus presenter logged
(oeaudio-present logs them; jpresent doesn't) and the session directory's
name, `<animal>_<date>_<time>_<protocol>`, which both presenters set. The
animal is looked up in the bird colony, and the name, experimenter and animal
are checked with the registry. Values that differ from what the recording
says are listed in the entry attribute `revised_after_recording`.

Then the session directory is copied to the target (the spike sorting
workstation's share), where it can be sorted once it has its own name. At the
same time, arfx-oephys writes the recording to `<name>.arf` in a staging
directory on the recording computer, at lower priority, so that a new
recording can be made meanwhile. The registry checks the ARF as it would be deposited, and
it's copied to the target with `<name>.json`, which records the ARF's hash and
metadata, the session's hashes, and where they came from. Each copy is read
back and checked by hash. Then the session directory is removed from the
recording computer, leaving the ARF in staging until it's deposited (from the
target, once the sort is curated) and removed by hand.

Files are written under a temporary name (`.partial`) until they're complete,
so a run that stops can be run again: what's complete is kept.

"""

import argparse
import datetime
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import socket
import subprocess
import sysconfig
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx

from dlab import __version__
from dlab.abfzip import file_sha1
from dlab.util import add_log_arguments, setup_log

log = logging.getLogger("dlab")

PROG = "prep-oephys"
DTYPE = "extracellular-arf"
# the entry attribute listing the values that differ from what was recorded
REVISED = "revised_after_recording"
# prompted for after the experimenter, animal and protocol; pen and site are
# required, as they're part of the name
FIELDS = ("pen", "site", "probe", "hemisphere", "location", "x", "y", "z")
REQUIRED = ("pen", "site")
# what the prompts say about some fields
HINTS = {
    "location": "brain area; provisional until histology",
    "x": "microns",
    "y": "microns",
    "z": "microns from the top of the brain",
}
# the ARF data type of the continuous channels (arfx-oephys -T); the recordings
# are wideband
DATATYPE = "EXTRAC_RAW"
# what the registry can't check before the ARF is written, and can't check
# at all here, as the resource has no location yet
NOT_YET_CHECKED = ("sha1", "locations")

SESSION_DIR_RX = re.compile(
    r"(?P<animal>[^_]+)_(?P<time>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})"
    r"(?:_(?P<protocol>.+))?"
)
# a band as recordings name it, the color's abbreviation and the number (C182);
# anything else is taken as the start of the animal's UUID
BAND_RX = re.compile(r"(?P<color>[A-Za-z]{0,3})(?P<number>\d{1,4})")


@dataclass
class Session:
    """An open-ephys session directory that holds one recording"""

    path: Path
    recording: Path
    animal: str
    time: datetime.datetime
    protocol: str | None


def find_session(path: Path) -> Session:
    """The session in path (a session directory, or one inside it). Raises
    ValueError if path doesn't hold exactly one recording or the session
    directory's name can't be parsed."""
    recordings = sorted(p.parent for p in path.glob("**/structure.oebin"))
    if not recordings:
        raise ValueError(f"{path}: no open-ephys recording (structure.oebin) here")
    if len(recordings) > 1:
        names = ", ".join(str(p.relative_to(path)) for p in recordings)
        raise ValueError(
            f"{path} holds {len(recordings)} recordings ({names}); an ARF file "
            "can only hold one, and we're still working out how to handle a "
            "session directory with more than one"
        )
    [recording] = recordings
    for dir in (recording, *recording.parents):
        m = SESSION_DIR_RX.fullmatch(dir.name)
        if m is not None:
            return Session(
                path=dir,
                recording=recording,
                animal=m.group("animal"),
                time=datetime.datetime.strptime(m.group("time"), "%Y-%m-%d_%H-%M-%S"),
                protocol=m.group("protocol"),
            )
    raise ValueError(
        f"{recording}: no session directory named <animal>_<date>_<time>_<protocol>"
    )


def read_presenter_metadata(recording: Path) -> dict[str, Any]:
    """The metadata the stimulus presenter logged in a recording, or {}"""
    try:
        from arfx.oephys import presenter_metadata
        from arfx.oephys import recording as oephys_recording
    except ImportError as err:
        raise RuntimeError(
            f"{PROG} needs arfx 3.1.1 or later: install melizalab-tools[oephys]"
        ) from err
    return presenter_metadata(oephys_recording(recording))


def recorded_values(session: Session, presenter: dict[str, Any]) -> dict[str, Any]:
    """What the recording says: the presenter's metadata, and the animal and
    protocol from the session directory's name. The animal is as the recording
    names it (a band or the start of a UUID)."""
    values = dict(presenter)
    values["animal"] = session.animal
    protocol = values.setdefault("protocol", session.protocol)
    if session.protocol is not None and protocol != session.protocol:
        log.warning(
            "- the presenter logged protocol '%s', but the session directory "
            "is named for '%s'",
            protocol,
            session.protocol,
        )
    return {k: v for k, v in values.items() if v is not None}


def colony_url(registry_url: str) -> str:
    """The bird colony's API, on the registry's server"""
    return urljoin(registry_url, "/birds/api/")


def find_animals(client: httpx.Client, colony: str, animal: str) -> list[dict]:
    """The animals in the colony that animal (a band or the start of a UUID) may
    name. A band is looked up by its number only, as the API can't match the
    color's abbreviation."""
    m = BAND_RX.fullmatch(animal)
    params = {"band": m.group("number")} if m else {"uuid": animal}
    r = client.get(urljoin(colony, "animals/"), params=params)
    r.raise_for_status()
    return r.json()


def animal_summary(
    client: httpx.Client, colony: str, uuid: str, when: datetime.datetime
) -> str:
    """A description of the animal for the user to confirm: who it is, and the
    events in its history nearest when"""
    r = client.get(urljoin(colony, f"animals/{uuid}/"))
    r.raise_for_status()
    animal = r.json()
    lines = [
        f"  {animal['name']} ({animal['uuid']}): {animal['species']}, "
        f"sex {animal['sex']}, {'alive' if animal['alive'] else 'not alive'}, "
        f"last at {animal['last_location'] or 'unknown'}"
    ]
    r = client.get(urljoin(colony, f"animals/{uuid}/events/"))
    r.raise_for_status()
    events = r.json()
    day = when.date()

    def distance(event):
        return abs((datetime.date.fromisoformat(event["date"]) - day).days)

    nearest = sorted(sorted(events, key=distance)[:3], key=lambda e: e["date"])
    for event in nearest:
        description = f" ({event['description']})" if event["description"] else ""
        lines.append(f"    {event['date']}: {event['status']}{description}")
    lines.append(f"    {day}: this recording")
    return "\n".join(lines)


Ask = Callable[[str], str]


def ask(
    input: Ask,
    label: str,
    default: Any = None,
    *,
    parse: Callable[[str], Any] = str,
    required: bool = False,
) -> Any:
    """Prompts for a value until one is given that parse accepts. An empty answer
    is the default, or None if the value isn't required."""
    shown = f" [{default}]" if default is not None else ""
    while True:
        text = input(f"{label}{shown}: ").strip()
        if not text:
            if default is not None or not required:
                return default
            print("  a value is needed")
            continue
        try:
            return parse(text)
        except ValueError as err:
            print(f"  {err}")


def parse_int(text: str) -> int:
    try:
        return int(text)
    except ValueError:
        raise ValueError("needs a whole number") from None


def parse_value(text: str) -> int | float | str:
    """A number if text is one, otherwise text"""
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass
    return text


def choose_animal(
    input: Ask,
    client: httpx.Client,
    colony: str,
    default: str | None,
    when: datetime.datetime,
) -> tuple[str, str]:
    """Prompts for the animal and confirms it from the colony. Returns (the
    animal as the user gave it, its UUID)."""
    while True:
        given = ask(
            input, "animal (band, e.g. C182, or start of UUID)", default, required=True
        )
        found = find_animals(client, colony, given)
        if not found:
            print(f"  no animal in the colony matches '{given}'")
            default = None
            continue
        if len(found) == 1:
            [animal] = found
        else:
            print(f"  animals that '{given}' may name:")
            for i, animal in enumerate(found, start=1):
                print(f"   {i}. {animal['name']} ({animal['uuid'][:8]})")
            choice = ask(input, "which one (blank to give another)", parse=parse_int)
            if choice is None or not 1 <= choice <= len(found):
                default = None
                continue
            animal = found[choice - 1]
        print(animal_summary(client, colony, animal["uuid"], when))
        if ask(input, "is this the animal? (y/n)", "y").lower().startswith("y"):
            return given, animal["uuid"]
        default = None


def default_name(animal: str, pen: int, site: int) -> str:
    """The recording's name by the lab's convention: the band (C182), or the
    first 8 characters of the UUID, then the pen and site"""
    animal = animal.upper() if BAND_RX.fullmatch(animal) else animal[:8].lower()
    return f"{animal}_{pen}_{site}"


def ask_values(
    input: Ask,
    client: httpx.Client,
    colony: str,
    session: Session,
    defaults: dict[str, Any],
) -> dict[str, Any]:
    """Prompts for the recording's metadata, with defaults. Returns the values,
    with the animal as given and its UUID as `animal_uuid`."""
    values = {}
    values["experimenter"] = ask(
        input,
        "experimenter (registry username)",
        defaults.get("experimenter"),
        required=True,
    )
    values["animal"], values["animal_uuid"] = choose_animal(
        input, client, colony, defaults.get("animal"), session.time
    )
    values["protocol"] = ask(input, "protocol", defaults.get("protocol"), required=True)
    for field in FIELDS:
        parse = parse_int if field in REQUIRED else parse_value
        label = f"{field} ({HINTS[field]})" if field in HINTS else field
        values[field] = ask(
            input, label, defaults.get(field), parse=parse, required=field in REQUIRED
        )
    # a name the user gave is kept; one by the convention follows the answers
    name = default_name(values["animal"], values["pen"], values["site"])
    given = defaults.get("name")
    if given is not None and given != default_name(
        defaults["animal"], defaults["pen"], defaults["site"]
    ):
        name = given
    values["name"] = ask(input, "name", name, required=True)
    return values


def resource_body(values: dict[str, Any], presenter: dict[str, Any]) -> dict:
    """The resource as nbank deposit would register it (without its hash and
    location): the presenter's metadata, overridden by the values"""
    metadata = {
        k: v
        for k, v in {**presenter, **values}.items()
        if k not in ("name", "experimenter", "animal", "animal_uuid") and v is not None
    }
    return {
        "name": values["name"],
        "dtype": DTYPE,
        "experimenter": values["experimenter"],
        "animal": values["animal_uuid"],
        "metadata": metadata,
    }


def check_with_registry(
    session: httpx.Client,
    registry_url: str,
    body: dict,
    ignore: tuple[str, ...] = NOT_YET_CHECKED,
) -> list[str]:
    """The registry's objections to the resource, except to the fields in
    ignore, as messages"""
    from nbank import registry

    url, bodies = registry.check_resources(registry_url, [body])
    r = session.post(url, json=bodies)
    r.raise_for_status()
    [errors] = r.json()
    return registry.field_messages({k: v for k, v in errors.items() if k not in ignore})


def revised_fields(recorded: dict[str, Any], values: dict[str, Any]) -> list[str]:
    """The fields whose values differ from what the recording says"""
    revised = []
    for field, value in recorded.items():
        if field not in values:
            continue
        if field == "animal":
            changed = value.lower() != values[field].lower()
        else:
            changed = value != values[field]
        if changed:
            revised.append(field)
    return sorted(revised)


def arfx_oephys() -> str:
    """The arfx-oephys installed with this environment's Python, so that it's
    the arfx that prep-oephys was installed with, or else the one on the PATH"""
    found = shutil.which("arfx-oephys", path=sysconfig.get_path("scripts"))
    return found or "arfx-oephys"


def arfx_command(
    session: Session,
    values: dict[str, Any],
    revised: list[str],
    presenter: dict[str, Any],
    arffile: Path,
    datatype: str = DATATYPE,
) -> list[str]:
    """The arfx-oephys command that writes the recording to arffile with the
    values. Values are JSON, so that a string that looks like a number stays one."""
    cmd = [
        arfx_oephys(),
        "-f",
        str(arffile),
        "-T",
        datatype,
        "--experimenter",
        values["experimenter"],
        "--animal",
        values["animal_uuid"],
    ]
    for field in ("protocol", *FIELDS):
        if values.get(field) is not None:
            cmd += ["-k", f"{field}={json.dumps(values[field])}"]
    if revised:
        cmd += ["-k", f"{REVISED}={json.dumps(revised)}"]
    if any(field in presenter for field in revised):
        cmd.append("--override-presenter-metadata")
    cmd.append(str(session.path))
    return cmd


def prepare(
    input: Ask,
    client: httpx.Client,
    colony: str,
    session: httpx.Client,
    registry_url: str,
    recording: Session,
    presenter: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Prompts for the recording's metadata until the registry accepts them.
    Returns the values and the fields that were revised."""
    recorded = recorded_values(recording, presenter)
    others = sorted(set(presenter) - {"experimenter", "protocol", *FIELDS})
    if others:
        print(
            "also logged by the presenter: "
            + ", ".join(f"{k}={presenter[k]!r}" for k in others)
        )
    defaults = recorded
    while True:
        values = ask_values(input, client, colony, recording, defaults)
        errors = check_with_registry(
            session, registry_url, resource_body(values, presenter)
        )
        if not errors:
            return values, revised_fields(recorded, values)
        print("the registry would refuse these values:")
        for message in errors:
            print(f"  {message}")
        defaults = values


def partial(path: Path) -> Path:
    """Where path is written until it's complete"""
    return path.with_name(path.name + ".partial")


@dataclass
class Outputs:
    """The files prep-oephys writes: in staging, the ARF, arfx-oephys's log and
    the session's hashes; in the target, the session directory, the ARF and the
    manifest"""

    arf: Path
    conversion_log: Path
    hashes: Path
    target_session: Path
    target_arf: Path
    manifest: Path

    @classmethod
    def for_recording(
        cls, recording: Session, name: str, staging: Path, target: Path
    ) -> "Outputs":
        return cls(
            arf=staging / f"{name}.arf",
            conversion_log=staging / f"{name}.arfx-oephys.log",
            hashes=staging / f"{name}.session.json",
            target_session=target / recording.path.name,
            target_arf=target / f"{name}.arf",
            manifest=target / f"{name}.json",
        )


def session_size(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def space_warnings(recording: Session, staging: Path, target: Path) -> list[str]:
    """Advisory: whether staging is on the recording's drive, and whether staging
    and the target look too full for the ARF and the copies, taking the ARF to
    be about the size of the recording"""
    warnings = []
    if staging.stat().st_dev == recording.path.stat().st_dev:
        warnings.append(f"{staging} is on the same drive as {recording.path}")
    size = session_size(recording.path)
    for where, needed in ((staging, size), (target, 2 * size)):
        free = shutil.disk_usage(where).free
        if free < needed:
            warnings.append(
                f"{where} has {free / 1e9:.1f} GB free, and may need "
                f"{needed / 1e9:.1f} GB"
            )
    return warnings


def lower_priority(cmd: list[str]) -> list[str]:
    """cmd, run at lower CPU and I/O priority where nice and ionice are found, so
    that a recording going on at the same time isn't slowed. ionice's lowest
    best-effort level, not its idle class, which a busy drive could starve."""
    prefix = []
    if shutil.which("ionice"):
        prefix += ["ionice", "-c2", "-n7"]
    if shutil.which("nice"):
        prefix += ["nice", "-n", "10"]
    return prefix + cmd


class Conversion:
    """arfx-oephys running in the background, at lower priority, with its output
    in a log file. It writes the ARF under its temporary name, which is renamed
    to arffile when it finishes."""

    def __init__(self, cmd: list[str], arffile: Path, logfile: Path):
        if shutil.which(cmd[0]) is None:
            raise RuntimeError(
                "arfx-oephys isn't installed: install melizalab-tools[oephys]"
            )
        self.arffile = arffile
        self.logfile = logfile
        # arfx-oephys adds to a file that's there, so one a run left is removed
        partial(arffile).unlink(missing_ok=True)
        cmd = lower_priority(cmd)
        log.info("converting in the background: %s", shlex.join(cmd))
        log.info("arfx-oephys's output is in %s", logfile)
        with open(logfile, "w") as fp:
            self.process = subprocess.Popen(
                cmd, stdin=subprocess.DEVNULL, stdout=fp, stderr=subprocess.STDOUT
            )

    def finish(self) -> None:
        """Waits for arfx-oephys and renames the ARF. Raises RuntimeError if it
        failed, after removing what it wrote."""
        if self.process.poll() is None:
            log.info("waiting for arfx-oephys to finish")
        status = self.process.wait()
        if status != 0:
            partial(self.arffile).unlink(missing_ok=True)
            tail = "".join(self.logfile.read_text().splitlines(True)[-5:])
            raise RuntimeError(
                f"arfx-oephys failed (exit status {status}); its output, in "
                f"{self.logfile}, ends:\n{tail}"
            )
        partial(self.arffile).rename(self.arffile)

    def stop(self) -> None:
        """Stops arfx-oephys, if it's running, and removes what it wrote"""
        if self.process.poll() is None:
            log.info("stopping arfx-oephys")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        partial(self.arffile).unlink(missing_ok=True)


def read_arf_metadata(arffile: Path) -> dict[str, Any]:
    """The metadata nbank deposit will register for an ARF, from arfx's reader"""
    from arfx.nbank import read_metadata

    return read_metadata(arffile)


def check_arf(
    session: httpx.Client, registry_url: str, arffile: Path, values: dict[str, Any]
) -> dict:
    """Checks the ARF as nbank deposit would register it: its metadata must
    have the confirmed values, and the registry must accept it. Returns the
    resource, with the ARF's hash. Raises ValueError if it fails."""
    metadata = read_arf_metadata(arffile)
    confirmed = {k: v for k, v in values.items() if k not in ("name", "animal")}
    confirmed["animal"] = confirmed.pop("animal_uuid")
    differ = sorted(
        k for k, v in confirmed.items() if v is not None and metadata.get(k) != v
    )
    if differ:
        raise ValueError(
            f"{arffile}: the metadata in the ARF differ from the confirmed "
            f"values ({', '.join(differ)})"
        )
    body = {
        "name": values["name"],
        "dtype": DTYPE,
        "sha1": file_sha1(arffile),
        "experimenter": metadata.pop("experimenter"),
        "animal": metadata.pop("animal"),
        "metadata": metadata,
    }
    errors = check_with_registry(session, registry_url, body, ignore=("locations",))
    if errors:
        raise ValueError(
            f"{arffile}: the registry would refuse it: {'; '.join(errors)}"
        )
    return body


CHUNK = 1 << 20


def copy_file(src: Path, dst: Path) -> str:
    """Copies the data of src to dst, and returns the SHA-1 hash of what was
    read. The copy is flushed to its drive and, where the OS allows, dropped
    from the cache, so that reading it back reads what was written (on a
    network share, what the server has)."""
    sha1 = hashlib.sha1()
    with open(src, "rb") as fin, open(dst, "wb") as fout:
        while chunk := fin.read(CHUNK):
            sha1.update(chunk)
            fout.write(chunk)
        fout.flush()
        os.fsync(fout.fileno())
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(fout.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return sha1.hexdigest()


def copy_tree(src: Path, dst: Path) -> dict[str, str]:
    """Copies the directory src to dst, which mustn't exist, and reads the copy
    back to check it. Returns each file's SHA-1 hash, by its path in src. Only
    the contents are copied: a network share may refuse to set permissions and
    times. Raises OSError if the copy differs."""
    dst.mkdir()
    hashes = {}
    for path in sorted(src.rglob("*")):
        rel = path.relative_to(src)
        if path.is_dir():
            (dst / rel).mkdir()
        else:
            hashes[rel.as_posix()] = copy_file(path, dst / rel)
    for rel, sha1 in hashes.items():
        if file_sha1(dst / rel) != sha1:
            raise OSError(f"{dst / rel}: the copy doesn't match {src / rel}")
    return hashes


def copy_session(recording: Session, outputs: Outputs) -> dict[str, str]:
    """Copies the session directory to the target, unless an earlier run did,
    and returns its files' hashes, which are kept in staging. Raises OSError if
    the copy fails or differs, after removing it."""
    target = outputs.target_session
    if target.exists():
        if not outputs.hashes.exists():
            raise OSError(f"{target} is already there, but {PROG} didn't copy it")
        hashes = json.loads(outputs.hashes.read_text())
        if hashes["session"] != str(recording.path.resolve()):
            raise OSError(f"{target} was copied from {hashes['session']}")
        log.info("%s was copied by an earlier run", target)
        return hashes["files"]
    shutil.rmtree(partial(target), ignore_errors=True)
    log.info("copying %s to %s", recording.path, target)
    try:
        files = copy_tree(recording.path, partial(target))
        with open(outputs.hashes, "w") as fp:
            json.dump({"session": str(recording.path.resolve()), "files": files}, fp)
        partial(target).rename(target)
    except BaseException:
        shutil.rmtree(partial(target), ignore_errors=True)
        raise
    log.info("%s is copied and can be sorted", target)
    return files


def copy_arf(outputs: Outputs, sha1: str) -> None:
    """Copies the ARF to the target, unless an earlier run did, and checks the
    copy's hash. Raises OSError if the copy fails or differs, after removing
    it."""
    target = outputs.target_arf
    if target.exists():
        if file_sha1(target) != sha1:
            raise OSError(f"{target} is already there, and isn't {outputs.arf}")
        log.info("%s was copied by an earlier run", target)
        return
    log.info("copying %s to %s", outputs.arf, target)
    try:
        read = copy_file(outputs.arf, partial(target))
        if read != sha1 or file_sha1(partial(target)) != sha1:
            raise OSError(f"{target}: the copy doesn't match {outputs.arf}")
        partial(target).rename(target)
    except BaseException:
        partial(target).unlink(missing_ok=True)
        raise


def write_manifest(
    recording: Session, outputs: Outputs, resource: dict, hashes: dict[str, str]
) -> None:
    """Writes the manifest: the resource as it will be registered, the session's
    hashes, and where the ARF and the session directory came from"""
    manifest = {
        **resource,
        "source": {
            "host": socket.gethostname(),
            "arf": str(outputs.arf.resolve()),
            "session": str(recording.path.resolve()),
        },
        "session": outputs.target_session.name,
        "session_sha1": hashes,
        "prepared_by": f"{PROG} {__version__}",
        "prepared_on": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with open(outputs.manifest, "w") as fp:
        json.dump(manifest, fp, indent=2)
        fp.write("\n")


def remove_session(
    recording: Session, outputs: Outputs, hashes: dict[str, str]
) -> None:
    """Removes the session directory, if it holds only the files that were
    copied, at the sizes of the copies. Raises OSError otherwise: something may
    have been recorded into it since."""
    files = {
        p.relative_to(recording.path).as_posix(): p
        for p in recording.path.rglob("*")
        if p.is_file()
    }
    if set(files) != set(hashes):
        raise OSError(f"{recording.path} holds files that weren't copied")
    for rel, path in files.items():
        if path.stat().st_size != (outputs.target_session / rel).stat().st_size:
            raise OSError(f"{path} has changed since it was copied")
    log.info("removing %s", recording.path)
    shutil.rmtree(recording.path)


def convert_command(args) -> int:
    from nbank import util

    from dlab.neurobank import default_auth

    for dir in (args.staging, args.target):
        if not dir.is_dir():
            log.error("error: %s isn't a directory", dir)
            return 1
    try:
        recording = find_session(args.path)
    except ValueError as err:
        log.error("error: %s", err)
        return 1
    for warning in space_warnings(recording, args.staging, args.target):
        log.warning("warning: %s", warning)
    presenter = read_presenter_metadata(recording.recording)
    colony = args.colony or colony_url(args.registry_url)
    with (
        httpx.Client(auth=default_auth) as client,
        util.registry_session(args.registry_url, auth=default_auth) as session,
    ):
        values, revised = prepare(
            input, client, colony, session, args.registry_url, recording, presenter
        )
        if revised:
            log.info("revised after the recording: %s", ", ".join(revised))
        outputs = Outputs.for_recording(
            recording, values["name"], args.staging, args.target
        )
        if outputs.manifest.exists():
            log.error(
                "error: %s is already prepared (%s)", values["name"], outputs.manifest
            )
            return 1
        # what an earlier run finished is kept
        reuse = outputs.arf.exists()
        cmd = arfx_command(
            recording, values, revised, presenter, partial(outputs.arf), args.datatype
        )
        if args.dry_run:
            for path in (outputs.target_session, outputs.arf, outputs.target_arf):
                if path.exists():
                    log.info("%s is already there and won't be written again", path)
            print(shlex.join(cmd if reuse else lower_priority(cmd)))
            return 0
        conversion = None
        try:
            if reuse:
                log.info("%s is already written", outputs.arf)
            else:
                conversion = Conversion(cmd, outputs.arf, outputs.conversion_log)
            # the conversion finishes even if the copy fails, for the next run
            try:
                hashes = copy_session(recording, outputs)
            except OSError as err:
                log.error("error: %s", err)
                hashes = None
            if conversion is not None:
                conversion.finish()
            log.info("checking %s", outputs.arf)
            resource = check_arf(session, args.registry_url, outputs.arf, values)
        except (RuntimeError, ValueError) as err:
            if reuse:
                log.error("error: %s; remove it to convert the recording again", err)
            else:
                # an ARF the registry would refuse has to be written again
                outputs.arf.unlink(missing_ok=True)
                log.error("error: %s", err)
            return 1
        except BaseException:
            if conversion is not None:
                conversion.stop()
            raise
    if hashes is None:
        log.error("error: %s isn't copied; run again to copy it", recording.path)
        return 1
    try:
        copy_arf(outputs, resource["sha1"])
    except OSError as err:
        log.error("error: %s; run again to copy %s", err, outputs.arf)
        return 1
    write_manifest(recording, outputs, resource, hashes)
    if args.keep_session:
        log.info("keeping %s", recording.path)
    else:
        try:
            remove_session(recording, outputs, hashes)
        except OSError as err:
            log.error("error: %s; %s is kept", err, recording.path)
            return 1
    log.info(
        "%s is ready for sorting in %s; deposit %s once the sort is curated",
        values["name"],
        args.target,
        outputs.target_arf.name,
    )
    return 0


def script(argv=None):
    from nbank import registry

    p = argparse.ArgumentParser(
        prog=PROG,
        description="Prepare an open-ephys recording for spike sorting: confirm "
        "its metadata, copy it to the sorting workstation, convert it to ARF in a "
        "staging directory and copy the ARF too, then remove the recording",
    )
    p.add_argument(
        "-v", "--version", action="version", version=f"%(prog)s {__version__}"
    )
    add_log_arguments(p)
    p.add_argument(
        "-r",
        dest="registry_url",
        default=registry.default_registry(),
        help="URL of the registry (default: the NBANK_REGISTRY environment variable)",
    )
    sub = p.add_subparsers(title="commands", required=True)

    pp = sub.add_parser(
        "convert",
        help="confirm the metadata, copy the recording and its ARF to the "
        "target, and remove the recording",
    )
    pp.set_defaults(func=convert_command)
    pp.add_argument(
        "--colony",
        metavar="URL",
        help="URL of the bird colony's API (default: birds/api/ on the "
        "registry's server)",
    )
    pp.add_argument(
        "-T",
        dest="datatype",
        default=DATATYPE,
        metavar="DATATYPE",
        help="ARF data type of the continuous channels (default: %(default)s; see "
        "arfx-oephys --help-datatypes)",
    )
    pp.add_argument(
        "--staging",
        type=Path,
        required=True,
        metavar="DIR",
        help="the directory on this host to write the ARF to, where it's kept "
        "until it's deposited",
    )
    pp.add_argument(
        "--keep-session",
        action="store_true",
        help="don't remove the session directory once it's copied",
    )
    pp.add_argument(
        "--dry-run",
        action="store_true",
        help="confirm the metadata and show the arfx-oephys command, but don't "
        "write or copy anything",
    )
    pp.add_argument("path", type=Path, help="the session directory")
    pp.add_argument(
        "target",
        type=Path,
        help="the directory to copy the ARF and the session directory to",
    )

    args = p.parse_args(argv)
    setup_log(args.debug, args.debug_http)
    if args.registry_url is None:
        p.error("give the registry's URL with -r or NBANK_REGISTRY")
    return args.func(args)
