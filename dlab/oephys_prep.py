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

Then arfx-oephys writes the recording to `<name>.arf` beside the session
directory, the registry checks the ARF as it would be deposited, and the ARF
and the session directory are copied to the target (the spike sorting
workstation's share) with `<name>.json`, which records the ARF's hash and
metadata and where they came from. The ARF is deposited from the target once
the sort is curated.

"""

import argparse
import datetime
import json
import logging
import re
import shlex
import shutil
import socket
import subprocess
from collections.abc import Callable
from dataclasses import astuple, dataclass
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
FIELDS = ("pen", "site", "probe", "x", "y", "z")
REQUIRED = ("pen", "site")
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
            "can only hold one, so move the others to their own session directories"
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
        values[field] = ask(
            input, field, defaults.get(field), parse=parse, required=field in REQUIRED
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


def arfx_command(
    session: Session,
    values: dict[str, Any],
    revised: list[str],
    presenter: dict[str, Any],
    arffile: Path,
) -> list[str]:
    """The arfx-oephys command that writes the recording to arffile with the
    values. Values are JSON, so that a string that looks like a number stays one."""
    cmd = [
        "arfx-oephys",
        "-f",
        str(arffile),
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


@dataclass
class Outputs:
    """The files prep-oephys writes: the ARF beside the session directory, and
    in the target, the ARF, the session directory, and the manifest"""

    arf: Path
    target_arf: Path
    target_session: Path
    manifest: Path

    @classmethod
    def for_recording(cls, recording: Session, name: str, target: Path) -> "Outputs":
        return cls(
            arf=recording.path.parent / f"{name}.arf",
            target_arf=target / f"{name}.arf",
            target_session=target / recording.path.name,
            manifest=target / f"{name}.json",
        )

    def existing(self) -> list[Path]:
        return [p for p in astuple(self) if p.exists()]


def convert(cmd: list[str], arffile: Path) -> None:
    """Runs arfx-oephys. Raises RuntimeError if it fails, after removing what it
    wrote."""
    log.info("converting: %s", shlex.join(cmd))
    try:
        result = subprocess.run(cmd)
    except FileNotFoundError as err:
        raise RuntimeError(
            "arfx-oephys isn't installed: install melizalab-tools[oephys]"
        ) from err
    if result.returncode != 0:
        arffile.unlink(missing_ok=True)
        raise RuntimeError(f"arfx-oephys failed (exit status {result.returncode})")


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


def copy_to_target(recording: Session, outputs: Outputs, sha1: str) -> None:
    """Copies the session directory and the ARF to the target, checking each
    file's size, and the ARF's hash. Raises OSError if a copy fails or differs,
    after removing the copies."""
    try:
        log.info("copying %s to %s", recording.path, outputs.target_session)
        shutil.copytree(recording.path, outputs.target_session)
        for src in recording.path.rglob("*"):
            dst = outputs.target_session / src.relative_to(recording.path)
            if src.is_file() and src.stat().st_size != dst.stat().st_size:
                raise OSError(f"{dst}: the copy isn't the same size as {src}")
        log.info("copying %s to %s", outputs.arf, outputs.target_arf)
        shutil.copyfile(outputs.arf, outputs.target_arf)
        if file_sha1(outputs.target_arf) != sha1:
            raise OSError(f"{outputs.target_arf}: the copy doesn't match {outputs.arf}")
    except BaseException:
        shutil.rmtree(outputs.target_session, ignore_errors=True)
        outputs.target_arf.unlink(missing_ok=True)
        raise


def write_manifest(recording: Session, outputs: Outputs, resource: dict) -> None:
    """Writes the manifest: the resource as it will be registered, and where its
    ARF and session directory came from"""
    manifest = {
        **resource,
        "source": {
            "host": socket.gethostname(),
            "arf": str(outputs.arf.resolve()),
            "session": str(recording.path.resolve()),
        },
        "session": outputs.target_session.name,
        "prepared_by": f"{PROG} {__version__}",
        "prepared_on": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with open(outputs.manifest, "w") as fp:
        json.dump(manifest, fp, indent=2)
        fp.write("\n")


def script(argv=None):
    from nbank import registry, util

    from dlab.neurobank import default_auth

    p = argparse.ArgumentParser(
        prog=PROG,
        description="Confirm an open-ephys recording's metadata and check them "
        "with the registry, for spike sorting",
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
    p.add_argument(
        "--colony",
        metavar="URL",
        help="URL of the bird colony's API (default: birds/api/ on the "
        "registry's server)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="confirm the metadata and show the arfx-oephys command, but don't "
        "write or copy anything",
    )
    p.add_argument("path", type=Path, help="the session directory")
    p.add_argument(
        "target",
        type=Path,
        help="the directory to copy the ARF and the session directory to",
    )
    args = p.parse_args(argv)
    setup_log(args.debug, args.debug_http)
    if args.registry_url is None:
        p.error("give the registry's URL with -r or NBANK_REGISTRY")
    if not args.target.is_dir():
        p.error(f"{args.target} isn't a directory")
    try:
        recording = find_session(args.path)
    except ValueError as err:
        p.error(str(err))
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
        outputs = Outputs.for_recording(recording, values["name"], args.target)
        existing = [p for p in outputs.existing() if p != outputs.arf]
        if existing:
            log.error("error: already there: %s", ", ".join(map(str, existing)))
            return 1
        # an ARF from a run whose copy failed is checked and copied again
        reuse = outputs.arf.exists()
        cmd = arfx_command(recording, values, revised, presenter, outputs.arf)
        if args.dry_run:
            if reuse:
                log.info("%s is already written and won't be again", outputs.arf)
            print(shlex.join(cmd))
            return 0
        try:
            if reuse:
                log.info("%s is already written; checking it", outputs.arf)
            else:
                convert(cmd, outputs.arf)
            resource = check_arf(session, args.registry_url, outputs.arf, values)
        except (RuntimeError, ValueError) as err:
            if reuse:
                log.error("error: %s; remove it to convert the recording again", err)
            else:
                # an ARF the registry would refuse has to be written again
                outputs.arf.unlink(missing_ok=True)
                log.error("error: %s", err)
            return 1
    try:
        copy_to_target(recording, outputs, resource["sha1"])
    except OSError as err:
        log.error("error: %s; run again to copy %s", err, outputs.arf)
        return 1
    write_manifest(recording, outputs, resource)
    log.info(
        "%s is ready for sorting in %s; deposit %s once the sort is curated",
        values["name"],
        args.target,
        outputs.target_arf.name,
    )
    return 0
