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

"""

import argparse
import datetime
import json
import logging
import re
import shlex
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx

from dlab import __version__
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
# the fields that check_with_registry reports errors for
CHECKED = ("name", "experimenter", "animal", "metadata")

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


def check_with_registry(session: httpx.Client, registry_url: str, body: dict) -> list:
    """The registry's objections to the name, experimenter, animal or metadata of
    the resource, as messages"""
    from nbank import registry

    url, bodies = registry.check_resources(registry_url, [body])
    r = session.post(url, json=bodies)
    r.raise_for_status()
    [errors] = r.json()
    return registry.field_messages({k: v for k, v in errors.items() if k in CHECKED})


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
    p.add_argument("path", type=Path, help="the session directory")
    args = p.parse_args(argv)
    setup_log(args.debug, args.debug_http)
    if args.registry_url is None:
        p.error("give the registry's URL with -r or NBANK_REGISTRY")
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
    cmd = arfx_command(
        recording, values, revised, presenter, Path(f"{values['name']}.arf")
    )
    print(shlex.join(cmd))
