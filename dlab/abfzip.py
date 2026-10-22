# -*- mode: python -*-
"""Make abfzip archives of intracellular recordings (abf-zip)

An abfzip archive (https://meliza.org/spec:4/abfzip/) holds the ABF files of
one recording, with a metadata.json member that describes the recording
(`metadata`, registered on deposit) and its files (`catalog`: the SHA-1 hash of
every file, and the acquisition protocol, start time, and settings of each ABF
file, from its header). The recording's metadata includes the software that
wrote the files, which must be the same for all of them, and abf-zip's name
and version as `processed_by`.

Everything in metadata.json is fixed by the archive's hash once it's built, so
abf-zip works in two steps: `abf-zip catalog` writes metadata.json beside the
recording's directory for review, and `abf-zip build` builds the archive from
the directory and the reviewed file. The archive is written in neurobank's
canonical form, so the same files always give the same archive.

"""

import argparse
import datetime
import hashlib
import json
import logging
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, BinaryIO

from dlab import __version__
from dlab.pprox import CREATOR_PREFIX
from dlab.util import add_log_arguments, setup_log

log = logging.getLogger("dlab")

PROG = "abf-zip"
SCHEMA = "https://meliza.org/spec:4/abfzip.json#"
VERSION = "1.0"
METADATA_NAME = "metadata.json"
PROCESSED_BY = f"{CREATOR_PREFIX}{PROG} {__version__}"
# the size of a block in an ABF file
ABF_BLOCK_SIZE = 512
# nOperationMode
OPERATION_MODES = {
    1: "event-variable",
    2: "event-fixed",
    3: "gap-free",
    4: "oscilloscope",
    5: "episodic",
}
EPISODIC = 5
# nTelegraphMode, when the amplifier's telegraphs are enabled (0 and 1 agree
# with the channel units in the archive; 2 is presumed)
TELEGRAPH_MODES = {0: "voltage", 1: "current", 2: "zero current"}


def is_abf(path: str) -> bool:
    """True if path names an ABF file (by its extension, in any case)"""
    return path.lower().endswith(".abf")


def read_header(fp: BinaryIO) -> dict[str, Any]:
    """The header of an ABF file, open for reading in binary mode, as neo
    parses it. Raises ValueError if it isn't a valid ABF header."""
    # neo's parse_axon_soup only takes a filename; these are the functions it
    # calls with the open file
    from neo.rawio import axonrawio

    f = axonrawio.StructFile(fp)
    signature = f.read(4)
    try:
        if signature == b"ABF ":
            return axonrawio._parse_abf_v1(f, axonrawio.headerDescriptionV1)
        if signature == b"ABF2":
            header = axonrawio._parse_abf_v2(f, axonrawio.headerDescriptionV2)
            # neo doesn't keep the name of the program that wrote the file
            strings = _read_strings(f, header)
            index = header["uCreatorNameIndex"]
            header["sCreatorName"] = strings[index] if 0 < index < len(strings) else ""
            return header
    except Exception as err:
        # a damaged header can fail in many ways
        raise ValueError(f"not a valid ABF header ({err!r})") from err
    raise ValueError(f"not an ABF file (starts with {signature!r})")


def _read_strings(f: BinaryIO, header: dict[str, Any]) -> list[str]:
    """The strings section of an ABF2 file, split as neo splits it"""
    section = header["sections"]["StringsSection"]
    f.seek(section["uBlockIndex"] * ABF_BLOCK_SIZE)
    data = f.read(section["uBytes"])
    data = data[data.rfind(b"\x00\x00") :]
    return [s.decode("utf-8", errors="replace") for s in data.split(b"\x00")[1:]]


def header_to_protocol(header: dict[str, Any]) -> str | None:
    """The name of the acquisition protocol in an ABF header: the stem of the
    pCLAMP protocol file (a Windows path), or None if there isn't one"""
    name = re.split(r"[\\/]", header.get("sProtocolPath", ""))[-1]
    name = re.sub(r"\.pro$", "", name, flags=re.IGNORECASE)
    return name or None


def header_to_start(header: dict[str, Any]) -> str | None:
    """When recording of an ABF file started, from its header, as an ISO 8601
    local time with milliseconds, or None if the header has no date"""
    if header.get("rec_datetime") is None:
        return None
    if "uFileStartDate" in header:
        # ABF2: the date and the time in ms (neo's datetime has float error)
        date = datetime.datetime.strptime(str(header["uFileStartDate"]), "%Y%m%d")
        start = date + datetime.timedelta(milliseconds=int(header["uFileStartTimeMS"]))
    else:
        start = header["rec_datetime"]
    return start.isoformat(timespec="milliseconds")


def header_to_software(header: dict[str, Any]) -> str | None:
    """The program that wrote an ABF2 file and its version, as in "Clampex
    11.2.2.17", or None if the header doesn't say"""
    name = header.get("sCreatorName", "").strip()
    version = header.get("uCreatorVersion", 0)
    if not name:
        return None
    if not version:
        return name
    # one byte each for major, minor, bugfix, and build
    return name + " " + ".".join(str((version >> s) & 0xFF) for s in (24, 16, 8, 0))


def header_to_channels(header: dict[str, Any]) -> list[dict[str, Any]]:
    """The recorded (ADC) channels of an ABF2 file, in order: name, units,
    and the clamp mode if the amplifier's telegraphs were enabled"""
    channels = []
    for info in header["listADCInfo"]:
        channel = {
            "name": info["ADCChNames"].decode("utf-8", errors="replace").strip(),
            "units": info["ADCChUnits"].decode("utf-8", errors="replace").strip(),
        }
        if info["nTelegraphEnable"] and info["nTelegraphMode"] in TELEGRAPH_MODES:
            channel["clamp"] = TELEGRAPH_MODES[info["nTelegraphMode"]]
        channels.append(channel)
    return channels


def header_to_settings(header: dict[str, Any]) -> dict[str, Any]:
    """The acquisition settings in an ABF2 header (none for ABF1): the
    operation mode, the sampling rate (Hz, per channel), the number of sweeps
    (episodic mode), and the channels"""
    if "protocol" not in header:
        return {}
    settings = {}
    mode = header["protocol"]["nOperationMode"]
    if mode in OPERATION_MODES:
        settings["mode"] = OPERATION_MODES[mode]
    interval = header["protocol"]["fADCSequenceInterval"]
    if interval > 0:
        rate = 1e6 / float(interval)
        settings["sampling_rate"] = int(rate) if rate.is_integer() else rate
    if mode == EPISODIC:
        settings["sweeps"] = int(header["lActualEpisodes"])
    settings["channels"] = header_to_channels(header)
    return settings


def header_to_record(header: dict[str, Any], path: str) -> dict[str, Any]:
    """The catalog record for the ABF file at path in the archive"""
    record = {"path": path}
    protocol = header_to_protocol(header)
    if protocol is not None:
        record["protocol"] = protocol
    start = header_to_start(header)
    if start is not None:
        record["start"] = start
    record.update(header_to_settings(header))
    return record


def headers_to_software(headers: Iterable[tuple[str, dict[str, Any]]]) -> str | None:
    """The software that wrote the ABF files with headers (path, header), for
    the recording's metadata, or None if no header says. Raises ValueError if
    the files were written by different software."""
    software = {}
    for path, header in headers:
        value = header_to_software(header)
        if value is not None:
            software.setdefault(value, path)
    if len(software) > 1:
        raise ValueError(
            "the files were written by different software: "
            + ", ".join(f"{value} ({path})" for value, path in software.items())
        )
    return next(iter(software), None)


def directory_files(src: Path) -> list[str]:
    """The files under src, as paths relative to it with '/' separators"""
    return sorted(p.relative_to(src).as_posix() for p in src.rglob("*") if p.is_file())


def directory_to_headers(src: Path) -> list[tuple[str, dict[str, Any]]]:
    """The path and header of each ABF file under src. Raises ValueError if
    there are none, or one isn't a valid ABF file."""
    headers = []
    for path in directory_files(src):
        if not is_abf(path):
            continue
        with open(src / path, "rb") as fp:
            try:
                headers.append((path, read_header(fp)))
            except ValueError as err:
                raise ValueError(f"{path}: {err}") from err
    if not headers:
        raise ValueError(f"no ABF files in {src}")
    return headers


def headers_to_catalog(
    headers: Iterable[tuple[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """A catalog record for each ABF file with headers (path, header), in the
    order they were recorded"""
    records = [header_to_record(header, path) for path, header in headers]
    return sorted(records, key=lambda r: (r.get("start", ""), r["path"]))


def file_sha1(path: Path) -> str:
    """The SHA-1 hash of the file at path, in lowercase hexadecimal"""
    with open(path, "rb") as fp:
        return hashlib.file_digest(fp, "sha1").hexdigest()


def files_to_catalog(
    src: Path, headers: Iterable[tuple[str, dict[str, Any]]]
) -> list[dict[str, Any]]:
    """A catalog record with its SHA-1 hash for every file under src: the ABF
    files with headers (path, header) in the order they were recorded, then
    the others by path"""
    records = headers_to_catalog(headers)
    abf_paths = {r["path"] for r in records}
    records += [
        {"path": path} for path in directory_files(src) if path not in abf_paths
    ]
    for record in records:
        record["sha1"] = file_sha1(src / record["path"])
    return records


def directory_to_catalog(src: Path) -> list[dict[str, Any]]:
    """A catalog record for every file under src (see files_to_catalog).
    Raises ValueError if there are no ABF files, or one isn't valid."""
    return files_to_catalog(src, directory_to_headers(src))


def metadata_document(
    metadata: dict[str, Any], catalog: Iterable[dict[str, Any]]
) -> dict[str, Any]:
    """The contents of metadata.json"""
    return {
        "$schema": SCHEMA,
        "version": VERSION,
        "metadata": dict(metadata),
        "catalog": list(catalog),
    }


def check_metadata(doc: Any, members: Iterable[str]) -> None:
    """Checks a metadata.json document for an archive with the members (paths
    of files), as a reader must: the schema and the major version, then the
    whole document against the schema, then that each catalog path names one
    of the members and appears only once. Raises ValueError if it fails."""
    import jsonschema
    import melizalab_specs

    if not isinstance(doc, dict):
        raise ValueError(f"{METADATA_NAME} isn't a JSON object")
    if doc.get("$schema") != SCHEMA:
        raise ValueError(
            f"{METADATA_NAME} has $schema {doc.get('$schema')!r}, not {SCHEMA!r}"
        )
    version = doc.get("version")
    if not isinstance(version, str) or version.split(".")[0] != VERSION.split(".")[0]:
        raise ValueError(
            f"{METADATA_NAME} has version {version!r}; only {VERSION.split('.')[0]}.x "
            "is known"
        )
    try:
        melizalab_specs.validate(doc)
    except jsonschema.ValidationError as err:
        where = "/".join(str(p) for p in err.absolute_path) or "(top level)"
        raise ValueError(
            f"{METADATA_NAME} is invalid at {where}: {err.message}"
        ) from err
    members = set(members)
    seen = set()
    for record in doc["catalog"]:
        path = record["path"]
        if path not in members:
            raise ValueError(
                f"{METADATA_NAME} lists {path!r}, which isn't in the archive"
            )
        if path in seen:
            raise ValueError(f"{METADATA_NAME} lists {path!r} more than once")
        seen.add(path)


def metadata_to_bytes(doc: dict[str, Any]) -> bytes:
    """metadata.json as it's stored in the archive"""
    return (json.dumps(doc, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def build(src: Path, doc: dict[str, Any], out: Path) -> Path:
    """Builds the abfzip archive of the directory src with doc as its
    metadata.json, at out. Raises ValueError if doc doesn't check, src has no
    ABF files or has its own metadata.json, and the errors of
    nbank.canonical_zip.zip_directory."""
    from nbank.canonical_zip import zip_directory

    files = directory_files(src)
    if METADATA_NAME in files:
        raise ValueError(
            f"{src} has a {METADATA_NAME} of its own; move it out of the directory"
        )
    if not any(is_abf(path) for path in files):
        raise ValueError(f"no ABF files in {src}")
    check_metadata(doc, files)
    for record in doc["catalog"]:
        if "sha1" in record and file_sha1(src / record["path"]) != record["sha1"]:
            raise ValueError(f"{record['path']} has changed since it was cataloged")
    return zip_directory(src, out, extra=[(METADATA_NAME, metadata_to_bytes(doc))])


def catalog_command(args) -> None:
    src = args.directory
    out = args.output or src.parent / f"{src.name}.json"
    if out.exists():
        raise FileExistsError(f"{out} already exists")
    metadata = dict(args.metadata)
    if "processed_by" in metadata:
        raise ValueError(f"processed_by is set by {PROG}; don't give it with -k")
    for key in ("experimenter", "animal"):
        value = getattr(args, key)
        if value is not None:
            metadata[key] = value
    if any(value is None for value in metadata.values()):
        raise ValueError("metadata values can't be None; leave the key out")
    log.info("- cataloging ABF files in %s", src)
    headers = directory_to_headers(src)
    software = headers_to_software(headers)
    catalog = files_to_catalog(src, headers)
    for record in filter(lambda r: is_abf(r["path"]), catalog):
        log.info(
            "  - %s: %s, %s",
            record["path"],
            record.get("protocol", "(no protocol)"),
            record.get("start", "(no start time)"),
        )
    if "software" in metadata:
        if metadata["software"] != software:
            log.warning(
                "- software is %r in the files; using %r",
                software,
                metadata["software"],
            )
    elif software is not None:
        metadata["software"] = software
    metadata["processed_by"] = PROCESSED_BY
    doc = metadata_document(metadata, catalog)
    check_metadata(doc, directory_files(src))
    out.write_bytes(metadata_to_bytes(doc))
    log.info("- wrote %s; review it, then run %s build", out, PROG)


def build_command(args) -> None:
    src = args.directory
    out = args.output or src.parent / f"{src.name}.zip"
    doc = json.loads(args.metadata.read_bytes())
    log.info("- building %s from %s and %s", out, src, args.metadata)
    build(src, doc, out)
    log.info("- wrote %s", out)


def script(argv=None):
    from nbank.script import ParseKeyVal

    p = argparse.ArgumentParser(
        prog=PROG,
        description="Make an abfzip archive of the ABF files of an intracellular "
        "recording: catalog the files for review, then build the archive.",
    )
    p.add_argument(
        "-v", "--version", action="version", version=f"%(prog)s {__version__}"
    )
    add_log_arguments(p)
    sub = p.add_subparsers(title="commands", required=True)

    pp = sub.add_parser(
        "catalog",
        help="write metadata.json for a recording's directory, for review",
    )
    pp.set_defaults(func=catalog_command)
    pp.add_argument(
        "--experimenter",
        metavar="USER",
        help="the person who made the recording (a registry username)",
    )
    pp.add_argument(
        "--animal",
        metavar="ID",
        help="the animal the recording came from (its full identifier, a "
        "lowercase UUID)",
    )
    pp.add_argument(
        "-k",
        help="a metadata field for the recording (repeat for more)",
        action=ParseKeyVal,
        default={},
        metavar="KEY=VALUE",
        dest="metadata",
    )
    pp.add_argument(
        "-o",
        "--output",
        type=Path,
        help="where to write metadata.json (default: <directory>.json beside it)",
    )
    pp.add_argument("directory", type=Path, help="the recording's directory")

    pp = sub.add_parser(
        "build", help="build the archive from the directory and reviewed metadata"
    )
    pp.set_defaults(func=build_command)
    pp.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="the reviewed metadata.json (from abf-zip catalog)",
    )
    pp.add_argument(
        "-o",
        "--output",
        type=Path,
        help="where to write the archive (default: <directory>.zip beside it)",
    )
    pp.add_argument("directory", type=Path, help="the recording's directory")

    args = p.parse_args(argv)
    setup_log(args.debug, args.debug_http)
    log.info("- %s version %s", PROG, __version__)
    args.func(args)
