# -*- mode: python -*-
"""Explains intracellular-abfdir resources whose directories don't match their
registered hashes. Changes nothing.

neurobank has hashed directories three ways. Each is the hash of lines
`<path>=<file sha1>` joined with newlines:

- `relative`: paths relative to the directory, sorted by component (since
  2025-07-03, nbank 0.11.0; also what earlier versions computed for a
  directory without subdirectories, up to 2023-02-13)
- `string-sorted`: relative paths sorted as strings (to 2023-02-13), which can
  order a directory with subdirectories differently
- `absolute`: the absolute path of each file on the computer that deposited it
  (2023-02-13 to 2025-07-03, nbank 0.8.0 to 0.10.x), so the hash depends on
  where the directory was. --prefix gives candidates for the directory that
  held it, and the directory's own name is tried as the registry's recording
  name with and without underscores, and as the resource id

Each is also tried without the files modified after the resource was
registered (e.g. images added later). For each resource the output gives the
first explanation that reproduces its hash, or none.

Run with `uv run python migrations/diagnose_abfdir_hashes.py`. Deleted once
the migration is done.
"""

import argparse
import csv
import datetime
import hashlib
import sys
from pathlib import Path, PurePosixPath

from nbank import archive, core, registry

from dlab import abfzip

DTYPE = "intracellular-abfdir"


def combine(lines: list[str]) -> str:
    return hashlib.sha1("\n".join(lines).encode("utf-8")).hexdigest()


def relative(files: dict[str, str]) -> str:
    paths = sorted(files, key=lambda p: PurePosixPath(p).parts)
    return combine([f"{p}={files[p]}" for p in paths])


def string_sorted(files: dict[str, str]) -> str:
    return combine([f"{p}={files[p]}" for p in sorted(files)])


def absolute(files: dict[str, str], directory: str) -> str:
    paths = sorted(files, key=lambda p: PurePosixPath(p).parts)
    return combine([f"{directory}/{p}={files[p]}" for p in paths])


def directory_names(record: dict) -> list[str]:
    """What the deposited directory may have been called"""
    names = [record["name"]]
    recording = record["metadata"].get("recording")
    if recording:
        names += [recording, recording.replace("_", "")]
    return list(dict.fromkeys(names))


def explain(record: dict, files: dict[str, str], late: set[str], prefixes) -> str:
    """The first explanation that reproduces the record's hash, or ''"""
    sets = [("", files)]
    if late:
        sets.append(
            (" without later files", {p: h for p, h in files.items() if p not in late})
        )
    for label, subset in sets:
        if relative(subset) == record["sha1"]:
            return "relative" + label
        if string_sorted(subset) == record["sha1"]:
            return "string-sorted" + label
        for prefix in prefixes:
            for name in directory_names(record):
                directory = f"{prefix.rstrip('/')}/{name}"
                if absolute(subset, directory) == record["sha1"]:
                    return f"absolute {directory}" + label
    return ""


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("-r", dest="registry_url", default=registry.default_registry())
    p.add_argument(
        "--prefix",
        action="append",
        default=[],
        help="a directory deposits may have been made from (repeat for more)",
    )
    p.add_argument(
        "-o", "--output", required=True, help="where to write the table (TSV)"
    )
    p.add_argument("archive", type=Path, help="the archive's root directory")
    p.add_argument("ids", nargs="*", help="check only these resources")
    args = p.parse_args(argv)
    if args.registry_url is None:
        p.error("supply a registry url with -r or NBANK_REGISTRY")
    try:
        cfg = archive.get_config(args.archive)
    except OSError as err:
        p.error(f"{args.archive}: not an archive ({err})")

    records = list(core.search(args.registry_url, dtype=DTYPE))
    if args.ids:
        records = [r for r in records if r["name"] in set(args.ids)]
    rows = []
    for i, record in enumerate(records, 1):
        src = archive.resource_path(cfg, record["name"])
        if not src.is_dir():
            continue
        registered = datetime.datetime.fromisoformat(record["created_on"]).timestamp()
        files, late = {}, set()
        for path in abfzip.directory_files(src):
            files[path] = abfzip.file_sha1(src / path)
            if (src / path).stat().st_mtime > registered:
                late.add(path)
        if relative(files) == record["sha1"]:
            continue
        rows.append(
            {
                "id": record["name"],
                "experimenter": record["experimenter"] or "",
                "created": record["created_on"][:10],
                "files": len(files),
                "later_files": "|".join(sorted(late)),
                "explanation": explain(record, files, late, args.prefix) or "none",
            }
        )
        if i % 100 == 0:
            print(f"  checked {i} of {len(records)}", file=sys.stderr)

    with open(args.output, "w", newline="") as fp:
        w = csv.DictWriter(
            fp,
            ["id", "experimenter", "created", "files", "later_files", "explanation"],
            delimiter="\t",
            lineterminator="\n",
        )
        w.writeheader()
        w.writerows(rows)
    counts = {}
    for row in rows:
        key = row["explanation"].split(" /")[0].split(" ")[0]
        key += (
            " without later files"
            if "without later files" in row["explanation"]
            else ""
        )
        counts[key] = counts.get(key, 0) + 1
    print(f"{len(rows)} of {len(records)} don't match their hashes:", file=sys.stderr)
    for key, n in sorted(counts.items()):
        print(f"  {key}: {n}", file=sys.stderr)


if __name__ == "__main__":
    main()
