# -*- mode: python -*-
"""Gathers what's known about resources in an archive that aren't in the
registry (orphans), to identify them. Changes nothing.

For each orphan:

- its files, with their SHA-1 hashes and modification times, and the
  directory's own timestamps;
- for each ABF file: its Clampex number, recording date and start time,
  protocol, software, and channel names;
- the registered intracellular resources recorded on the same dates, with their
  experimenter, animal, recording name and Clampex numbers, and any files they
  share with the orphan (by hash);
- the animals used for slices on those dates, from the bird database's event
  log, and who entered them.

Writes the evidence as JSON (a list, one object per orphan) and prints a
summary of each.

Run with `uv run python migrations/identify_orphans.py`. Deleted once the
migration is done.
"""

import argparse
import datetime
import json
import sys
import zipfile
from pathlib import Path

from nbank import archive, core, registry
from normalize_abfdir import ABF_NAME, EVENTS, SliceEvents

from dlab import abfzip

DTYPES = ("intracellular-abfdir", "intracellular-abfzip")


def when(timestamp: float) -> str:
    return datetime.datetime.fromtimestamp(timestamp).isoformat(timespec="seconds")


def orphan_evidence(src: Path) -> dict:
    """What the orphan directory src holds"""
    files, abf = [], []
    for path in abfzip.directory_files(src):
        stat = (src / path).stat()
        files.append(
            {
                "path": path,
                "sha1": abfzip.file_sha1(src / path),
                "size": stat.st_size,
                "modified": when(stat.st_mtime),
            }
        )
        if not abfzip.is_abf(path):
            continue
        record = {"path": path}
        if (m := ABF_NAME.fullmatch(path.rsplit("/", 1)[-1])) is not None:
            record["number"] = int(m.group(1))
        try:
            with open(src / path, "rb") as fp:
                header = abfzip.read_header(fp)
        except ValueError as err:
            record["error"] = str(err)
        else:
            record.update(
                start=abfzip.header_to_start(header),
                protocol=abfzip.header_to_protocol(header),
                software=abfzip.header_to_software(header),
                channels=[c["name"] for c in abfzip.header_to_channels(header)]
                if "listADCInfo" in header
                else [],
            )
        abf.append(record)
    stat = src.stat()
    return {
        "id": src.name,
        "directory_modified": when(stat.st_mtime),
        "directory_changed": when(stat.st_ctime),
        "files": files,
        "abf": sorted(abf, key=lambda r: (r.get("start") or "", r["path"])),
        "dates": sorted({r["start"][:10] for r in abf if r.get("start")}),
    }


def registered_files(cfg, record: dict) -> dict[str, str]:
    """{sha1: path} of a registered resource's files: from its zip's catalog,
    or by hashing its directory"""
    path = archive.resource_path(cfg, record["name"])
    if path.is_dir():
        return {abfzip.file_sha1(path / p): p for p in abfzip.directory_files(path)}
    zipped = path.with_name(f"{record['name']}.zip")
    if zipped.exists():
        with zipfile.ZipFile(zipped) as zf:
            doc = json.loads(zf.read(abfzip.METADATA_NAME))
        return {r["sha1"]: r["path"] for r in doc["catalog"] if "sha1" in r}
    return {}


def numbers(files) -> list[int]:
    return sorted(
        int(m.group(1))
        for f in files
        if (m := ABF_NAME.fullmatch(f.rsplit("/", 1)[-1])) is not None
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("-r", dest="registry_url", default=registry.default_registry())
    p.add_argument(
        "--events", default=EVENTS, help="the bird database's event log (%(default)s)"
    )
    p.add_argument(
        "-o", "--output", required=True, help="where to write the evidence (JSON)"
    )
    p.add_argument("archive", type=Path, help="the archive's root directory")
    p.add_argument("ids", nargs="*", help="look only at these (default: every orphan)")
    args = p.parse_args(argv)
    if args.registry_url is None:
        p.error("supply a registry url with -r or NBANK_REGISTRY")
    try:
        cfg = archive.get_config(args.archive)
    except OSError as err:
        p.error(f"{args.archive}: not an archive ({err})")

    if args.ids:
        ids = args.ids
    else:
        stored = sorted(
            {q.name.split(".")[0] for q in (cfg["path"] / "resources").glob("*/*")}
        )
        known = {r["name"] for r in core.describe_many(args.registry_url, *stored)}
        ids = [id for id in stored if id not in known]
    print(f"{len(ids)} orphans", file=sys.stderr)

    records = [
        r for dtype in DTYPES for r in core.search(args.registry_url, dtype=dtype)
    ]
    by_date = {}
    for r in records:
        recording = r["metadata"].get("recording", "")
        if len(recording) >= 8:
            day = f"{recording[:4]}-{recording[4:6]}-{recording[6:8]}"
            by_date.setdefault(day, []).append(r)
    events = SliceEvents(args.events)

    out = []
    for id in ids:
        src = archive.resource_path(cfg, id)
        if not src.is_dir():
            print(f"{id}: not a directory; skipped", file=sys.stderr)
            continue
        ev = orphan_evidence(src)
        hashes = {f["sha1"] for f in ev["files"]}
        ev["registered_same_day"] = []
        for day in ev["dates"]:
            for r in by_date.get(day, []):
                theirs = registered_files(cfg, r)
                ev["registered_same_day"].append(
                    {
                        "id": r["name"],
                        "recording": r["metadata"].get("recording"),
                        "experimenter": r["experimenter"],
                        "animal": r["animal"],
                        "numbers": numbers(theirs.values()),
                        "shared_files": sorted(
                            theirs[h] for h in hashes & theirs.keys()
                        ),
                    }
                )
        ev["used_for_slices"] = {
            day: events(datetime.date.fromisoformat(day)) for day in ev["dates"]
        }
        out.append(ev)

        abf = ev["abf"]
        print(f"\n== {id}  ({len(ev['files'])} files, {len(abf)} ABF)")
        print(
            f"   dates {', '.join(ev['dates']) or '?'}; numbers {numbers(f['path'] for f in abf)}"
        )
        print(f"   software {sorted({a.get('software') or '?' for a in abf})}")
        print(f"   protocols {sorted({a.get('protocol') or '?' for a in abf})}")
        print(f"   channels {sorted({' '.join(a.get('channels', [])) for a in abf})}")
        print(f"   directory changed {ev['directory_changed']}")
        for r in ev["registered_same_day"]:
            shared = (
                f", SHARES {len(r['shared_files'])} files" if r["shared_files"] else ""
            )
            print(
                f"   same day: {r['id'][:8]} {r['recording']} {r['experimenter']} "
                f"numbers {r['numbers'][:1] + r['numbers'][-1:]}{shared}"
            )
        for day, used in ev["used_for_slices"].items():
            print(f"   used for slices {day}: {used or 'none'}")

    with open(args.output, "w") as fp:
        json.dump(out, fp, indent=1)


if __name__ == "__main__":
    main()
