# -*- mode: python -*-
"""Plans the normalization of intracellular-abfdir metadata in the registry,
before the migration to intracellular-abfzip copies it into the archives.

Writes a change file (JSON lines, one per resource: the update to send, with
null deleting a key, the record as it was, and anything needing review) and a
table of the recording names for review. Changes nothing in the registry.

- `name` becomes `recording`, of the form YYYYMMDD_SLICE_CELL. Names that
  `nbank -k` read as integers (e.g. 20140603_1_1 as 2014060311, since Python
  allows underscores as digit separators) are recovered where only one reading
  has a zero-padded date; the others are listed for review.
- `filled: "biocytin"` becomes `biocytin: true`; `rearing` becomes `condition`;
  `region` becomes `location`.
- `hemisphere` is L or R (an unknown one is deleted); `slice: "saggital"` is
  spelled `sagittal`.
- Records from 2018 without an experimenter are anc4kj's.
- The bird database's event log says which animal was used for slices on each
  recording date. A record without an animal gets that one (if several were
  used that day, the one the experimenter entered). A record whose animal was
  used a day before or after, with no animal used on the recording date, is
  accepted (a mistyped date) and noted; other records whose animal wasn't used
  that day are listed for review, with the animals other experimenters
  recorded from that day, since people cutting slices on the same day usually
  shared a bird. `biid` (the animal's identifier, entered by hand) is deleted
  where the log confirms the record's animal.

--names takes a TSV of `id` and `recording` (e.g. from analysis control files),
which is used in place of the recovered name; --animals a TSV of `id` and
`animal`, used in place of the registered animal. Other columns are ignored.

--archive takes the archive's root directory, where the resources are
readable. The table then lists each resource's files and the dates its ABF
files were recorded (from their headers); an ambiguous name is recovered if
only one reading has that date, and a recording whose date isn't one of them is
listed for review, as is one with files from any other day than its date and
the next (a session past midnight, noted). Clampex numbers the files of a session in order, so on a
day when one experimenter's file numbers run on from cell to cell (their
ranges don't overlap), a later slice or cell should have later files; ones that
don't are listed for review. Days when the numbering restarts (for each cell,
or partway through) aren't checked. Without it, nothing is read from the
archive.

Run with `uv run python migrations/normalize_abfdir.py`. Deleted once the
migration is done.
"""

import argparse
import csv
import datetime
import itertools
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import httpx
from nbank import archive, core, registry

from dlab import abfzip

DTYPE = "intracellular-abfdir"
RECORDING = re.compile(r"(\d{8})_([1-9]\d*)_([1-9]\d*)")
# the registry copies the fields into metadata for older clients
FIELDS = ("experimenter", "animal")
RENAMED = {"name": "recording", "rearing": "condition", "region": "location"}
# values of other keys, compared in lower case; None deletes the key
VALUES = {
    "hemisphere": {"l": "L", "left": "L", "r": "R", "right": "R", "unknown": None},
    "slice": {"saggital": "sagittal"},
}
EXPERIMENTER_2018 = "anc4kj"
EVENTS = "https://gracula.psyc.virginia.edu/birds/api/events/"


class SliceEvents:
    """The animals used for slices on a date, from the bird database's event
    log, as {animal: who entered the event}."""

    def __init__(self, url: str):
        self.url = url
        self.session = httpx.Client(timeout=60)
        self.cache = {}

    def __call__(self, date: datetime.date) -> dict[str, str]:
        if date not in self.cache:
            params = {"date": date.isoformat(), "status": "used for slices"}
            found, url = {}, self.url
            # paged, with the next page in the Link header
            while url is not None:
                r = self.session.get(url, params=params)
                r.raise_for_status()
                for e in r.json():
                    if e["status"] == "used for slices" and e["date"] == params["date"]:
                        found[e["animal"]] = e["entered_by"]
                url, params = r.links.get("next", {}).get("url"), None
            self.cache[date] = found
        return self.cache[date]


# Clampex's file names: the date and the file's number in the session
ABF_NAME = re.compile(r"\d{4}_\d{2}_\d{2}_(\d{4})\.abf", re.IGNORECASE)


def read_directory(root: str, id: str) -> dict:
    """The files of a resource in the archive at root, the dates its ABF files
    were recorded, and their numbers in Clampex's session, as {"files",
    "dates", "numbers", "error"}."""
    src = archive.resource_path(root, id)
    if not src.is_dir():
        return {"files": [], "dates": [], "numbers": [], "error": f"no directory {src}"}
    files = abfzip.directory_files(src)
    numbers = sorted(
        int(m.group(1))
        for f in files
        if (m := ABF_NAME.fullmatch(f.rsplit("/", 1)[-1])) is not None
    )
    dates, errors = set(), []
    for path in filter(abfzip.is_abf, files):
        with open(src / path, "rb") as fp:
            try:
                start = abfzip.header_to_start(abfzip.read_header(fp))
            except ValueError as err:
                errors.append(f"{path}: {err}")
                continue
        if start is not None:
            dates.add(start[:10].replace("-", ""))
    return {
        "files": files,
        "dates": sorted(dates),
        "numbers": numbers,
        "error": "; ".join(errors),
    }


# recordings are deposited within this many days
MAX_LAG = 400


def parse_date(text: str) -> datetime.date | None:
    try:
        return datetime.datetime.strptime(text, "%Y%m%d").date()
    except ValueError:
        return None


def readings(digits: str) -> list[str]:
    """Every YYYY M D SLICE CELL split of a string of digits with a valid date,
    in the canonical form (date zero-padded), without leading zeros in the
    slice or the cell."""
    out = []
    year, rest = digits[:4], digits[4:]
    for m in (1, 2):
        for d in (1, 2):
            for s in range(1, len(rest) - m - d):
                month, day = rest[:m], rest[m : m + d]
                slice_, cell = rest[m + d : m + d + s], rest[m + d + s :]
                if slice_[0] == "0" or cell[0] == "0":
                    continue
                if (m == 1 and month == "0") or (d == 1 and day == "0"):
                    continue
                date = parse_date(f"{year}{int(month):02d}{int(day):02d}")
                if date is not None:
                    out.append(f"{date:%Y%m%d}_{slice_}_{cell}")
    return sorted(set(out))


def recover(value, created: datetime.date) -> tuple[str | None, list[str], str]:
    """(recording, candidates, how) for a metadata name. recording is None when
    the name needs review."""
    text = str(value)
    if isinstance(value, str) and RECORDING.fullmatch(text):
        return text, [text], "kept"
    if not re.fullmatch(r"\d+", text.replace("_", "")):
        return None, [], "not a recording name"
    candidates = [
        c
        for c in readings(text.replace("_", ""))
        if 0 <= (created - parse_date(c[:8])).days <= MAX_LAG
    ]
    # the convention zero-pads the date, so a reading that keeps every digit
    # in place is the likely one
    padded = [c for c in candidates if c.replace("_", "") == text.replace("_", "")]
    if len(padded) == 1:
        return padded[0], candidates, "recovered"
    if len(candidates) == 1:
        return candidates[0], candidates, "recovered, date padded"
    return None, candidates, "ambiguous" if candidates else "no reading"


def plan(
    record: dict,
    names: dict[str, str],
    animals: dict[str, str],
    events: SliceEvents,
    abf: dict | None,
) -> dict:
    metadata = {k: v for k, v in record["metadata"].items() if k not in FIELDS}
    created = datetime.date.fromisoformat(record["created_on"][:10])
    update, review, notes = {}, [], []
    name = {
        "id": record["name"],
        "old": metadata.get("name"),
        "how": "",
        "candidates": [],
    }

    if "name" in metadata:
        recording, candidates, how = recover(metadata["name"], created)
        if recording is None and abf and len(abf["dates"]) == 1:
            dated = [c for c in candidates if c[:8] == abf["dates"][0]]
            if len(dated) == 1:
                recording, how = dated[0], "recovered, ABF date"
        if record["name"] in names:
            given = names[record["name"]]
            if recording is not None and given != recording:
                how = f"given (recovered {recording})"
            else:
                how = "given"
            recording = given
        name.update(recording=recording, candidates=candidates, how=how)
        if recording is None:
            review.append(f"name {metadata['name']!r}: {how}")
        elif not RECORDING.fullmatch(recording) or parse_date(recording[:8]) is None:
            review.append(f"recording {recording!r} isn't YYYYMMDD_SLICE_CELL")
        elif abf and abf["dates"]:
            next_day = parse_date(recording[:8]) + datetime.timedelta(days=1)
            session = {recording[:8], f"{next_day:%Y%m%d}"}
            if not session & set(abf["dates"]) or set(abf["dates"]) - session:
                review.append(f"ABF files recorded on {', '.join(abf['dates'])}")
            elif recording[:8] not in abf["dates"]:
                notes.append("ABF files recorded after midnight")
    if abf and abf["error"]:
        review.append(abf["error"])
    for old, new in RENAMED.items():
        if old not in metadata:
            continue
        value = name.get("recording") if old == "name" else metadata[old]
        if new in metadata and metadata[new] != value:
            review.append(f"{old} -> {new}: {new} is already {metadata[new]!r}")
            continue
        if value is not None:
            update[new] = value
            update[old] = None

    if "filled" in metadata:
        if metadata["filled"] == "biocytin" and metadata.get("biocytin", True) is True:
            update["biocytin"] = True
            update["filled"] = None
        else:
            review.append(f"filled {metadata['filled']!r}")

    for key, mapping in VALUES.items():
        if key not in metadata:
            continue
        value = metadata[key]
        new = mapping.get(str(value).strip().lower(), value)
        if key == "hemisphere" and new not in ("L", "R", None):
            review.append(f"hemisphere {value!r}")
        elif new != value:
            update[key] = new

    if record["experimenter"] is None:
        if created.year == 2018:
            update["experimenter"] = EXPERIMENTER_2018
        else:
            review.append("no experimenter")
    experimenter = update.get("experimenter", record["experimenter"])

    mismatch = False
    recording = name.get("recording")
    used = events(parse_date(recording[:8])) if recording else {}
    animal = record["animal"]
    if record["name"] in animals:
        animal = update["animal"] = animals[record["name"]]
        notes.append(f"animal given (registered {record['animal']})")
        if animal not in used:
            review.append(f"given animal wasn't used for slices on {recording[:8]}")
    elif animal is None:
        found = list(used)
        if len(found) > 1:
            found = [a for a in found if used[a] == experimenter]
        if len(found) == 1:
            animal = update["animal"] = found[0]
        else:
            review.append(
                f"no animal; used for slices that day: {list(used) or 'none'}"
            )
    elif recording and animal not in used:
        date = parse_date(recording[:8])
        near = [
            d
            for d in (
                date - datetime.timedelta(days=1),
                date + datetime.timedelta(days=1),
            )
            if animal in events(d)
        ]
        if near and not used:
            notes.append(f"animal used for slices on {near[0]:%Y%m%d}; accepted")
        else:
            mismatch = True
            review.append(
                f"animal wasn't used for slices on {recording[:8]} "
                f"(used: {list(used) or 'none'})"
            )

    if "biid" in metadata:
        if metadata["biid"] == animal or animal in used:
            update["biid"] = None
        else:
            review.append(f"biid {metadata['biid']} isn't the animal")
    if record["locations"] != ["intracellular"]:
        review.append(f"locations {record['locations']}")

    return {
        "id": record["name"],
        "update": update,
        "review": review,
        "notes": notes,
        "before": {
            "metadata": metadata,
            **{f: record[f] for f in FIELDS},
        },
        "_name": name,
        "_animal": animal,
        "_experimenter": experimenter,
        "_created": str(created),
        "_mismatch": mismatch,
        "_abf": abf or {"files": [], "dates": [], "numbers": []},
    }


def cell_order(p: dict) -> tuple[int, int]:
    _, slice_, cell = RECORDING.fullmatch(p["_name"]["recording"]).groups()
    return int(slice_), int(cell)


def numbers(p: dict) -> str:
    n = p["_abf"]["numbers"]
    return f"{n[0]:04d}-{n[-1]:04d}" if n else ""


def check_order(plans: list[dict]) -> None:
    """Lists for review the recordings whose ABF files aren't in the order of
    their slices and cells, on days when one experimenter's file numbers run on
    from cell to cell"""
    sessions = defaultdict(list)
    for p in plans:
        recording = p["_name"].get("recording")
        if recording and RECORDING.fullmatch(recording) and p["_abf"]["numbers"]:
            sessions[(p["_experimenter"], recording[:8])].append(p)
    for group in sessions.values():
        by_number = sorted(group, key=lambda p: p["_abf"]["numbers"][0])
        if any(
            b["_abf"]["numbers"][0] <= a["_abf"]["numbers"][-1]
            for a, b in itertools.pairwise(by_number)
        ):
            continue
        group.sort(key=lambda p: (cell_order(p), p["_abf"]["numbers"][0]))
        for a, b in itertools.pairwise(group):
            if b["_abf"]["numbers"][0] <= a["_abf"]["numbers"][-1]:
                for p, other in ((a, b), (b, a)):
                    p["review"].append(
                        f"ABF files {numbers(p)} out of order with "
                        f"{other['_name']['recording']} ({numbers(other)})"
                    )


def read_tsv(path, column: str) -> dict[str, str]:
    """{id: value of column} from a TSV with a header row; other columns (such
    as a reason) are ignored"""
    with open(path, newline="") as fp:
        return {
            row["id"].strip(): row[column].strip()
            for row in csv.DictReader(fp, delimiter="\t")
        }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("-r", dest="registry_url", default=registry.default_registry())
    p.add_argument(
        "--names", help="TSV of id and recording, used in place of the recovered names"
    )
    p.add_argument(
        "--animals", help="TSV of id and animal, used in place of the registered animal"
    )
    p.add_argument("-o", "--output", required=True, help="the change file (JSON lines)")
    p.add_argument("--table", required=True, help="the names, for review (TSV)")
    p.add_argument(
        "--archive", help="the archive's root directory, to read the resources' files"
    )
    p.add_argument(
        "--events", default=EVENTS, help="the bird database's event log (%(default)s)"
    )
    args = p.parse_args(argv)
    if args.registry_url is None:
        p.error("supply a registry url with -r or NBANK_REGISTRY")
    if args.archive is not None:
        try:
            args.archive = archive.get_config(Path(args.archive))["path"]
        except OSError as err:
            p.error(
                f"--archive {args.archive}: not an archive ({err}); give its root "
                "directory, as `nbank archive list` shows it"
            )
    names = read_tsv(args.names, "recording") if args.names else {}
    animals = read_tsv(args.animals, "animal") if args.animals else {}

    records = list(core.search(args.registry_url, dtype=DTYPE))
    events = SliceEvents(args.events)
    plans = []
    for i, r in enumerate(records):
        abf = None
        if args.archive and r["locations"] == ["intracellular"]:
            abf = read_directory(args.archive, r["name"])
            if i % 100 == 0:
                print(f"  read {i} of {len(records)}", file=sys.stderr)
        plans.append(plan(r, names, animals, events, abf))

    # two experimenters sharing an animal number slices and cells independently,
    # but one experimenter's recording names are unique for an animal
    def key(p):
        return (p["_animal"], p["_experimenter"], p["_name"].get("recording"))

    seen = Counter(key(p) for p in plans if p["_name"].get("recording"))
    for p in plans:
        if key(p)[2] and seen[key(p)] > 1:
            p["review"].append(f"recording {key(p)[2]} is also another resource's")

    check_order(plans)

    by_date = defaultdict(set)
    for p in plans:
        if p["_name"].get("recording") and p["_animal"]:
            by_date[p["_name"]["recording"][:8]].add((p["_experimenter"], p["_animal"]))
    for p in plans:
        if p["_mismatch"]:
            others = sorted(
                f"{x}: {a}"
                for x, a in by_date[p["_name"]["recording"][:8]]
                if x != p["_experimenter"]
            )
            if others:
                p["notes"].append(f"other experimenters that day: {', '.join(others)}")

    with open(args.output, "w") as fp:
        for p in plans:
            line = {k: v for k, v in p.items() if not k.startswith("_")}
            fp.write(json.dumps(line, sort_keys=True) + "\n")
    with open(args.table, "w", newline="") as fp:
        w = csv.writer(fp, delimiter="\t", lineterminator="\n")
        w.writerow(
            [
                "id",
                "experimenter",
                "animal",
                "created",
                "old",
                "recording",
                "how",
                "candidates",
                "review",
                "notes",
                "abf_dates",
                "abf_numbers",
                "files",
            ]
        )
        for p in sorted(plans, key=lambda p: (not p["review"], p["_created"], p["id"])):
            n = p["_name"]
            w.writerow(
                [
                    p["id"],
                    p["_experimenter"] or "",
                    p["_animal"] or "",
                    p["_created"],
                    "" if n["old"] is None else n["old"],
                    n.get("recording") or "",
                    n["how"],
                    " ".join(n["candidates"]),
                    "; ".join(p["review"]),
                    "; ".join(p["notes"]),
                    " ".join(p["_abf"]["dates"]),
                    numbers(p),
                    " ".join(p["_abf"]["files"]),
                ]
            )

    how = Counter(p["_name"]["how"].split(" (")[0] for p in plans)
    changed = Counter(k for p in plans for k, v in p["update"].items() if v is not None)
    print(
        f"{len(plans)} {DTYPE} resources; {sum(bool(p['review']) for p in plans)} need review",
        file=sys.stderr,
    )
    print(f"  names: {dict(how)}", file=sys.stderr)
    print(f"  keys set: {dict(changed)}", file=sys.stderr)
    for reason, n in Counter(
        re.sub(r"[\"'].*?[\"']|\d{8}_\d+_\d+|\d{6,}|/\S+|\d{4}-\d{4}", "…", r)
        for p in plans
        for r in p["review"]
    ).most_common():
        print(f"  review: {reason} ({n})", file=sys.stderr)


if __name__ == "__main__":
    main()
