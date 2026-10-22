# -*- mode: python -*-
"""Resolves the orphans in an archive (resources not in the registry), as
abfdir-orphans.tsv says.

- `remove`: the orphan is a copy of a registered resource (`twin`), left
  behind when the registry refused a second deposit of the same files. It's
  removed after checking that the twin is registered, the orphan isn't, and
  the two directories hold the same files, byte for byte.
- `register`: the orphan is a recording that was never registered. It's
  registered under its own id as intracellular-abfdir, with the directory's
  hash and normalized metadata (`recording`, `location: CM`, experimenter and
  animal), after checking that it isn't registered and that no resource has
  its hash.

--dry-run checks each row and changes nothing.

Run with `uv run python migrations/resolve_orphans.py`. Deleted once the
migration is done.
"""

import argparse
import csv
import sys
from pathlib import Path

import httpx
from nbank import archive, core, registry, util

from dlab import abfzip

DTYPE = "intracellular-abfdir"
LOCATION = "CM"


def file_hashes(src: Path) -> dict[str, str]:
    return {p: abfzip.file_sha1(src / p) for p in abfzip.directory_files(src)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("-r", dest="registry_url", default=registry.default_registry())
    p.add_argument(
        "-a", dest="auth", help="username:password for the registry (default: .netrc)"
    )
    p.add_argument("--dry-run", action="store_true", help="check, but change nothing")
    p.add_argument("archive", type=Path, help="the archive's root directory")
    p.add_argument("orphans", type=Path, help="the decisions (abfdir-orphans.tsv)")
    args = p.parse_args(argv)
    if args.registry_url is None:
        p.error("supply a registry url with -r or NBANK_REGISTRY")
    try:
        cfg = archive.get_config(args.archive)
    except OSError as err:
        p.error(f"{args.archive}: not an archive ({err})")
    auth = tuple(args.auth.split(":", 1)) if args.auth else None
    session = httpx.Client(auth=core.make_auth(auth), timeout=60)
    url, params = registry.find_archive_by_path(args.registry_url, cfg["path"])
    archive_name = util.query_registry_first(session, url, params)["name"]

    with open(args.orphans, newline="") as fp:
        rows = list(csv.DictReader(fp, delimiter="\t"))
    failed = 0
    for row in rows:
        id = row["id"]
        src = archive.resource_path(cfg, id)
        try:
            if not src.is_dir():
                raise ValueError(f"no directory at {src}")
            if core.describe(args.registry_url, id) is not None:
                raise ValueError("it's registered")
            if row["action"] == "remove":
                twin = core.describe(args.registry_url, row["twin"])
                if twin is None:
                    raise ValueError(f"its twin {row['twin']} isn't registered")
                twin_src = archive.resource_path(cfg, row["twin"])
                if not twin_src.is_dir():
                    raise ValueError(f"its twin isn't a directory at {twin_src}")
                if file_hashes(src) != file_hashes(twin_src):
                    raise ValueError(f"its files aren't the same as {row['twin']}'s")
                if not args.dry_run:
                    archive.remove(src)
                print(
                    f"{id}: {'would remove' if args.dry_run else 'removed'} (copy of {row['twin']})"
                )
            elif row["action"] == "register":
                sha1 = util.hash_directory(src)
                url, params = registry.find_resource(args.registry_url, sha1=sha1)
                if util.query_registry_first(session, url, params) is not None:
                    raise ValueError(f"a resource already has its hash {sha1}")
                metadata = {
                    "recording": row["recording"],
                    "location": LOCATION,
                    "experimenter": row["experimenter"],
                    "animal": row["animal"],
                }
                if not args.dry_run:
                    url, data = registry.add_resource(
                        args.registry_url, id, DTYPE, archive_name, sha1, **metadata
                    )
                    r = session.post(url, json=data)
                    if r.status_code >= 400:
                        raise ValueError(f"the registry refused it: {r.text}")
                print(
                    f"{id}: {'would register' if args.dry_run else 'registered'} as "
                    f"{row['recording']} ({row['experimenter']}, {row['animal'][:8]}), sha1 {sha1}"
                )
            else:
                raise ValueError(f"unknown action {row['action']!r}")
        except ValueError as err:
            failed += 1
            print(f"{id}: not done: {err}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
