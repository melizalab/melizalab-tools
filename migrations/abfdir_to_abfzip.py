# -*- mode: python -*-
"""Migrates the intracellular-abfdir resources in one archive to
intracellular-abfzip.

For each resource, in this order:

1. build `<id>.zip` in the archive's staging directory with abf-zip's builder:
   `metadata` is the registry record's (experimenter, animal and metadata) plus
   the ABF files' `software` and this script as `processed_by`; the catalog has
   a record for every file, with its SHA-1 hash, and the ABF files' headers;
2. check the zip: it passes the intracellular-abfzip reader, carries the
   registry's metadata, has a record for every member, each member matches its
   record's hash, and the records' hashes give the directory's registered hash
   (neurobank hashes a directory from its files' hashes), which ties the zip to
   the resource it replaces;
3. move it into the archive as `<id>.zip`;
4. change the resource's sha1 and dtype in the registry, and add `software` and
   `processed_by` to its metadata, in one request (this needs a superuser);
5. remove the directory.

The directory stays until the registry knows the zip, so the data is on disk at
every step. A resource whose zip can't be built or doesn't check (including a
directory that no longer matches its hash) is reported and skipped. An error while storing,
updating the registry or removing stops the run. A resource interrupted after
step 4 is finished from where it stopped when the script is run again, and one
already migrated is skipped. --dry-run builds and checks each zip in staging,
then deletes it, and changes nothing.

Every resource's outcome is logged as a JSON line.

Run with `uv run python migrations/abfdir_to_abfzip.py`. Deleted once the
migration is done.
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import zipfile
from pathlib import Path
from typing import Any

import httpx
from nbank import archive, core, registry, util

from dlab import __version__, abfzip, readers
from dlab.pprox import CREATOR_PREFIX

log = logging.getLogger("abfdir-to-abfzip")

DTYPE_FROM = "intracellular-abfdir"
DTYPE_TO = "intracellular-abfzip"
FIELDS = ("experimenter", "animal")
PROCESSED_BY = f"{CREATOR_PREFIX}abfdir-to-abfzip {__version__}"
# the metadata the migration adds to the registry's
ADDED = ("software", "processed_by")


class Skip(Exception):
    """A problem with one resource; the run goes on to the next"""


class Registry:
    """The registry calls the migration makes"""

    def __init__(self, url: str, auth=None):
        self.url = url
        self.session = httpx.Client(auth=core.make_auth(auth), timeout=60)

    def archive_name(self, path: Path) -> str:
        url, params = registry.find_archive_by_path(self.url, path)
        found = util.query_registry_first(self.session, url, params)
        if found is None:
            raise RuntimeError(f"archive '{path}' isn't in the registry")
        return found["name"]

    def resources(self, dtype: str) -> list[dict]:
        return list(core.search(self.url, dtype=dtype))

    def get(self, id: str) -> dict:
        return core.describe(self.url, id)

    def migrate(self, id: str, sha1: str, metadata: dict) -> dict:
        data = {"sha1": sha1, "dtype": DTYPE_TO, "metadata": metadata}
        r = self.session.patch(registry.full_url(self.url, id), json=data)
        if r.status_code >= 400:
            raise RuntimeError(f"{id}: the registry refused the update: {r.text}")
        return r.json()


def sha1_of(fp) -> str:
    h = hashlib.sha1()
    for chunk in iter(lambda: fp.read(1 << 20), b""):
        h.update(chunk)
    return h.hexdigest()


def registry_metadata(record: dict) -> dict[str, Any]:
    """The metadata a record's zip carries: its metadata, with experimenter
    and animal from its fields (the registry also copies them into metadata
    for older clients)"""
    metadata = {k: v for k, v in record["metadata"].items() if k not in FIELDS}
    for field in FIELDS:
        if record[field] is not None:
            metadata[field] = record[field]
    return metadata


def build_zip(src: Path, record: dict, out: Path) -> None:
    """Builds the zip of the directory src for the registry record at out.
    Raises Skip if it can't be built."""
    metadata = registry_metadata(record)
    if "processed_by" in metadata:
        raise Skip(
            f"the registry already has processed_by {metadata['processed_by']!r}"
        )
    try:
        headers = abfzip.directory_to_headers(src)
        software = abfzip.headers_to_software(headers)
        if software is not None:
            if metadata.get("software", software) != software:
                raise ValueError(
                    f"the registry's software is {metadata['software']!r}, the "
                    f"files' is {software!r}"
                )
            metadata["software"] = software
        metadata["processed_by"] = PROCESSED_BY
        catalog = abfzip.files_to_catalog(src, headers)
        abfzip.build(src, abfzip.metadata_document(metadata, catalog), out)
    except ValueError as err:
        out.unlink(missing_ok=True)
        raise Skip(str(err)) from err


def check_zip(path: Path, record: dict) -> dict:
    """Checks the zip at path against the registry record of the directory it
    replaces (see the module's docstring), and returns the metadata it adds to
    the registry's. Raises Skip if it doesn't check."""
    try:
        metadata = readers.read_abfzip_metadata(path)
    except ValueError as err:
        raise Skip(f"the zip fails the reader: {err}") from err
    expected = registry_metadata(record)
    added = {k: v for k, v in metadata.items() if k not in expected}
    if {k: metadata.get(k) for k in expected} != expected or set(added) - set(ADDED):
        raise Skip(f"the zip's metadata is {metadata}, not the registry's {expected}")
    hasher = util.DirectoryHasher()
    with zipfile.ZipFile(path) as zf:
        catalog = json.loads(zf.read(abfzip.METADATA_NAME))["catalog"]
        members = sorted(n for n in zf.namelist() if n != abfzip.METADATA_NAME)
        if sorted(r["path"] for r in catalog) != members:
            raise Skip("the zip's catalog doesn't have a record for every member")
        for r in catalog:
            if "sha1" not in r:
                raise Skip(f"{r['path']} has no sha1 in the catalog")
            with zf.open(r["path"]) as fp:
                if sha1_of(fp) != r["sha1"]:
                    raise Skip(f"{r['path']} in the zip doesn't match its sha1")
            hasher.add(r["path"], r["sha1"])
    if hasher.hexdigest() != record["sha1"]:
        raise Skip(
            "the files' hashes don't give the directory's registered hash (the "
            "directory has changed)"
        )
    return added


def migrate(cfg, reg: Registry, record: dict, *, dry_run: bool) -> dict:
    """Migrates one resource. Returns its log entry."""
    id = record["name"]
    src = archive.resource_path(cfg, id)
    dst = archive.new_resource_path(cfg, id, f"{id}.zip")
    entry = {"id": id, "old_sha1": record["sha1"]}

    if record["dtype"] == DTYPE_TO:
        # migrated before; the directory may still be there
        with open(dst, "rb") as fp:
            if sha1_of(fp) != record["sha1"]:
                raise RuntimeError(f"{id}: {dst} doesn't match the registry's hash")
        if not src.is_dir():
            return {**entry, "outcome": "already migrated"}
        if dry_run:
            return {**entry, "outcome": "would remove the directory"}
        archive.remove(src)
        return {**entry, "outcome": "finished: directory removed"}
    if record["dtype"] != DTYPE_FROM:
        raise Skip(f"its dtype is {record['dtype']}")
    if not src.is_dir():
        raise Skip(f"no directory at {src}")

    if dst.exists():
        # stored by an interrupted run, before the registry was updated
        added = check_zip(dst, record)
        with open(dst, "rb") as fp:
            entry["new_sha1"] = sha1_of(fp)
        if dry_run:
            return {**entry, "outcome": "would finish: zip already stored"}
        entry["outcome"] = "finished: zip already stored"
    else:
        staged = archive.staging_dir(cfg) / f"{id}.zip"
        staged.unlink(missing_ok=True)
        build_zip(src, record, staged)
        try:
            added = check_zip(staged, record)
        except Skip:
            staged.unlink()
            raise
        with open(staged, "rb") as fp:
            entry["new_sha1"] = sha1_of(fp)
        if dry_run:
            staged.unlink()
            return {**entry, "outcome": "would migrate"}
        os.replace(staged, dst)
        archive.permission_fixer(cfg)(dst)
        entry["outcome"] = "migrated"

    updated = reg.migrate(id, entry["new_sha1"], added)
    if (
        updated.get("sha1") != entry["new_sha1"]
        or updated.get("dtype") != DTYPE_TO
        or any(updated["metadata"].get(k) != v for k, v in added.items())
    ):
        raise RuntimeError(f"{id}: the registry's record after the update is {updated}")
    archive.remove(src)
    return entry


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("-r", dest="registry_url", default=registry.default_registry())
    p.add_argument(
        "-a", dest="auth", help="username:password for the registry (default: .netrc)"
    )
    p.add_argument(
        "--dry-run", action="store_true", help="build and check, change nothing"
    )
    p.add_argument("--log", help="where to append the log (required unless --dry-run)")
    p.add_argument("--limit", type=int, help="migrate at most this many resources")
    p.add_argument("archive", type=Path, help="the archive's root directory")
    p.add_argument("ids", nargs="*", help="migrate only these resources")
    args = p.parse_args(argv)
    if args.registry_url is None:
        p.error("supply a registry url with -r or NBANK_REGISTRY")
    if args.log is None and not args.dry_run:
        p.error("--log is required")
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    try:
        cfg = archive.get_config(args.archive)
    except OSError as err:
        p.error(f"{args.archive}: not an archive ({err})")
    if not cfg["policy"]["keep_extensions"]:
        p.error(f"{args.archive} doesn't keep extensions, so <id>.zip can't be stored")
    if not archive.staging_dir(cfg).is_dir():
        p.error(
            f"{args.archive} has no staging directory; run `nbank check archive --fix`"
        )
    auth = tuple(args.auth.split(":", 1)) if args.auth else None
    reg = Registry(args.registry_url, auth)
    name = reg.archive_name(cfg["path"])

    records = [
        r
        for dtype in (DTYPE_FROM, DTYPE_TO)
        for r in reg.resources(dtype)
        if name in r["locations"]
        and (dtype == DTYPE_FROM or archive.resource_path(cfg, r["name"]).is_dir())
    ]
    if args.ids:
        wanted = set(args.ids)
        records = [r for r in records if r["name"] in wanted]
        if missing := wanted - {r["name"] for r in records}:
            p.error(
                f"not {DTYPE_FROM} resources in {name}: {', '.join(sorted(missing))}"
            )
    if args.limit is not None:
        records = records[: args.limit]
    log.info(
        "%s: %d resources to migrate%s",
        name,
        len(records),
        " (dry run)" if args.dry_run else "",
    )

    outcomes = {}
    logfile = open(args.log, "a") if args.log else None
    try:
        for i, listed in enumerate(records, 1):
            # the record as it is now, not as listed at the start
            id = listed["name"]
            try:
                record = reg.get(id)
                if record is None:
                    raise Skip("no longer in the registry")
                entry = migrate(cfg, reg, record, dry_run=args.dry_run)
            except Skip as err:
                entry = {"id": id, "outcome": "skipped", "error": str(err)}
                log.warning("%s: skipped: %s", id, err)
            except Exception as err:
                if logfile:
                    logfile.write(
                        json.dumps({"id": id, "outcome": "stopped", "error": repr(err)})
                        + "\n"
                    )
                log.error("%s: stopped: %s", id, err)
                raise
            outcomes[entry["outcome"]] = outcomes.get(entry["outcome"], 0) + 1
            if logfile:
                logfile.write(json.dumps(entry, sort_keys=True) + "\n")
                logfile.flush()
            if i % 50 == 0:
                log.info("  %d of %d", i, len(records))
    finally:
        if logfile:
            logfile.close()
        for outcome, n in sorted(outcomes.items()):
            log.info("  %s: %d", outcome, n)


if __name__ == "__main__":
    sys.exit(main())
