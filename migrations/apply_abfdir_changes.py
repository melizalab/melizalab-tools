# -*- mode: python -*-
"""Applies the change file from normalize_abfdir.py to the registry.

Only records with nothing left for review are applied. Each record must still
be as the plan saw it (its metadata, experimenter and animal); one changed
since is skipped and reported, and one already changed as planned is skipped,
so an interrupted run can be repeated. After each update the record the
registry returns is checked against the plan, and the run stops at the first
error or mismatch.

Every update is logged as a JSON line: the id, the record before, the update
sent, and the record after. --dry-run checks the records and changes nothing.

Run with `uv run python migrations/apply_abfdir_changes.py`. Deleted once the
migration is done.
"""

import argparse
import json
import sys

import httpx
from nbank import core, registry
from normalize_abfdir import FIELDS


def state(record: dict) -> dict:
    """A registry record as the change file describes it"""
    return {
        "metadata": {k: v for k, v in record["metadata"].items() if k not in FIELDS},
        **{f: record[f] for f in FIELDS},
    }


def expected(before: dict, update: dict) -> dict:
    """The state of a record after update, with None deleting a key"""
    after = {"metadata": dict(before["metadata"]), **{f: before[f] for f in FIELDS}}
    for key, value in update.items():
        if key in FIELDS:
            after[key] = value
        elif value is None:
            after["metadata"].pop(key, None)
        else:
            after["metadata"][key] = value
    return after


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("-r", dest="registry_url", default=registry.default_registry())
    p.add_argument(
        "-a",
        dest="auth",
        help="username:password for the registry (default: .netrc)",
    )
    p.add_argument("--dry-run", action="store_true", help="check, but change nothing")
    p.add_argument("--log", help="where to append the log (required unless --dry-run)")
    p.add_argument("changes", help="the change file from normalize_abfdir.py")
    args = p.parse_args(argv)
    if args.registry_url is None:
        p.error("supply a registry url with -r or NBANK_REGISTRY")
    if args.log is None and not args.dry_run:
        p.error("--log is required")
    auth = tuple(args.auth.split(":", 1)) if args.auth else None

    with open(args.changes) as fp:
        plans = [json.loads(line) for line in fp]
    ready = [p for p in plans if not p["review"] and p["update"]]
    current = {
        r["name"]: state(r)
        for r in core.describe_many(args.registry_url, *(p["id"] for p in ready))
    }
    print(
        f"{len(plans)} records: {len(ready)} to update, "
        f"{sum(bool(p['review']) for p in plans)} still for review",
        file=sys.stderr,
    )

    todo, skipped = [], {"gone": [], "changed since the plan": [], "already done": []}
    for p in ready:
        now = current.get(p["id"])
        if now is None:
            skipped["gone"].append(p["id"])
        elif now == expected(p["before"], p["update"]):
            skipped["already done"].append(p["id"])
        elif now != p["before"]:
            skipped["changed since the plan"].append(p["id"])
        else:
            todo.append(p)
    for reason, ids in skipped.items():
        if ids:
            print(f"  skipped, {reason}: {len(ids)}", file=sys.stderr)
            for id in ids:
                print(f"    {id}", file=sys.stderr)
    if args.dry_run:
        print(f"  would update {len(todo)}", file=sys.stderr)
        return

    done = 0
    with open(args.log, "a") as log:
        for p in todo:
            try:
                (record,) = core.update(
                    args.registry_url, p["id"], auth=auth, metadata=p["update"]
                )
            except httpx.HTTPStatusError as err:
                sys.exit(
                    f"{p['id']}: {err.response.status_code} {err.response.text}; "
                    f"stopped after {done} updates"
                )
            if "error" in record:
                sys.exit(f"{p['id']}: {record['error']}; stopped after {done} updates")
            after = state(record)
            log.write(
                json.dumps(
                    {
                        "id": p["id"],
                        "before": p["before"],
                        "update": p["update"],
                        "after": after,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            log.flush()
            if after != expected(p["before"], p["update"]):
                sys.exit(
                    f"{p['id']}: the registry's record isn't what was planned (see the "
                    f"log); stopped after {done + 1} updates"
                )
            done += 1
    print(f"  updated {done}", file=sys.stderr)


if __name__ == "__main__":
    main()
