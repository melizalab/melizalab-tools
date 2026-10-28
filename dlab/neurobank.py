# -*- mode: python -*-
"""Functions for interfacing with the neurobank repository"""

import json
import logging
import shutil
from collections.abc import Iterator
from pathlib import Path

from httpx import HTTPStatusError, NetRCAuth
from nbank import cache, core, registry
from nbank.core import deposit, describe, describe_many
from nbank.registry import log_error

log = logging.getLogger(__name__)
default_registry = registry.default_registry()
try:
    default_auth = NetRCAuth(None)
except FileNotFoundError:
    default_auth = None

MaybeResourcePath = tuple[str, Path | FileNotFoundError]
MaybeResourceMetadata = tuple[str, dict | FileNotFoundError]


def find_resources(
    *resource_ids: str,
    registry_url: str | None = default_registry,
    alt_base: Path | str | None = None,
    no_download: bool = False,
) -> Iterator[MaybeResourcePath]:
    """Locate resources using neurobank or a local directory.

    Looks for each resource in alt_base (a directory of files named by id), and
    then where the registry says it is: a neurobank archive on this host (with
    alt_base, if set, replacing the directory that contains the archive), the
    cache, or a download into the cache unless no_download is set. Credentials
    are from default_auth (~/.netrc).

    Yields (id, result) as results become available, where result is a Path, or a
    FileNotFoundError if the resource can't be found. See
    nbank.core.find_resources.

    """
    base = None if alt_base is None else Path(alt_base)
    yield from core.find_resources(
        registry_url,
        *resource_ids,
        local_dir=base,
        alt_base=base,
        download=not no_download,
        auth=default_auth,
    )


def find_resource(
    resource_id: str,
    *,
    registry_url: str = default_registry,
    alt_base: Path | str | None = None,
    no_download: bool = False,
) -> Path:
    """Locate a resource using neurobank. This is a convenience wrapper for find_resources"""
    for _, result in find_resources(
        resource_id,
        registry_url=registry_url,
        alt_base=alt_base,
        no_download=no_download,
    ):
        if isinstance(result, Path):
            return result
        else:
            raise result


def describe_resources(
    *resource_ids: str,
    registry_url: str | None = default_registry,
    alt_base: Path | str | None = None,
) -> Iterator[MaybeResourceMetadata]:
    """Fetch resource metadata using neurobank or a local directory.

    This function will try to load resource metadta from the following locations:
    - a local directory (alt_base, if set),
    - a remote neurobank registry

    Yields results as they become available. The result for each requested id is
    a dict if the metadata was successfully loaded or FileNotFoundError if the resource cannot be located.

    """
    to_locate = set(resource_ids)
    if alt_base is not None:
        alt_base = Path(alt_base)
        for resource_id in resource_ids:
            path = (alt_base / resource_id).with_suffix(".json")
            if path.exists():
                yield (resource_id, json.loads(path.read_text()))
                to_locate.remove(resource_id)
                log.debug("%s: found in alt_base", resource_id)
    if len(to_locate) == 0:
        return
    try:
        for result in describe_many(registry_url, *to_locate):
            to_locate.discard(result["name"])
            yield (result["name"], result)
    except Exception:
        log.debug("unable to access registry", exc_info=True)
        for name in to_locate:
            yield (
                name,
                FileNotFoundError("not found in local dir, unable to access registry"),
            )
        return
    for name in to_locate:
        yield (name, FileNotFoundError(f"no such resource {name}"))


def add_registry_argument(parser, dest="registry_url"):
    """Add a registry argument to an argparse parser"""
    parser.add_argument(
        "-r",
        "--registry",
        dest=dest,
        help="URL of the registry service. "
        "Default is to use the environment variable 'NBANK_REGISTRY'",
        default=default_registry,
    )


def main(argv=None):
    """Locate neurobank resources from the command line (python -m dlab.neurobank)"""
    import argparse

    from dlab.util import add_log_arguments, setup_log

    p = argparse.ArgumentParser(description="locate neurobank resources ")
    add_log_arguments(p)
    add_registry_argument(p)
    p.add_argument(
        "-b",
        "--base",
        type=Path,
        help="set an alternative base directory to search for local resources",
    )
    p.add_argument(
        "--clear-cache",
        action="store_true",
        help="clear the contents of the local cache",
    )
    p.add_argument("id", help="identifier(s) of the resource(s) to locate", nargs="*")
    args = p.parse_args(argv)
    setup_log(args.debug, args.debug_http)

    if args.clear_cache:
        # the cache is organized by the host of each resource's archive, which
        # may differ from the registry host, so clear all of it
        log.info("clearing local cache dir %s", cache.cache_dir())
        shutil.rmtree(cache.cache_dir(), ignore_errors=True)

    if len(args.id) > 0:
        for name, location in find_resources(
            *args.id, registry_url=args.registry_url, alt_base=args.base
        ):
            logging.info("%s: %s", name, location)


__all__ = [
    "HTTPStatusError",
    "add_registry_argument",
    "default_auth",
    "default_registry",
    "deposit",
    "describe",
    "describe_resources",
    "find_resource",
    "find_resources",
    "log_error",
]

if __name__ == "__main__":
    main()
