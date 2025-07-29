# -*- mode: python -*-
"""Functions for interfacing with the neurobank repository"""

import concurrent.futures
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlparse

from httpx import Client, HTTPStatusError, NetRCAuth
from nbank import registry, util
from nbank.archive import resolve_extension
from nbank.core import deposit, describe, describe_many
from nbank.registry import log_error

from dlab import cache

log = logging.getLogger(__name__)
default_registry = registry.default_registry()
default_auth = NetRCAuth(None)

MaybeResourcePath = tuple[str, Path | FileNotFoundError]
MaybeResourceMetadata = tuple[str, dict | FileNotFoundError]


def find_resources(
    *resource_ids: str,
    registry_url: str | None = default_registry,
    alt_base: Path | str | None = None,
    no_download: bool = False,
) -> Iterator[MaybeResourcePath]:
    """Locate resources using neurobank or a local directory.

    This function will try to locate resources from the following locations:
    - a local directory (alt_base, if set),
    - a local neurobank archive (using registry_url and alt_base, if provided),
    - a local cache of previously fetched resources
    - a remote HTTP archive (using registry_url, caching the file for local access later)

    Yields results as they become available. The result for each requested id is
    a Path if the resource was successfully located, or exist, or a
    FileNotFoundError if the resource cannot be located.

    """
    to_locate = set(resource_ids)
    if alt_base is not None:
        path = Path(alt_base)
        for resource_id in resource_ids:
            stem = path / resource_id
            try:
                yield (resource_id, resolve_extension(stem))
                to_locate.remove(resource_id)
                log.debug("%s: found in alt_base", resource_id)
            except FileNotFoundError:
                pass
    if len(to_locate) == 0:
        return
    url, query = registry.get_locations_bulk(registry_url, to_locate)
    with Client() as client, concurrent.futures.ThreadPoolExecutor() as executor:
        response = util.query_registry_bulk(client, url, query)
        future_to_name = {
            executor.submit(
                fetch_resource,
                client,
                resource,
                alt_base=alt_base,
                no_download=no_download,
            ): resource["name"]
            for resource in response
        }
        for future in concurrent.futures.as_completed(future_to_name):
            resource_id = future_to_name[future]
            try:
                result = future.result()
            except FileNotFoundError as err:
                result = err
            to_locate.remove(resource_id)
            yield (resource_id, result)
    for resource_id in to_locate:
        yield (resource_id, FileNotFoundError(f"no such resource {resource_id}"))


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


def fetch_resource(
    client: Client,
    resource: dict,
    *,
    alt_base: Path | str | None = None,
    no_download: bool = False,
) -> Path:
    """Fetch a resource.

    Given a sequence of locations where a resource could be found, this function
    will try to locate the file, prioritizing local archives.

    """
    resource_id = resource["name"]
    filename = resource.get("filename", resource_id)
    for location in resource["locations"]:
        loc = util.parse_location(location, alt_base=alt_base, http_session=client)
        if hasattr(loc, "path"):
            log.debug("- found in local repository: %s", location)
            return loc.path
        elif hasattr(loc, "url"):
            target = cache.locate(
                filename, Path(urlparse(loc.url).netloc) / "resources"
            )
            if target.exists():
                log.debug("- found in local cache: %s", target)
                return target
            if no_download:
                log.debug("- found at %s but configured not to download", loc.url)
                continue
            log.debug("- fetching from %s", loc.url)
            try:
                return loc.fetch(target)
            except HTTPStatusError as err:
                log.warning(
                    "%s: failed to retrieve from %s (http status %d)",
                    resource_id,
                    loc.url,
                    err.response.status_code,
                )
    raise FileNotFoundError(
        "resource not found in local archive, cache, or downloadable remote"
    )


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
            yield (result["name"], result)
    except Exception:
        for name in to_locate:
            yield (
                name,
                FileNotFoundError("not found in local dir, unable to access registry"),
            )


def add_registry_argument(parser, dest="registry_url"):
    """Add a registry argument to an argparse parser"""
    parser.add_argument(
        "-r",
        "--registry",
        dest=dest,
        help="URL of the registry service. "
        f"Default is to use the environment variable '{registry._env_registry}'",
        default=default_registry,
    )


def main(argv=None):
    import argparse

    from dlab.util import setup_log

    p = argparse.ArgumentParser(description="locate neurobank resources ")
    p.add_argument("--debug", help="show verbose log messages", action="store_true")
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
    setup_log(args.debug)

    if args.clear_cache:
        cache.clear(urlparse(args.registry_url).netloc)

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
    "fetch_resource",
    "find_resource",
    "find_resources",
    "log_error",
]

if __name__ == "__main__":
    main()
