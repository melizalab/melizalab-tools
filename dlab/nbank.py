# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""Functions for interfacing with the neurobank repository """
import concurrent.futures
import logging
from typing import Tuple, Dict, Union, Iterator, Sequence
from urllib.parse import urlparse
from pathlib import Path

from httpx import Client
from nbank import registry, util
from nbank.archive import resolve_extension
from nbank.core import describe, describe_many, search

from dlab import cache

logging.getLogger(__name__).addHandler(logging.NullHandler())
default_registry = registry.default_registry()


MaybeResourcePath = Tuple[str, Union[Path, FileNotFoundError]]


def find_resources(
    *resource_ids: str,
    registry_url: str = default_registry,
    alt_base: Union[Path, str, None] = None,
    no_download: bool = False,
) -> Iterator[MaybeResourcePath]:
    """Locate resources using neurobank.

    This function will try to locate resources from the following locations:
    - a local directory (alt_base, if set),
    - a local neurobank archive (using alt_base, if provided),
    - a local cache
    - a remote HTTP archive (caching the file for local access later)

    Yields results as they become available. The result for each requested id is
    a Path if the resource was successfully located, or exist, or a
    FileNotFoundError if the resource cannot be located.

    """
    to_locate = set(resource_ids)
    if alt_base is not None:
        for resource_id in resource_ids:
            stem = Path(alt_base) / resource_id
            try:
                yield (resource_id, resolve_extension(stem))
                to_locate.remove(resource_id)
                logging.debug("%s: found in alt_base", resource_id)
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
                resource["locations"],
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
        yield (resource_id, FileNotFoundError("no such resource"))


def find_resource(
    resource_id: str,
    *,
    registry_url: str = default_registry,
    alt_base: Union[Path, str, None] = None,
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
    locations: Sequence[Dict],
    *,
    alt_base: Union[Path, str, None] = None,
    no_download: bool = False,
) -> Path:
    """Fetch a resource.

    Given a sequence of locations where a resource could be found, this function
    will try to locate the file, prioritizing local archives.

    """
    # search for local files
    for loc in locations:
        if loc["scheme"] not in registry._local_schemes:
            continue
        stem = util.parse_location(loc, alt_base)
        try:
            target = resolve_extension(stem)
            logging.debug("%s: found in local repository", loc["resource_name"])
            return target
        except FileNotFoundError:
            pass
    for loc in locations:
        if loc["scheme"] in registry._local_schemes:
            continue
        resource_id = loc["resource_name"]
        url = util.parse_location(loc)
        target = cache.locate(resource_id, Path(urlparse(url).netloc) / "resources")
        if target.exists():
            logging.debug("%s: found in local cache", resource_id)
            return target
        if no_download:
            continue
        logging.debug("%s: fetching from registry", resource_id)
        with client.stream("GET", url) as response:
            if response.status_code != 200:
                logging.warn(
                    "%s: not available at %s (http status %d)",
                    resource_id,
                    url,
                    response.status_code,
                )
                continue
            with target.open("wb") as fp:
                for data in response.iter_bytes():
                    fp.write(data)
            return target
    raise FileNotFoundError(
        "resource not found in local archive, cache, or downloadable remote"
    )


def add_registry_argument(parser, dest="registry_url"):
    """Add a registry argument to an argparse parser"""
    parser.add_argument(
        "-r",
        dest=dest,
        help="URL of the registry service. "
        "Default is to use the environment variable '%s'" % registry._env_registry,
        default=default_registry,
    )


def main(argv=None):
    import argparse

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
    logging.basicConfig(
        format="%(message)s", level=logging.DEBUG if args.debug else logging.INFO
    )

    if args.clear_cache:
        cache.clear(urlparse(args.registry_url).netloc)

    if len(args.id) > 0:
        for name, location in find_resources(
            *args.id, registry_url=args.registry_url, alt_base=args.base
        ):
            logging.info("%s: %s", name, location)


if __name__ == "__main__":
    main()
