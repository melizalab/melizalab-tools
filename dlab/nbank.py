# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""Functions for interfacing with the neurobank repository """
import logging
import concurrent.futures
from typing import Tuple, Dict, Union
from urllib.parse import urlparse
from pathlib import Path

from httpx import Client
from nbank import registry, util
from nbank.core import search
from nbank.archive import resolve_extension

from dlab import cache

logging.getLogger(__name__).addHandler(logging.NullHandler())
default_registry = registry.default_registry()


def find_resources(
    *resource_ids: str,
    registry_url: str = default_registry,
    alt_base: Union[Path, str, None] = None,
    no_download: bool = False,
) -> Dict[str, Union[Path, ValueError, FileNotFoundError]]:
    """Locate resources using neurobank.

    This function will try to locate resources from the following locations:
    - a local directory (alt_base, if set),
    - a local neurobank archive (using alt_base, if provided),
    - a local cache
    - a remote HTTP archive (caching the file for local access later)

    Returns a dictionary of results. The result for each requested id is a Path
    if the resource was successfully located, a ValueError if the resource does
    not exist, or a FileNotFoundError if the resource cannot be located.

    """
    results = {}
    if alt_base is not None:
        for resource_id in resource_ids:
            stem = Path(alt_base) / resource_id
            try:
                results[resource_id] = resolve_extension(stem)
                logging.debug("%s: found in alt_base", resource_id)
            except FileNotFoundError:
                pass
    url, query = registry.get_locations_bulk(
        registry_url, [id for id in resource_ids if id not in results]
    )
    to_fetch = {}
    with Client() as client:
        response = util.query_registry_bulk(client, url, query)
        for resource in response:
            name = resource["name"]
            # search for local files
            for loc in resource["locations"]:
                if loc["scheme"] not in registry._local_schemes:
                    continue
                stem = util.parse_location(loc, alt_base)
                try:
                    results[name] = resolve_extension(stem)
                    logging.debug("%s: found in local repository", resource_id)
                except FileNotFoundError:
                    pass
            # search remote locations
            for loc in resource["locations"]:
                if loc["scheme"] in registry._local_schemes:
                    continue
                to_fetch[name] = util.parse_location(loc)
    with Client() as client, concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_name = {
            executor.submit(fetch_resource, client, name, url, no_download): name
            for name, url in to_fetch.items()
        }
        for future in concurrent.futures.as_completed(future_to_name):
            name = future_to_name[future]
            try:
                path = future.result()
            except FileNotFoundError as err:
                path = err
            results[name] = path
    return results


def fetch_resource(
    client: Client,
    resource_id: str,
    url: str,
    no_download: bool = False,
) -> Tuple[str, Union[Path, FileNotFoundError]]:
    """Fetch a downloadable resource from the registry."""
    target = cache.locate(resource_id, Path(urlparse(url).netloc) / "resources")
    if target.exists():
        logging.debug("%s: found in local cache", resource_id)
        return target
    elif no_download:
        raise FileNotFoundError(f"{url}: resource not found in local cache")
    logging.debug("%s: fetching from registry", resource_id)
    with client.stream("GET", url) as response:
        if response.status_code == 404:
            raise FileNotFoundError(f"{url}: resource not found")
        elif response.status_code == 415:
            raise FileNotFoundError(f"{url}: this data type is not downloadable")
        with target.open("wb") as fp:
            for data in response.iter_bytes():
                fp.write(data)
    return target


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
        find_resources(*args.id, registry_url=args.registry_url)


if __name__ == "__main__":
    main()
