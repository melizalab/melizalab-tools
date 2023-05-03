# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""Functions for interfacing with the neurobank repository """
import logging
from typing import Dict, Union
from urllib.parse import urlparse
import json

from anyio import Path, create_task_group
from httpx import AsyncClient, HTTPStatusError

from dlab import cache
from nbank import registry, util

logging.getLogger(__name__).addHandler(logging.NullHandler())
default_registry = registry.default_registry()


async def find_resource(
    session: AsyncClient,
    registry_url: str,
    resource_id: str,
    alt_base: Union[Path, str, None] = None,
    no_download: bool = False,
) -> Path:
    """Locate a neurobank resource

    This function will try to locate resources from the following locations:
    - a local directory (alt_base, if set),
    - a local neurobank archive (using alt_base, if provided),
    - a local cache
    - a remote HTTP archive (caching the file for local access later)

    It will raise a ValueError if the resource does not exist and a
    FileNotFoundError if the resource cannot be located.

    """
    if alt_base is not None:
        stem = Path(alt_base) / resource_id
        path = await resolve_local_path(stem)
        if path is not None:
            logging.debug("%s: found in alt_base", resource_id)
            return path
    url, params = registry.get_locations(registry_url, resource_id)
    response = await session.get(url, params=params)
    response.raise_for_status()
    locations = response.json()
    # search for local files
    for loc in locations:
        if loc["scheme"] not in registry._local_schemes:
            continue
        path = await resolve_local_path(Path(util.parse_location(loc, alt_base)))
        if path is not None:
            logging.debug("%s: found in local repository", resource_id)
            return path
    # search remote locations
    for loc in locations:
        if loc["scheme"] in registry._local_schemes:
            continue
        url = util.parse_location(loc)
        try:
            return await fetch_resource(session, url, resource_id, no_download)
        except FileNotFoundError:
            pass
    # all locations failed; raise an error
    raise FileNotFoundError(f"{resource_id}: unable to locate file")


async def resolve_local_path(stem: Path) -> Path:
    """Find a local resource based on a path or path stem.

    Because resource names often don't have extensions, a local filesystem
    search may be needed to resolve the full path.

    """
    if await stem.exists():
        return stem
    async for path in stem.parent.glob(f"{stem.name}.*"):
        return path


async def fetch_resource(
    session: AsyncClient,
    url: str,
    resource_id: str,  # this could be parsed out of the url
    no_download: bool = False,
) -> Path:
    """Fetch a downloadable resource from the registry.

    The file will be cached locally. Returns the path of the file.
    """
    target = await cache.locate(resource_id, urlparse(url).netloc)
    if await target.exists():
        logging.debug("%s: found in local cache", resource_id)
        return target
    elif no_download:
        raise FileNotFoundError(f"{url}: resource not found in local cache")
    logging.debug("%s: fetching from registry", resource_id)
    async with session.stream("GET", url) as response:
        if response.status_code == 404:
            raise FileNotFoundError(f"{url}: resource not found")
        elif response.status_code == 415:
            # should be possible to get the error message from registry?
            raise FileNotFoundError(f"{url}: this data type is not downloadable")
        async with await target.open("wb") as fp:
            async for data in response.aiter_bytes():
                await fp.write(data)
    return target


async def fetch_metadata(
    session: AsyncClient, registry_url: str, resource_id: str
) -> Dict:
    """Fetch metadata for a resource. Results will be cached locally."""
    target = await cache.locate(resource_id, urlparse(registry_url).netloc)
    if await target.exists():
        logging.debug("%s: metadata found in local cache", resource_id)
        return json.loads(await target.read_text())
    else:
        logging.debug("%s: fetching metadata from registry", resource_id)
        url, params = registry.get_resource(registry_url, resource_id)
        response = await session.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        await target.write_text(json.dumps(data))
        return data


def add_registry_argument(parser, dest="registry_url"):
    """Add a registry argument to an argparse parser"""
    parser.add_argument(
        "-r",
        dest=dest,
        help="URL of the registry service. "
        "Default is to use the environment variable '%s'" % registry._env_registry,
        default=default_registry,
    )


async def main(argv=None):
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
        await cache.clear(urlparse(args.registry_url).netloc)

    async def _find_and_show(session, id):
        try:
            path = await find_resource(
                session, args.registry_url, id, alt_base=args.base
            )
        except HTTPStatusError:
            logging.info("%s: no such resource", id)
        except FileNotFoundError:
            logging.info("%s: not found, unable to download", id)
        else:
            logging.info("%s: %s", id, path)

    headers = {"Accept": "application/json"}
    async with AsyncClient(
        timeout=None, headers=headers
    ) as session, create_task_group() as tg:
        for id in args.id:
            tg.start_soon(_find_and_show, session, id)


if __name__ == "__main__":
    from anyio import run

    run(main)
