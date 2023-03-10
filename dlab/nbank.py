# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""Functions for interfacing with the neurobank repository """
import logging
from typing import Dict, Union

from anyio import Path
from httpx import AsyncClient, HTTPStatusError
from nbank import registry, util

from dlab.util import setup_log

log = logging.getLogger("dlab.nbank")
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
    url, params = registry.get_locations(registry_url, resource_id)
    if alt_base is not None:
        path = await resolve_local_path(alt_base)
        if path is not None:
            return path
    response = await session.get(url, params=params)
    response.raise_for_status()
    locations = response.json()
    # search for local files
    for loc in locations:
        if loc["scheme"] not in registry._local_schemes:
            continue
        path = await resolve_local_path(Path(util.parse_location(loc, alt_base)))
        if path is not None:
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
    from urllib.parse import urlparse
    from dlab.core import user_cache_dir

    parsed_url = urlparse(url)
    cache_dir = Path(user_cache_dir()) / parsed_url.netloc
    target = cache_dir / resource_id
    if await target.exists():
        return target
    elif no_download:
        raise FileNotFoundError(f"{url}: resource not found in local cache")
    await cache_dir.mkdir(parents=True, exist_ok=True)
    log.debug("- fetching %s from registry", resource_id)
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
    """Fetch metadata for a resource"""
    url, params = registry.get_resource(registry_url, resource_id)
    response = await session.get(url, params=params)
    response.raise_for_status()
    return response.json()


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
    from anyio import create_task_group
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
    p.add_argument("id", help="the identifier of the resource", nargs="+")
    args = p.parse_args(argv)
    setup_log(log, args.debug)

    # async with AsyncClient() as session:
    #     for id in args.id:
    #         md = await fetch_metadata(session, args.registry_url, id)
    #         print(md)

    async def _find_and_show(session, id):
        try:
            path = await find_resource(
                session, args.registry_url, id, alt_base=args.base
            )
        except HTTPStatusError:
            log.info("%s: no such resource", id)
        except FileNotFoundError:
            log.info("%s: not found, unable to download", id)
        else:
            log.info("%s: %s", id, path)

    headers = {"Accept": "application/json"}
    async with AsyncClient(headers=headers) as session, create_task_group() as tg:
        for id in args.id:
            tg.start_soon(_find_and_show, session, id)


if __name__ == "__main__":
    from anyio import run

    run(main)
