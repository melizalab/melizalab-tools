# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""Functions for interfacing with the neurobank repository """
from appdirs import user_cache_dir
from aiohttp import ClientSession
import asyncio
from aiopath import AsyncPath
import logging
from nbank import registry, util
from typing import Optional
from dlab.util import setup_log

APP_NAME = "dlab"
APP_AUTHOR = "melizalab"
log = logging.getLogger("dlab.nbank")


async def find_resource(
    session: ClientSession,
    registry_url: str,
    resource_id: str,
    alt_base: Optional[str] = None,
) -> AsyncPaths:
    """Locate a neurobank resource

    This function will try to locate resources from the following locations: a
     local neurobank archive, a local cache, and finally a remote HTTP archive.
     It will raise a ValueError if the resource does not exist and a
     FileNotFoundError if the resource cannot be located.

    """
    url, params = registry.get_locations(registry_url, resource_id)
    async with session.get(url, params=params) as response:
        if response.status == 404:
            raise ValueError(f"{resource_id}: not a valid resource name")
        response.raise_for_status()
        locations = await response.json()
    # search for local files
    for loc in locations:
        if loc["scheme"] not in registry._local_schemes:
            continue
        path = AsyncPath(util.parse_location(loc, alt_base))
        if await path.exists():
            return path
        async for path in path.parent.glob(f"{path.name}.*"):
            return path
    # search remote locations
    for loc in locations:
        if loc["scheme"] in registry._local_schemes:
            continue
        url = util.parse_location(loc)
        try:
            return await fetch_resource(session, url, resource_id)
        except FileNotFoundError:
            pass
    # all locations failed; raise an error
    raise FileNotFoundError(f"{resource_id}: unable to locate file")


async def fetch_resource(
    session: ClientSession, url: str, resource_id: str, chunk_size: int = 8192
) -> AsyncPath:
    """Fetch a downloadable resource from the registry.

    The file will be cached locally. Returns the path of the file.
    """
    from urllib.parse import urlparse

    parsed_url = urlparse(url)
    cache_dir = AsyncPath(user_cache_dir(APP_NAME, APP_AUTHOR)) / parsed_url.netloc
    target = cache_dir / resource_id
    if await target.exists():
        return target
    await cache_dir.mkdir(parents=True, exist_ok=True)
    log.debug("- fetching %s from registry", resource_id)
    async with session.get(url) as response:
        if response.status == 404:
            raise FileNotFoundError(f"{url}: resource not found")
        elif response.status == 415:
            # should be possible to get the error message from registry?
            raise FileNotFoundError(f"{url}: this data type is not downloadable")
        async with target.open("wb") as fp:
            async for data in response.content.iter_chunked(chunk_size):
                await fp.write(data)
    return target


async def main(argv=None):
    import argparse

    p = argparse.ArgumentParser(description="locate neurobank resources ")
    p.add_argument("--debug", help="show verbose log messages", action="store_true")
    p.add_argument(
        "-r",
        dest="registry_url",
        help="URL of the registry service. "
        "Default is to use the environment variable '%s'" % registry._env_registry,
        default=registry.default_registry(),
    )
    p.add_argument("id", help="the identifier of the resource", nargs="+")
    args = p.parse_args(argv)
    setup_log(log, args.debug)

    async with ClientSession(headers={"Accept": "application/json"}) as session:
        tasks = [find_resource(session, args.registry_url, id) for id in args.id]
        for id, task in zip(args.id, asyncio.as_completed(tasks)):
            try:
                path = await task
            except ValueError:
                log.info("%s: no such resource", id)
            except FileNotFoundError:
                log.info("%s: not found, and unable to download")
            else:
                log.info("%s: %s", id, path)


if __name__ == "__main__":
    asyncio.run(main())