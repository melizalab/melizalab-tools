# -*- coding: utf-8 -*-
# -*- mode: python -*-
""" Local cache, primarily for storing neurobank resources """
import logging

from anyio import Path, run_process
import appdirs

APP_NAME = "dlab"
APP_AUTHOR = "melizalab"
user_dir = appdirs.user_cache_dir(APP_NAME, APP_AUTHOR)
logging.getLogger(__name__).addHandler(logging.NullHandler())


async def locate(name: str, subdir: str) -> Path:
    """Return any anyio.Path for a cached resource, creating subdir in the cache if needed"""
    cache_dir = Path(user_dir) / subdir
    await cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / name


async def clear(subdir: str) -> None:
    """Clear the contents of the cache"""
    cache_dir = Path(user_dir) / subdir
    cmd = ["rm", "-rf", str(await cache_dir.resolve())]
    logging.debug("clearing local cache dir %s: '%s'", cache_dir, " ".join(cmd))
    _ = await run_process(cmd)
