# -*- mode: python -*-
"""Local cache, primarily for storing neurobank resources"""

import logging
from pathlib import Path
from shutil import rmtree

import appdirs

APP_NAME = "dlab"
APP_AUTHOR = "melizalab"
user_dir = appdirs.user_cache_dir(APP_NAME, APP_AUTHOR)
log = logging.getLogger(__name__)


def locate(name: Path | str, subdir: Path | str) -> Path:
    """Return Path for a cached resource, creating subdir in the cache if needed"""
    cache_dir = Path(user_dir) / subdir
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / name


def clear(subdir: str) -> None:
    """Clear the contents of the cache"""
    cache_dir = Path(user_dir) / subdir
    log.info("clearing local cache dir %s", cache_dir)
    rmtree(cache_dir, ignore_errors=True)
