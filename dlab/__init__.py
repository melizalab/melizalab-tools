# -*- mode: python -*-
try:
    from importlib.metadata import version

    __version__ = version("dlab")
except Exception:
    # If package is not installed (e.g. during development)
    __version__ = "unknown"
