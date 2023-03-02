# -*- coding: utf-8 -*-
# -*- mode: python -*-
""" Shared code for all scripts and modules """
__version__ = "2023.03.02"
APP_NAME = "dlab"
APP_AUTHOR = "melizalab"


def user_cache_dir():
    from appdirs import user_cache_dir

    return user_cache_dir(APP_NAME, APP_AUTHOR)
