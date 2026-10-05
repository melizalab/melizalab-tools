# -*- mode: python -*-
"""Utility functions for scripts and modules"""

import argparse
import logging
from functools import singledispatch

import numpy as np


def setup_log(debug=False):
    logging.basicConfig(
        format="%(message)s", level=logging.DEBUG if debug else logging.INFO
    )
    # suppress info messages from httpx
    logging.getLogger("httpx").setLevel(logging.DEBUG if debug else logging.WARNING)


class ParseKeyVal(argparse.Action):
    """argparse action for parsing -k key=value arguments

    Example: p.add_argument("-k", action=ParseKeyVal, default=dict(),
                            metavar="KEY=VALUE", dest="metadata")
    """

    def parse_value(self, value):
        import ast

        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value

    def __call__(self, parser, namespace, values, option_string=None):
        # copy, so a default dict is not modified
        kv = dict(getattr(namespace, self.dest) or {})
        if not values.count("=") == 1:
            raise argparse.ArgumentError(
                self, f"{values} argument badly formed; needs key=value"
            )
        key, val = values.split("=")
        kv[key] = self.parse_value(val)
        setattr(namespace, self.dest, kv)


@singledispatch
def json_serializable(val):
    """Serialize a value for the json module."""
    return str(val)


@json_serializable.register(np.generic)
def __js_numpy(val):
    """Used if *val* is an instance of a numpy scalar."""
    return val.item()


@json_serializable.register(np.ndarray)
def __js_numpy_arr(arr):
    """Used if *arr* is an instance of a numpy array."""
    return arr.tolist()


def all_same(seq):
    """If all elements of seq are equal, returns the value, otherwise None.

    An empty sequence returns None."""
    it = iter(seq)
    first = next(it, None)
    for e in it:
        if e != first:
            return None
    return first
