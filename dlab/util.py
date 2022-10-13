# -*- coding: utf-8 -*-
# -*- mode: python -*-
""" Utility functions for scripts and modules """
import logging
import argparse
import numpy as np
from functools import singledispatch


def setup_log(log, debug=False):
    """Set up logging for a module.

    log: generate by calling e.g. `log = logging.getLogger("dlab.extracellular")`
    """
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(message)s")
    loglevel = logging.DEBUG if debug else logging.INFO
    log.setLevel(loglevel)
    ch.setLevel(loglevel)
    ch.setFormatter(formatter)
    log.addHandler(ch)


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

    def __call__(self, parser, namespace, arg, option_string=None):
        kv = getattr(namespace, self.dest)
        if kv is None:
            kv = dict()
        if not arg.count("=") == 1:
            raise ValueError("%s argument badly formed; needs key=value" % arg)
        else:
            key, val = arg.split("=")
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


def memodict(f):
    """Memoization decorator for a function taking a single argument"""

    class memodict(dict):
        def __missing__(self, key):
            ret = self[key] = f(key)
            return ret

    return memodict().__getitem__


def all_same(seq):
    """If all the elements of seq are the same, returns the value. If not, returns None"""
    it = iter(seq)
    first = next(it)
    for e in it:
        if e != first:
            return None
    return first
