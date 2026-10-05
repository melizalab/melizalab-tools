# -*- mode: python -*-
from dlab import util


def test_all_same():
    """Returns the common value, or None if the elements differ."""
    assert util.all_same([3, 3, 3]) == 3
    assert util.all_same([3, 4, 3]) is None


def test_all_same_empty():
    """An empty sequence returns None (it used to raise StopIteration)."""
    assert util.all_same([]) is None


def make_parser(**kwargs):
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("-k", action=util.ParseKeyVal, dest="meta", **kwargs)
    return p


def test_parse_key_val_literals():
    """Values are parsed as Python literals where possible, else kept as str."""
    args = make_parser().parse_args(["-k", "n=3", "-k", "s=abc", "-k", "l=[1, 2]"])
    assert args.meta == {"n": 3, "s": "abc", "l": [1, 2]}


def test_parse_key_val_without_arguments():
    """With no default and no -k, the attribute is None."""
    assert make_parser().parse_args([]).meta is None


def test_parse_key_val_badly_formed():
    """An argument without exactly one '=' raises ValueError (not an argparse
    usage error)."""
    import pytest

    for arg in ("noequals", "a=b=c"):
        with pytest.raises(ValueError, match="badly formed"):
            make_parser().parse_args(["-k", arg])


def test_parse_key_val_shares_mutable_default():
    """PINNED BUG (see TODO.md): with default=dict(), as the docstring suggests,
    the action updates the default in place, so values leak into later parses
    with the same parser."""
    p = make_parser(default=dict())
    p.parse_args(["-k", "a=1"])
    assert p.parse_args([]).meta == {"a": 1}, "PINNED: should be {}"


def test_json_serializable():
    """numpy scalars and arrays become Python numbers and lists; anything else
    is converted with str."""
    from pathlib import Path

    import numpy as np

    assert util.json_serializable(np.int64(3)) == 3
    assert type(util.json_serializable(np.float32(1.5))) is float
    assert util.json_serializable(np.arange(3)) == [0, 1, 2]
    assert util.json_serializable(Path("/a/b")) == "/a/b"


def test_setup_log_quiets_httpx():
    """httpx info messages are suppressed unless debugging."""
    import logging

    util.setup_log(False)
    assert logging.getLogger("httpx").level == logging.WARNING
    util.setup_log(True)
    assert logging.getLogger("httpx").level == logging.DEBUG
