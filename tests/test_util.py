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


def test_parse_key_val_badly_formed(capsys):
    """An argument without exactly one '=' is an argparse usage error."""
    import pytest

    for arg in ("noequals", "a=b=c"):
        with pytest.raises(SystemExit):
            make_parser().parse_args(["-k", arg])
        assert "badly formed" in capsys.readouterr().err


def test_parse_key_val_does_not_modify_default():
    """With default=dict(), as the docstring suggests, values from one parse
    don't leak into the default (they used to)."""
    default = {"x": 0}
    p = make_parser(default=default)
    assert p.parse_args(["-k", "a=1"]).meta == {"x": 0, "a": 1}
    assert p.parse_args([]).meta == {"x": 0}
    assert default == {"x": 0}


def test_json_serializable():
    """numpy scalars and arrays become Python numbers and lists; anything else
    is converted with str."""
    from pathlib import Path

    import numpy as np

    assert util.json_serializable(np.int64(3)) == 3
    assert type(util.json_serializable(np.float32(1.5))) is float
    assert util.json_serializable(np.arange(3)) == [0, 1, 2]
    assert util.json_serializable(Path("/a/b")) == "/a/b"


def test_setup_log_quiets_http_libraries():
    """The HTTP libraries (httpx, httpcore) only log warnings, even with
    debug, unless debug_http is set."""
    import logging

    for debug in (False, True):
        util.setup_log(debug)
        for name in ("httpx", "httpcore"):
            assert logging.getLogger(name).level == logging.WARNING, (name, debug)
    util.setup_log(False, debug_http=True)
    for name in ("httpx", "httpcore"):
        assert logging.getLogger(name).level == logging.DEBUG


def test_add_log_arguments():
    import argparse

    p = argparse.ArgumentParser()
    util.add_log_arguments(p)
    args = p.parse_args(["--debug-http"])
    assert args.debug_http and not args.debug
    assert p.parse_args([]).debug is False
