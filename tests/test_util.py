# -*- mode: python -*-
from dlab import util


def test_all_same():
    """Returns the common value, or None if the elements differ."""
    assert util.all_same([3, 3, 3]) == 3
    assert util.all_same([3, 4, 3]) is None


def test_all_same_empty():
    """An empty sequence returns None (it used to raise StopIteration)."""
    assert util.all_same([]) is None
