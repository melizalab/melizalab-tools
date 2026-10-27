# -*- mode: python -*-
"""Tests for dlab.abfio: each ABF file read in place from a zip matches neo's
reading of the file itself."""

import shutil
import zipfile
from pathlib import Path

import numpy as np
import pytest
from nbank.canonical_zip import zip_directory
from neo.io.axonio import AxonIO
from neo.rawio import axonrawio
from neo.rawio.axonrawio import AxonRawIO

from dlab import abfio

DATA = Path(__file__).parent / "data" / "abf"
NAMES = ["2014_09_11_0013.abf", "2017_06_07_0044.abf", "2025_05_07_0003.abf"]


@pytest.fixture(scope="module")
def archive(tmp_path_factory):
    """A canonical zip of the test files, in a subdirectory, with other members
    before and after them"""
    src = tmp_path_factory.mktemp("rec")
    (src / "cells").mkdir()
    for name in NAMES:
        shutil.copy(DATA / name, src / "cells" / name)
    (src / "notes.txt").write_text("cell 5\n")
    out = tmp_path_factory.mktemp("zip") / "rec.zip"
    return zip_directory(src, out, extra=[("metadata.json", b"{}\n")])


def test_abf_members(archive):
    assert abfio.abf_members(archive) == [f"cells/{name}" for name in NAMES]


def signals(reader):
    reader.parse_header()
    return [
        reader.get_analogsignal_chunk(block, seg, stream_index=0)
        for block in range(reader.block_count())
        for seg in range(reader.segment_count(block))
    ]


@pytest.mark.parametrize("name", NAMES)
def test_raw_signals_match(archive, name):
    expected = signals(AxonRawIO(filename=str(DATA / name)))
    found = signals(abfio.ZippedAxonRawIO(archive, f"cells/{name}"))
    assert len(found) == len(expected)
    for a, b in zip(found, expected, strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("name", NAMES)
def test_block_matches(archive, name):
    expected = AxonIO(str(DATA / name)).read_block()
    found = abfio.ZippedAxonIO(archive, f"cells/{name}").read_block()
    pairs = list(
        zip(
            (s for seg in found.segments for s in seg.analogsignals),
            (s for seg in expected.segments for s in seg.analogsignals),
            strict=True,
        )
    )
    assert pairs
    for a, b in pairs:
        np.testing.assert_array_equal(a.magnitude, b.magnitude)
        assert a.units == b.units
        assert a.t_start == b.t_start
        assert a.sampling_rate == b.sampling_rate
        assert a.name == b.name


@pytest.mark.parametrize("name", NAMES)
def test_protocol_matches(archive, name):
    expected = AxonIO(str(DATA / name)).read_protocol()
    found = abfio.ZippedAxonIO(archive, f"cells/{name}").read_protocol()
    assert len(found) == len(expected)
    for a, b in zip(found, expected, strict=True):
        for x, y in zip(a.analogsignals, b.analogsignals, strict=True):
            np.testing.assert_array_equal(x.magnitude, y.magnitude)


def test_neo_is_restored(archive):
    """The names replaced while a header is parsed are neo's again after, so
    plain files still read"""
    abfio.ZippedAxonRawIO(archive, f"cells/{NAMES[0]}").parse_header()
    assert axonrawio.np is np
    assert axonrawio.os.path.getsize is __import__("os").path.getsize
    AxonRawIO(filename=str(DATA / NAMES[0])).parse_header()


def test_missing_member(archive):
    with pytest.raises(KeyError):
        abfio.ZippedAxonIO(archive, "cells/none.abf")


def test_compressed_member(tmp_path):
    path = tmp_path / "rec.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(DATA / NAMES[0], NAMES[0])
    with pytest.raises(ValueError, match="compressed"):
        abfio.ZippedAxonIO(path, NAMES[0])
