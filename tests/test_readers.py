# -*- mode: python -*-
"""Tests for dlab.readers, the metadata readers for neurobank deposit."""

import json
from importlib.metadata import entry_points
from pathlib import Path

import pytest

from dlab import pprox, readers

DATA = Path(__file__).parent / "data"


def write_json(path: Path, obj) -> Path:
    path.write_text(json.dumps(obj))
    return path


def test_rename_legacy():
    """bird and experiment become animal and protocol; other keys are kept."""
    assert readers.rename_legacy({"bird": "a", "experiment": "b", "pen": 1}) == {
        "animal": "a",
        "protocol": "b",
        "pen": 1,
    }


def test_rename_legacy_agreeing():
    """A legacy key with the same value as its replacement is dropped."""
    assert readers.rename_legacy({"bird": "a", "animal": "a"}) == {"animal": "a"}


def test_rename_legacy_disagreeing():
    with pytest.raises(ValueError, match=r"'bird'.*'animal'.*disagree"):
        readers.rename_legacy({"bird": "a", "animal": "b"})


def test_pprox_entry_point():
    """The reader is registered under the name of its dtype."""
    (ep,) = entry_points(group="nbank.metadata_readers", name="spikes-pprox")
    assert ep.load() is readers.read_pprox_metadata


@pytest.mark.parametrize(
    "path", sorted((DATA / "E36_excerpt_golden").glob("*.pprox")), ids=lambda p: p.name
)
def test_pprox_golden_outputs(path):
    """Files written by group-kilo-spikes: the top-level scalars, with bird
    renamed to animal; lists and objects are left out."""
    doc = json.loads(path.read_text())
    metadata = readers.read_pprox_metadata(path)
    assert metadata["animal"] == doc["bird"]
    assert "bird" not in metadata
    for key in ("recording", "protocol", "experimenter", "kilosort_n_spikes"):
        assert metadata[key] == doc[key]
    for key in ("$schema", "pprox", "entry_metadata", "processed_by", "sort"):
        assert key not in metadata
    json.dumps(metadata, allow_nan=False)


def test_pprox_values(tmp_path):
    """Numbers stay numbers and booleans booleans; null and non-finite numbers
    are left out."""
    path = tmp_path / "unit.pprox"
    path.write_text(
        '{"$schema": "https://meliza.org/spec:2/pprox.json#", "pprox": [],'
        ' "pen": 1, "depth": 2.5, "good": true, "name": "x", "none": null,'
        ' "nan": NaN, "inf": Infinity, "obj": {"a": 1}, "list": [1]}'
    )
    assert readers.read_pprox_metadata(path) == {
        "pen": 1,
        "depth": 2.5,
        "good": True,
        "name": "x",
    }


def test_pprox_schema_without_hash(tmp_path):
    doc = pprox.from_trials([], schema="https://meliza.org/spec:2/stimtrial.json")
    assert readers.read_pprox_metadata(write_json(tmp_path / "a.pprox", doc)) == {}


def test_pprox_legacy_conflict(tmp_path):
    doc = pprox.from_trials([], bird="a", animal="b")
    with pytest.raises(ValueError, match="disagree"):
        readers.read_pprox_metadata(write_json(tmp_path / "a.pprox", doc))


@pytest.mark.parametrize(
    "contents,message",
    [
        (b"\x89HDF\r\n", "not a JSON file"),
        (b"{", "not a JSON file"),
        (b"[]", "top level"),
        (b'{"pprox": []}', r"\$schema is None"),
        (
            b'{"$schema": "https://meliza.org/spec:4/abfzip.json#", "pprox": []}',
            "abfzip",
        ),
        (b'{"$schema": "https://meliza.org/spec:2/pprox.json#"}', "no 'pprox'"),
        (
            b'{"$schema": "https://meliza.org/spec:2/pprox.json#", "pprox": {}}',
            "no 'pprox'",
        ),
        (
            b'{"$schema": "https://meliza.org/spec:2/pprox.json#", "pprox": [1]}',
            "trial",
        ),
    ],
)
def test_pprox_invalid(tmp_path, contents, message):
    path = tmp_path / "bad.pprox"
    path.write_bytes(contents)
    with pytest.raises(ValueError, match=message):
        readers.read_pprox_metadata(path)


def test_pprox_reader_is_read_only(tmp_path):
    path = write_json(tmp_path / "a.pprox", pprox.from_trials([], pen=1))
    before = path.read_bytes(), path.stat().st_mtime_ns
    readers.read_pprox_metadata(path)
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
