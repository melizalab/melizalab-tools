# -*- mode: python -*-
"""Tests for dlab.readers, the metadata readers for neurobank deposit."""

import json
from importlib.metadata import entry_points
from pathlib import Path

import h5py
import numpy as np
import pytest

from dlab import pprox, readers, spikes

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


@pytest.mark.parametrize(
    "dtype,reader",
    [
        ("spikes-pprox", readers.read_pprox_metadata),
        ("spikes-hdf5", readers.read_spikes_hdf5_metadata),
    ],
)
def test_entry_points(dtype, reader):
    """Each reader is registered under the name of its dtype."""
    (ep,) = entry_points(group="nbank.metadata_readers", name=dtype)
    assert ep.load() is reader


def test_registered():
    """Only the registered keys are kept, after renaming legacy keys."""
    assert readers.registered(
        {"bird": "a", "experiment": "b", "pen": 1, "recording": "r", "x": 2.0}
    ) == {"animal": "a", "recording": "r", "protocol": "b"}


@pytest.mark.parametrize(
    "path", sorted((DATA / "E36_excerpt_golden").glob("*.pprox")), ids=lambda p: p.name
)
def test_pprox_golden_outputs(path):
    """Files written by group-kilo-spikes 2026.10.07: the registered keys, with
    bird renamed to animal and processed_by as a string."""
    doc = json.loads(path.read_text())
    assert readers.read_pprox_metadata(path) == {
        "animal": doc["bird"],
        "experimenter": "bple",
        "recording": "https://gracula.psyc.virginia.edu/neurobank/resources/E36_5_1/",
        "protocol": "arc6opto1",
        "processed_by": "group-kilo-spikes 2026.10.07",
    }


def test_pprox_values(tmp_path):
    """processed_by is joined; null and non-finite values are left out, and so
    are keys that aren't registered."""
    path = tmp_path / "unit.pprox"
    path.write_text(
        '{"$schema": "https://meliza.org/spec:2/pprox.json#", "pprox": [],'
        ' "processed_by": ["group-kilo-spikes 2026.10.07", "regenerate-pprox 2026.10.17"],'
        ' "recording": "r", "protocol": null, "sorted_by": NaN, "pen": 1,'
        ' "kilosort_n_spikes": 10, "sort": {"dtype": "int16"}}'
    )
    assert readers.read_pprox_metadata(path) == {
        "recording": "r",
        "processed_by": "group-kilo-spikes 2026.10.07; regenerate-pprox 2026.10.17",
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
    path = write_json(tmp_path / "a.pprox", pprox.from_trials([], protocol="x"))
    before = path.read_bytes(), path.stat().st_mtime_ns
    readers.read_pprox_metadata(path)
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


def write_waveforms(path: Path, n_spikes: int = 3, **attributes) -> Path:
    waveforms = np.random.default_rng(0).normal(size=(n_spikes, 30))
    times = np.arange(n_spikes) * 1000
    spikes.save_waveforms(
        path, spikes.SpikeWaveforms(waveforms, times, 30000, 10), **attributes
    )
    return path


def test_attr_to_json():
    """numpy scalars become Python scalars and bytes are decoded; arrays,
    non-finite numbers and bytes that aren't UTF-8 become None."""
    values = [np.int64(3), np.float32(12.5), np.bool_(True), np.bytes_(b"ADC3")]
    converted = [readers._attr_to_json(v) for v in values]
    assert converted == [3, 12.5, True, "ADC3"]
    assert [type(v) for v in converted] == [int, float, bool, str]
    for value in (np.array([300.0, 6000.0]), np.nan, np.bytes_(b"\xff")):
        assert readers._attr_to_json(value) is None


def test_spikes_hdf5_values(tmp_path):
    """The registered root attributes, as JSON scalars, with bird renamed."""
    path = write_waveforms(
        tmp_path / "unit_spikes.h5",
        recording="https://example.org/resources/R1/",
        processed_by="group-kilo-spikes 2026.10.17",
        protocol=np.bytes_(b"chorus"),
        sorted_by=np.bytes_(b"\xff"),
        bird="7c7c975c-0340-42ea-935f-6e9f12d132a4",
        presenter="jpresent",
        kilosort_n_spikes=np.int64(3),
        sort_dtype="int16",
    )
    metadata = readers.read_spikes_hdf5_metadata(path)
    assert metadata == {
        "animal": "7c7c975c-0340-42ea-935f-6e9f12d132a4",
        "recording": "https://example.org/resources/R1/",
        "protocol": "chorus",
        "processed_by": "group-kilo-spikes 2026.10.17",
        "presenter": "jpresent",
    }
    json.dumps(metadata, allow_nan=False)


def test_spikes_hdf5_no_spikes(tmp_path):
    """A unit with no spikes has no mean waveform, and that's fine."""
    path = write_waveforms(tmp_path / "unit_spikes.h5", n_spikes=0, protocol="x")
    assert readers.read_spikes_hdf5_metadata(path) == {"protocol": "x"}


def test_spikes_hdf5_version_1(tmp_path):
    """Version 1 files have no mean waveform."""
    path = write_waveforms(tmp_path / "unit_spikes.h5", protocol="x")
    with h5py.File(path, "r+") as fp:
        fp.attrs["schema_ver"] = 1
        del fp["mean_waveform"], fp["sd_waveform"]
    assert readers.read_spikes_hdf5_metadata(path) == {"protocol": "x"}


def break_root_attr(fp, name, value):
    fp.attrs[name] = value


def delete(fp, name):
    del fp[name]


@pytest.mark.parametrize(
    "change,message",
    [
        (lambda fp: break_root_attr(fp, "schema", "other"), "not a waveform file"),
        (lambda fp: fp.attrs.__delitem__("schema"), "schema is None"),
        (lambda fp: break_root_attr(fp, "schema_ver", 3), "unknown version"),
        (lambda fp: delete(fp, "waveforms"), "no 2-D dataset 'waveforms'"),
        (lambda fp: delete(fp, "times"), "no 1-D dataset 'times'"),
        (lambda fp: fp["times"].attrs.__delitem__("sampling_rate"), "sampling_rate"),
        (lambda fp: fp["waveforms"].attrs.__delitem__("peak_index"), "peak_index"),
        (lambda fp: delete(fp, "mean_waveform"), "mean_waveform"),
    ],
)
def test_spikes_hdf5_invalid(tmp_path, change, message):
    path = write_waveforms(tmp_path / "unit_spikes.h5")
    with h5py.File(path, "r+") as fp:
        change(fp)
    with pytest.raises(ValueError, match=message):
        readers.read_spikes_hdf5_metadata(path)


def test_spikes_hdf5_count_mismatch(tmp_path):
    path = write_waveforms(tmp_path / "unit_spikes.h5")
    with h5py.File(path, "r+") as fp:
        del fp["times"]
        fp.create_dataset("times", data=np.arange(2)).attrs["sampling_rate"] = 30000
    with pytest.raises(ValueError, match="3 waveforms but 2 spike times"):
        readers.read_spikes_hdf5_metadata(path)


@pytest.mark.parametrize("contents", [b"", b"{}", b"\x89HDF\r\n\x1a\n"])
def test_spikes_hdf5_not_hdf5(tmp_path, contents):
    path = tmp_path / "unit_spikes.h5"
    path.write_bytes(contents)
    with pytest.raises(ValueError, match="not an HDF5 file"):
        readers.read_spikes_hdf5_metadata(path)


def test_spikes_hdf5_reader_is_read_only(tmp_path):
    path = write_waveforms(tmp_path / "unit_spikes.h5", protocol="x")
    before = path.read_bytes(), path.stat().st_mtime_ns
    readers.read_spikes_hdf5_metadata(path)
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before
