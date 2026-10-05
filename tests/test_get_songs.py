# -*- mode: python -*-
"""Tests for dlab.get_songs: extracting an interval from an ARF file, and a
smoke test of the get-songs script with neurobank lookups faked."""

import logging

import arf
import ewave
import numpy as np
import pytest

from dlab import get_songs, signal

RATE = 40000
DATASET = "entry_00004/pcm_000"


@pytest.fixture
def source(tmp_path):
    """An ARF file with 2 s of a 1 kHz tone (int16) at DATASET"""
    path = tmp_path / "O103_song.arf"
    t = np.arange(2 * RATE) / RATE
    data = (8000 * np.sin(2 * np.pi * 1000 * t)).astype("int16")
    with arf.open_file(path, "w") as fp:
        entry = arf.create_entry(fp, "entry_00004", 0)
        arf.create_dataset(entry, "pcm_000", data, sampling_rate=RATE)
    return path


def test_get_interval(source):
    """The interval in ms is converted to samples (truncating) at the dataset's
    sampling rate."""
    sig = get_songs.get_interval(source, DATASET, [90.0, 900.0])
    assert sig.sampling_rate == RATE
    assert sig.samples.size == int(900 * RATE / 1000) - int(90 * RATE / 1000)
    assert sig.samples.dtype == np.float32


def test_get_interval_scales_integer_samples(source):
    """int16 samples are scaled to +/-1 (by ewave.rescale), so levels logged by
    the script are in dBFS."""
    sig = get_songs.get_interval(source, DATASET, [0.0, 100.0])
    assert np.abs(sig.samples).max() == pytest.approx(8000 / 32768, abs=1e-4)
    assert signal.dBFS(sig.samples) == pytest.approx(
        20 * np.log10(8000 / 32768), abs=0.05
    ), "full-scale-relative level of the 8000-count tone"


def test_get_interval_leaves_float_samples(tmp_path):
    """Floating-point samples are assumed to be scaled already."""
    path = tmp_path / "float.arf"
    with arf.open_file(path, "w") as fp:
        entry = arf.create_entry(fp, "entry_00004", 0)
        arf.create_dataset(entry, "pcm_000", np.full(RATE, 0.25), sampling_rate=RATE)
    sig = get_songs.get_interval(path, DATASET, [0.0, 100.0])
    assert sig.samples.dtype == np.float32
    assert (sig.samples == 0.25).all()


@pytest.mark.parametrize(
    "interval_ms",
    [[1500.0, 3000.0], [-10.0, 100.0], [500.0, 500.0], [900.0, 100.0]],
    ids=["past-end", "negative-start", "empty", "reversed"],
)
def test_get_interval_outside_dataset_is_an_error(source, interval_ms):
    """An interval that is empty or not entirely inside the dataset (2 s here)
    raises ValueError instead of returning fewer samples."""
    with pytest.raises(ValueError, match="is outside"):
        get_songs.get_interval(source, DATASET, interval_ms)


@pytest.fixture
def songs_yaml(tmp_path, source):
    path = tmp_path / "songs.yml"
    path.write_text(
        f"- name: O103\n"
        f"  source: {source}\n"
        f"  dataset: {DATASET}\n"
        f"  interval_ms: [90.0, 900.0]\n"
        f"- name: gone\n"
        f"  source: {tmp_path / 'missing'}\n"
        f"  dataset: x\n"
        f"  interval_ms: [0, 10]\n"
    )
    return path


@pytest.fixture
def no_neurobank(monkeypatch):
    """Make every neurobank lookup fail, so the script uses local paths."""

    def not_found(*args, **kwargs):
        raise FileNotFoundError("not in neurobank")

    monkeypatch.setattr(get_songs.nbank, "find_resource", not_found)


def run(songs_yaml, *extra):
    get_songs.script(["-r", "http://registry.test/", *extra, str(songs_yaml)])


def test_script_writes_wav(tmp_path, monkeypatch, songs_yaml, no_neurobank):
    """Each song is written to <name>.wav in the current directory, resampled
    to --rate and rescaled to --dBFS."""
    monkeypatch.chdir(tmp_path)
    run(songs_yaml, "--rate", "44100", "--dBFS", "-25")
    with ewave.open(tmp_path / "O103.wav") as fp:
        assert fp.sampling_rate == 44100
        assert fp.dtype == np.dtype("int16")
        data = ewave.rescale(fp.read(), "d")
    assert data.size == pytest.approx(0.81 * 44100, abs=2), "810 ms at 44.1 kHz"
    assert signal.dBFS(data) == pytest.approx(-25, abs=0.05)


def test_script_interval_outside_dataset_is_an_error(
    tmp_path, monkeypatch, source, no_neurobank
):
    """A song whose interval runs past the end of its dataset stops the script."""
    path = tmp_path / "songs.yml"
    path.write_text(
        f"- name: O103\n  source: {source}\n  dataset: {DATASET}\n"
        f"  interval_ms: [1500.0, 2500.0]\n"
    )
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="is outside"):
        run(path)
    assert not (tmp_path / "O103.wav").exists()


def test_script_skips_missing_source(
    tmp_path, monkeypatch, songs_yaml, no_neurobank, caplog
):
    """A source found neither in neurobank nor locally is skipped with a warning."""
    monkeypatch.chdir(tmp_path)
    with caplog.at_level(logging.WARNING, logger="dlab"):
        run(songs_yaml)
    assert "unable to find resource, skipping" in caplog.text
    assert not (tmp_path / "gone.wav").exists()


def test_script_prefers_neurobank(tmp_path, monkeypatch, songs_yaml, source):
    """A source is looked up in neurobank (with the --registry URL) first."""
    calls = []

    def find(name, registry_url):
        calls.append((name, registry_url))
        if name.endswith("missing"):
            raise FileNotFoundError(name)
        return source

    monkeypatch.setattr(get_songs.nbank, "find_resource", find)
    monkeypatch.chdir(tmp_path)
    run(songs_yaml)
    assert calls[0] == (str(source), "http://registry.test/")
    assert (tmp_path / "O103.wav").exists()


def test_script_deposit(tmp_path, monkeypatch, songs_yaml, no_neurobank):
    """With --deposit, each output file is deposited with metadata describing
    how it was made."""
    deposits = []

    def deposit(archive, files, **kwargs):
        deposits.append((archive, files, kwargs))
        yield {"name": "id"}

    monkeypatch.setattr(get_songs.nbank, "deposit", deposit)
    monkeypatch.chdir(tmp_path)
    run(songs_yaml, "--deposit", str(tmp_path / "archive"), "--highpass", "500")
    ((archive, files, kwargs),) = deposits
    assert archive == tmp_path / "archive"
    assert [f.name for f in files] == ["O103.wav"]
    assert kwargs["dtype"] == "vocalization-wav"
    assert kwargs["source_dataset"] == DATASET
    assert kwargs["source_interval_ms"] == [90.0, 900.0]
    assert kwargs["highpass_cutoff"] == 500.0
    assert kwargs["dBFS"] == pytest.approx(-20, abs=0.05), "default --dBFS"


def http_error(status, **response):
    import httpx

    request = httpx.Request("POST", "http://registry.test/")
    return httpx.HTTPStatusError(
        "error",
        request=request,
        response=httpx.Response(status, request=request, **response),
    )


def test_script_deposit_permission_error_is_logged(
    tmp_path, monkeypatch, songs_yaml, no_neurobank, caplog
):
    """A 403 from the registry is logged with its message, and the script
    carries on."""

    def deposit(*args, **kwargs):
        raise http_error(403, json={"detail": "not allowed"})

    monkeypatch.setattr(get_songs.nbank, "deposit", deposit)
    monkeypatch.chdir(tmp_path)
    with caplog.at_level(logging.WARNING):
        run(songs_yaml, "--deposit", str(tmp_path / "archive"))
    assert "unable to deposit" in caplog.text
    assert "not allowed" in caplog.text


def test_script_deposit_server_error_aborts(
    tmp_path, monkeypatch, songs_yaml, no_neurobank
):
    """NB: other HTTP errors (e.g. 500) are re-raised by nbank's log_error, so
    the script stops."""
    import httpx

    def deposit(*args, **kwargs):
        raise http_error(500)

    monkeypatch.setattr(get_songs.nbank, "deposit", deposit)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(httpx.HTTPStatusError):
        run(songs_yaml, "--deposit", str(tmp_path / "archive"))
