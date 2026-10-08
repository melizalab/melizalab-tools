# -*- mode: python -*-
"""Tests of regenerate-pprox (dlab.kilo_regenerate).

Most tests remove a unit's pprox from the output of group-kilo-spikes on the
E36 excerpt (see test_group_spikes_excerpt.py), regenerate it from the
waveform file, and compare it with the original.
"""

import json
import shutil
from pathlib import Path

import h5py
import pytest

from dlab import kilo, kilo_regenerate

DATA = Path(__file__).parent / "data"
EXAMPLES = Path(__file__).parent.parent / "examples"
UNIT = "E36_5_1_c52"
PROVENANCE = {"processed_by", "derived_from", "trials_from"}


@pytest.fixture(scope="module")
def excerpt_output(tmp_path_factory):
    from test_group_spikes_excerpt import run_excerpt

    return run_excerpt(tmp_path_factory.mktemp("regenerate"))


@pytest.fixture
def output(excerpt_output, tmp_path):
    """A copy of the excerpt output with UNIT's pprox moved to original/"""
    out = shutil.copytree(excerpt_output, tmp_path / "out")
    (tmp_path / "original").mkdir()
    shutil.move(out / f"{UNIT}.pprox", tmp_path / "original")
    return out


def original(output, unit=UNIT):
    return json.loads((output.parent / "original" / f"{unit}.pprox").read_text())


def regenerate(output, *extra, units=None, recording="E36_5_1"):
    """Runs the script; returns the new directory"""
    new = output.parent / "new"
    units = units or [str(output / f"{UNIT}_spikes.h5")]
    kilo_regenerate.script(
        ["--units", ",".join(units), "-o", str(new), *extra, recording]
    )
    return new


def load(new, unit=UNIT):
    return json.loads((new / f"{unit}.pprox").read_text())


def without(doc, keys=PROVENANCE):
    return {k: v for k, v in doc.items() if k not in keys}


def test_regenerated_from_sibling_is_identical(output):
    """With the trials of another unit from the same run, the regenerated pprox
    is identical to the original, apart from its provenance."""
    doc = load(regenerate(output))
    assert without(doc) == without(original(output))
    assert doc["processed_by"][0] == original(output)["processed_by"][0]
    assert doc["processed_by"][1].startswith("regenerate-pprox ")
    assert doc["derived_from"] == str(output / f"{UNIT}_spikes.h5")
    assert doc["trials_from"] in {"E36_5_1_c675", "E36_5_1_c676"}


def test_trials_option(output):
    """--trials names the pprox to take the trials from."""
    sibling = output / "E36_5_1_c676.pprox"
    doc = load(regenerate(output, "--trials", str(sibling)))
    assert doc["trials_from"] == "E36_5_1_c676"
    assert without(doc) == without(original(output))


def test_from_arf(output, monkeypatch):
    """--from-arf makes the trials with the current version, using the sync
    track and prepad recorded in the waveform file; for output from the current
    version the result is the same."""
    nb = json.loads((DATA / "E36_excerpt_neurobank.json").read_text())
    monkeypatch.setattr(
        kilo_regenerate.nbank, "describe", lambda url, name: nb["record"]
    )
    monkeypatch.setattr(
        kilo.StimulusFinder,
        "get_durations",
        lambda self, names: {name: nb["durations"][name] for name in names},
    )
    arf = shutil.copy(DATA / "E36_excerpt.arf", output.parent / "E36_5_1.arf")
    doc = load(
        regenerate(output, "-r", nb["registry"], "--from-arf", recording=str(arf))
    )
    assert without(doc) == without(original(output))
    assert doc["trials_from"].startswith("E36_5_1.arf (regenerate-pprox ")


def test_from_arf_needs_sync_track(output):
    """Without the sync track (in the options or the waveform file), --from-arf
    can't run."""
    path = output / f"{UNIT}_spikes.h5"
    with h5py.File(path, "r+") as fp:
        del fp.attrs["sync_track"]
    with pytest.raises(SystemExit) as exc:
        regenerate(output, "--from-arf")
    assert exc.value.code == 1


def test_siblings_from_registry(output, monkeypatch):
    """Units given as neurobank ids take their trials from the recording's
    other units in the registry; derived_from is the waveform file's URL."""
    files = {p.stem: p for p in output.iterdir()}

    def find_resource(name, registry_url, **kwargs):
        if name not in files:
            raise FileNotFoundError(name)
        return files[name]

    def search(registry_url, dtype, name):
        assert dtype == "spikes-pprox"
        return [{"name": stem} for stem in files if not stem.endswith("_spikes")]

    monkeypatch.setattr(kilo_regenerate.nbank, "find_resource", find_resource)
    monkeypatch.setattr(kilo_regenerate.nbank_core, "search", search)
    new = regenerate(output, "-r", "https://registry/", units=[f"{UNIT}_spikes"])
    doc = load(new)
    assert without(doc) == without(original(output))
    assert doc["derived_from"] == f"https://registry/resources/{UNIT}_spikes/"


def test_sibling_must_reproduce_its_own_events(output, caplog):
    """A sibling whose events don't match its own waveform file isn't used."""
    for name in ("E36_5_1_c675", "E36_5_1_c676"):
        path = output / f"{name}.pprox"
        doc = json.loads(path.read_text())
        doc["pprox"][0]["events"] = doc["pprox"][0]["events"][1:]
        path.write_text(json.dumps(doc))
    with pytest.raises(SystemExit) as exc:
        regenerate(output)
    assert exc.value.code == 1
    assert "don't reproduce its own events" in caplog.text
    assert not (output.parent / "new" / f"{UNIT}.pprox").exists()


def test_siblings_with_different_trials(output, caplog):
    """If the other units have different trials, only those processed by the
    waveform file's version are used; with none, the script asks for
    --trials."""
    path = output / "E36_5_1_c675.pprox"
    doc = json.loads(path.read_text())
    doc["pprox"][0]["offset"] += 1.0
    doc["processed_by"] = ["group-kilo-spikes 0.1"]
    path.write_text(json.dumps(doc))
    assert load(regenerate(output))["trials_from"] == "E36_5_1_c676"

    other = output / "E36_5_1_c676.pprox"
    doc = json.loads(other.read_text())
    doc["processed_by"] = ["group-kilo-spikes 0.2"]
    other.write_text(json.dumps(doc))
    shutil.rmtree(output.parent / "new")
    with pytest.raises(SystemExit):
        regenerate(output)
    assert "choose one with --trials" in caplog.text


def test_no_siblings(output, caplog):
    for path in output.glob("*.pprox"):
        path.unlink()
    with pytest.raises(SystemExit):
        regenerate(output)
    assert "use --trials or --from-arf" in caplog.text


def test_refuses_to_overwrite(output, caplog):
    new = regenerate(output)
    with pytest.raises(SystemExit):
        regenerate(output)
    assert f"{new / UNIT}.pprox exists" in caplog.text


def test_unit_of_another_recording(output, caplog):
    with pytest.raises(SystemExit):
        regenerate(output, recording="E36_5_2")
    assert "not a unit of E36_5_2" in caplog.text


@pytest.mark.slow
def test_p397_regenerated_exactly(tmp_path):
    """Three P397 units (2026.06.22; two have spikes exactly at trial
    boundaries) regenerated from their waveform files are identical to their
    deposited pprox files, apart from provenance."""
    ex = EXAMPLES / "P397_1_1" / "output"
    if not ex.exists():
        pytest.skip("examples/P397_1_1/output not present")
    out = shutil.copytree(ex, tmp_path / "out")
    (tmp_path / "original").mkdir()
    units = ["P397_1_1_c114", "P397_1_1_c357", "P397_1_1_c373"]
    for unit in units:
        shutil.move(out / f"{unit}.pprox", tmp_path / "original")
    new = regenerate(
        out,
        units=[str(out / f"{u}_spikes.h5") for u in units],
        recording="P397_1_1",
    )
    for unit in units:
        assert without(load(new, unit)) == without(original(out, unit))


def test_regenerated_passes_audit_and_schema(output):
    """A regenerated pprox, with its siblings, passes the audit, and conforms
    to the stimtrial schema."""
    from dlab import kilo_audit
    from dlab.pprox import validate

    new = regenerate(output)
    shutil.copy(new / f"{UNIT}.pprox", output)
    units = kilo_audit.load_units([str(output)], None)
    report = kilo_audit.audit_recording(
        DATA / "E36_excerpt.arf", units, recording="E36_5_1"
    )
    assert report["status"] == "ok"
    validate(load(new))
