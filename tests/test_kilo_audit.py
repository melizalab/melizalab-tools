# -*- mode: python -*-
"""Tests of audit-kilo-spikes (dlab.kilo_audit).

Most tests audit the output of group-kilo-spikes on the E36 excerpt (see
test_group_spikes_excerpt.py), which should be clean, after corrupting a copy
of it in one specific way. The slow tests at the end audit the example
recordings, whose problems are known (see TODO.md).
"""

import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest

from dlab import kilo, kilo_audit

DATA = Path(__file__).parent / "data"
E36 = DATA / "E36_excerpt.arf"
EXAMPLES = Path(__file__).parent.parent / "examples"
UNIT = "E36_5_1_c52"


# --- shared spike assignment


def test_assign_spikes():
    """A spike exactly at a trial's start goes to the previous trial (as
    group-kilo-spikes has always done); spikes before the first trial get -1
    and spikes after the last trial's start go to the last trial."""
    starts = np.array([100, 200, 300])
    times = np.array([50, 100, 101, 200, 250, 900])
    assert kilo.assign_spikes(times, starts).tolist() == [-1, -1, 0, 0, 1, 2]


def test_waveforms_to_events():
    """Events are rebuilt per trial, relative to the stimulus onset (offset),
    and spikes before the first trial are counted, not assigned."""
    trials = [
        {"offset": 0.01, "recording": {"start": 100}},
        {"offset": 0.02, "recording": {"start": 400}},
    ]
    times = np.array([50, 450, 150, 600])  # unsorted, as a sanity check
    events, n_before = kilo.waveforms_to_events(times, trials, 10000.0)
    assert n_before == 1
    np.testing.assert_allclose(events[0], [(150 - 100) / 10000])
    np.testing.assert_allclose(events[1], [(450 - 200) / 10000, (600 - 200) / 10000])


# --- audits of the excerpt output


@pytest.fixture(scope="module")
def excerpt_output(tmp_path_factory):
    from test_group_spikes_excerpt import run_excerpt

    return run_excerpt(tmp_path_factory.mktemp("audit"))


@pytest.fixture
def output(excerpt_output, tmp_path):
    """A copy of the excerpt output that a test can modify"""
    return shutil.copytree(excerpt_output, tmp_path / "out")


def audit(output, *units, **kwargs):
    units = kilo_audit.load_units([str(u) for u in units] or [str(output)], None)
    return kilo_audit.audit_recording(E36, units, **kwargs)


def unit_report(report, name=UNIT):
    (unit,) = [u for u in report["units"] if u["name"] == name]
    return unit


def checks(unit):
    return [(f["check"], f["severity"], f.get("trials")) for f in unit["findings"]]


def edit_pprox(output, fn, name=UNIT):
    path = output / f"{name}.pprox"
    pprox = json.loads(path.read_text())
    fn(pprox)
    path.write_text(json.dumps(pprox))


def test_excerpt_output_is_clean(excerpt_output):
    """The current version's output has no findings at all."""
    report = audit(excerpt_output)
    assert report["status"] == "ok"
    assert len(report["units"]) == 3
    assert [checks(u) for u in report["units"]] == [[], [], []]
    assert report["findings"] == []


def test_missing_event(output):
    """An event missing from the pprox is caught by rebuilding the events from
    the waveform file."""

    def drop(pp):
        trial = pp["pprox"][2]
        trial["events"] = trial["events"][1:]

    edit_pprox(output, drop)
    assert checks(unit_report(audit(output))) == [("waveforms-events", "fail", [2])]


def test_mislabeled_trial(output):
    """A trial labeled with a stimulus other than the one in the message before
    its onset fails."""

    def swap(pp):
        pp["pprox"][1]["stimulus"]["name"] = pp["pprox"][0]["stimulus"]["name"]

    edit_pprox(output, swap)
    unit = unit_report(audit(output))
    assert unit["status"] == "fail"
    assert checks(unit) == [("stimulus-labels", "fail", [1])]


def shift_onset(trial, seconds):
    """Moves a trial's onset (and so its lag) without changing its spikes or
    sample range"""
    trial["offset"] += seconds
    trial["interval"] = [x - seconds for x in trial["interval"]]
    trial["events"] = [x - seconds for x in trial["events"]]


def test_late_onset(output):
    """A trial whose onset is out of line with its message is a warning,
    listing the trial (it can be excluded)."""
    edit_pprox(output, lambda pp: shift_onset(pp["pprox"][3], 0.5))
    report = audit(output)
    assert checks(unit_report(report)) == [("sync-lag", "warn", [3])]
    # the other units now have a different trial table
    assert [f["check"] for f in report["findings"]] == ["trial-tables"]


def test_late_onsets_in_old_version(output):
    """In output from a version before the sync fixes, the finding says the
    outliers are probably a known error."""

    def old(pp):
        shift_onset(pp["pprox"][3], 0.5)
        pp["processed_by"] = ["group-kilo-spikes 2026.07.15"]

    edit_pprox(output, old)
    (f,) = unit_report(audit(output))["findings"]
    assert "expected in versions before 2026.10.07" in f["message"]


def test_most_onsets_out_of_line(output):
    """Lag outliers in more than half of the trials fail the unit."""

    def drift(pp):
        for i, trial in enumerate(pp["pprox"]):
            shift_onset(trial, 0.3 * i)

    edit_pprox(output, drift)
    unit = unit_report(audit(output))
    assert ("sync-lag", "fail", [0, 1, 3, 4]) in checks(unit)


def test_events_outside_interval(output):
    def stray(pp):
        pp["pprox"][4]["events"].append(pp["pprox"][4]["interval"][1] + 1.0)

    edit_pprox(output, stray)
    assert ("events-in-interval", "warn", [4]) in checks(unit_report(audit(output)))


def test_missing_required_field(output):
    """A trial without a field stimtrial requires fails, and isn't checked
    further."""
    edit_pprox(output, lambda pp: pp["pprox"][0].pop("stimulus"))
    assert checks(unit_report(audit(output))) == [("pprox-fields", "fail", [0])]


def test_no_waveform_file(output):
    (output / f"{UNIT}_spikes.h5").unlink()
    assert checks(unit_report(audit(output))) == [("waveforms", "info", None)]


def rewrite_waveforms(path, **changes):
    """Rewrites a waveform file with changed times or attributes"""
    with h5py.File(path, "r") as fp:
        times = changes.pop("times", fp["times"][:])
        rate = fp["times"].attrs["sampling_rate"]
        attrs = {**fp.attrs, **changes}
    with h5py.File(path, "w") as fp:
        fp.create_dataset("times", data=times).attrs["sampling_rate"] = rate
        fp.attrs.update(attrs)


def test_waveform_file_from_another_recording(output):
    """A waveform file naming another recording fails, but its events are
    still compared (here they match: only the attribute is wrong)."""
    rewrite_waveforms(output / f"{UNIT}_spikes.h5", recording="elsewhere")
    assert checks(unit_report(audit(output))) == [("waveforms-recording", "fail", None)]


def test_spikes_before_first_trial(output):
    """Spikes before the first trial in the waveform file (kept by earlier
    versions) are only noted."""
    path = output / f"{UNIT}_spikes.h5"
    with h5py.File(path) as fp:
        times = fp["times"][:]
    rewrite_waveforms(path, times=np.r_[[10, 20], times])
    unit = unit_report(audit(output))
    assert unit["status"] == "info"
    assert checks(unit) == [("waveforms-before-first-trial", "info", None)]


def test_different_versions(output):
    edit_pprox(output, lambda pp: pp.update(processed_by=["group-kilo-spikes 0.1"]))
    assert [(f["check"], f["severity"]) for f in audit(output)["findings"]] == [
        ("versions", "info")
    ]


def test_units_from_registry(excerpt_output, monkeypatch):
    """A unit given as a neurobank id is located through the registry, along
    with its waveform file, the resource <id>_spikes; a unit whose waveform
    file isn't in the registry has none."""
    files = {
        UNIT: excerpt_output / f"{UNIT}.pprox",
        f"{UNIT}_spikes": excerpt_output / f"{UNIT}_spikes.h5",
        "E36_5_1_c675": excerpt_output / "E36_5_1_c675.pprox",
    }
    requested = []

    def find_resource(name, registry_url):
        requested.append((name, registry_url))
        if name not in files:
            raise FileNotFoundError(name)
        return files[name]

    monkeypatch.setattr(kilo_audit.nbank, "find_resource", find_resource)
    units = kilo_audit.load_units([UNIT, "E36_5_1_c675"], "https://registry/")
    assert [(u.name, u.waveforms_path) for u in units] == [
        (UNIT, files[f"{UNIT}_spikes"]),
        ("E36_5_1_c675", None),
    ]
    assert {url for _, url in requested} == {"https://registry/"}


# --- script


def test_script_writes_report(excerpt_output, tmp_path):
    """--units takes a directory or comma-separated files; the report is
    written to --output and the exit status is 0."""
    out = tmp_path / "report.json"
    units = ",".join(str(p) for p in sorted(excerpt_output.glob("*.pprox"))[:2])
    kilo_audit.script(["--units", units, "-o", str(out), str(E36)])
    report = json.loads(out.read_text())
    assert report["status"] == "ok" and len(report["units"]) == 2
    assert report["audited_by"].startswith("audit-kilo-spikes ")


def test_script_report_to_stdout(excerpt_output, capsys):
    kilo_audit.script(["--units", str(excerpt_output), str(E36)])
    assert json.loads(capsys.readouterr().out)["status"] == "ok"


def test_script_exits_nonzero_if_it_cannot_run(excerpt_output, tmp_path, monkeypatch):
    """Failing to run (here, a recording that can't be found) is the only
    non-zero exit."""

    def not_found(*args, **kwargs):
        raise FileNotFoundError("not in the registry")

    monkeypatch.setattr(kilo_audit.nbank, "find_resource", not_found)
    with pytest.raises(SystemExit) as exc:
        kilo_audit.script(["--units", str(excerpt_output), "no-such-recording"])
    assert exc.value.code == 1


# --- example recordings (slow)


def example_report(name, units, *extra):
    ex = EXAMPLES / name
    if not (ex / units).exists():
        pytest.skip(f"examples/{name}/{units} not present")
    kwargs = {"oeaudio_log": ex / extra[0]} if extra else {}
    loaded = kilo_audit.load_units([str(ex / units)], None)
    return kilo_audit.audit_recording(ex / f"{name}.arf", loaded, **kwargs)


@pytest.mark.slow
def test_p397_reference_is_clean():
    """P397's reference output (2026.06.22, clicks) only has spikes before the
    first trial in its waveform files."""
    report = example_report("P397_1_1", "output", "oeaudio_20260617-130556.log")
    assert report["status"] == "info"
    assert {f["check"] for u in report["units"] for f in u["findings"]} == {
        "waveforms-before-first-trial"
    }


@pytest.mark.slow
def test_e36_old_version_flags_end_of_pulse_trials():
    """The version before the sync fixes reported the pulses of trials 0, 3
    and 12 at their end; these are flagged in every unit, as warnings."""
    report = example_report("E36_5_1", "output-a20b62a")
    assert report["status"] == "warn"
    for unit in report["units"]:
        lag = [f for f in unit["findings"] if f["check"] == "sync-lag"]
        assert [(f["severity"], f["trials"]) for f in lag] == [("warn", [0, 3, 12])]


@pytest.mark.slow
def test_c401_fails():
    """C401's reference output (no sync line) fails: its onsets drift away from
    the messages, so most trials are mislabeled, and its waveform files name
    another recording."""
    report = example_report("C401_1_1b", "output")
    assert report["status"] == "fail"
    for unit in report["units"]:
        found = {f["check"] for f in unit["findings"] if f["severity"] == "fail"}
        assert {"stimulus-labels", "sync-lag", "waveforms-recording"} <= found
