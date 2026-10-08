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
REGISTRY = "https://registry/"
E36_RECORD = json.loads((DATA / "E36_excerpt_neurobank.json").read_text())["record"]


@pytest.fixture(autouse=True)
def fake_records(monkeypatch):
    """Registry records for describe_many: E36_5_1 as registered (it matches
    the excerpt's ARF attributes and pprox files). Tests can change or add
    records. Keeps the scripts off the network if NBANK_REGISTRY is set."""
    records = {"E36_5_1": json.loads(json.dumps(E36_RECORD))}

    def describe_many(registry_url, *ids):
        return [records[i] for i in ids if i in records]

    monkeypatch.setattr(kilo_audit.nbank_core, "describe_many", describe_many)
    return records


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
def e36_arf(tmp_path):
    """The excerpt under its recording's id, as the script expects"""
    return shutil.copy(E36, tmp_path / "E36_5_1.arf")


@pytest.fixture
def output(excerpt_output, tmp_path):
    """A copy of the excerpt output that a test can modify"""
    return shutil.copytree(excerpt_output, tmp_path / "out")


def audit(output, *units, **kwargs):
    units = kilo_audit.load_units([str(u) for u in units] or [str(output)], None)
    return kilo_audit.audit_recording(E36, units, recording="E36_5_1", **kwargs)


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
    assert checks(unit_report(audit(output))) == [
        ("schema", "warn", [0]),
        ("pprox-fields", "fail", [0]),
    ]


def test_schema_violation(output):
    """A pprox that doesn't conform to its schema is a warning, listing the
    trials and the first few problems."""

    def bad(pp):
        pp["pprox"][1]["events"][0] = "x"
        pp["pprox"][3]["stimulus"]["interval"] = "later"

    edit_pprox(output, bad)
    unit = unit_report(audit(output))
    schema = [f for f in unit["findings"] if f["check"] == "schema"]
    assert [(f["severity"], f["trials"]) for f in schema] == [("warn", [1, 3])]
    assert "pprox[1].events[0]: 'x' is not of type 'number'" in schema[0]["message"]
    # the unusable trials fail the unit, rather than crashing the audit
    assert ("pprox-fields", "fail", [1, 3]) in checks(unit)


def test_schema_unknown_or_missing(output):
    """A pprox with an unknown $schema, or none, is noted but not validated."""
    edit_pprox(output, lambda pp: pp.update({"$schema": "https://example.org/x.json"}))
    assert checks(unit_report(audit(output))) == [("schema", "info", None)]
    edit_pprox(output, lambda pp: pp.pop("$schema"))
    (f,) = unit_report(audit(output))["findings"]
    assert f["check"] == "schema" and "no $schema" in f["message"]


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


def test_pprox_names_another_recording(excerpt_output):
    """Units are grouped by name for the audit, so a pprox that names another
    recording is a warning."""
    units = kilo_audit.load_units([str(excerpt_output)], None)
    report = kilo_audit.audit_recording(E36, units, recording="E36_5_2")
    for unit in report["units"]:
        assert checks(unit) == [("recording-name", "warn", None)]


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

    def find_resource(name, registry_url, **kwargs):
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


def test_script_writes_report(excerpt_output, e36_arf, tmp_path):
    """--units takes a directory or comma-separated files; the report is
    written to --output and the exit status is 0."""
    out = tmp_path / "report.json"
    units = ",".join(str(p) for p in sorted(excerpt_output.glob("*.pprox"))[:2])
    kilo_audit.script(["--units", units, "-o", str(out), str(e36_arf)])
    report = json.loads(out.read_text())
    assert report["status"] == "ok" and len(report["units"]) == 2
    assert report["audited_by"].startswith("audit-kilo-spikes ")
    assert report["recording"] == "E36_5_1"


def test_script_report_to_stdout(excerpt_output, e36_arf, capsys):
    kilo_audit.script(["--units", str(excerpt_output), str(e36_arf)])
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


# --- selection


def test_group_units():
    """Units are grouped by recording; waveform files without a pprox are
    orphans, and names that don't fit <recording>_c<N> are listed."""
    groups, unmatched = kilo_audit.group_units(
        ["A_1_1_c3", "A_1_1_c12", "B_2_c1", "summary"],
        ["A_1_1_c3_spikes", "A_1_1_c12_spikes", "A_1_1_c40_spikes", "C_c2_spikes"],
    )
    assert groups == {
        "A_1_1": {
            "units": ["A_1_1_c12", "A_1_1_c3"],
            "orphans": ["A_1_1_c40_spikes"],
            "no_waveforms": [],
        },
        "B_2": {"units": ["B_2_c1"], "orphans": [], "no_waveforms": ["B_2_c1"]},
        "C": {"units": [], "orphans": ["C_c2_spikes"], "no_waveforms": []},
    }
    assert unmatched == ["summary"]


def test_already_audited(tmp_path):
    """A report covers a recording if it lists exactly the current units."""
    report = tmp_path / "A.json"
    report.write_text(json.dumps({"units": [{"name": "A_c1"}, {"name": "A_c2"}]}))
    assert kilo_audit.already_audited(report, ["A_c2", "A_c1"])
    assert not kilo_audit.already_audited(report, ["A_c1", "A_c2", "A_c3"])
    assert not kilo_audit.already_audited(tmp_path / "missing.json", ["A_c1"])
    (tmp_path / "bad.json").write_text("{")
    assert not kilo_audit.already_audited(tmp_path / "bad.json", ["A_c1"])


def local_archive(root: Path, *names: str) -> Path:
    """A neurobank archive on this host holding an (empty) ARF file for each
    name; returns the archive's path"""
    from nbank import archive

    archive.create(root, "https://registry/")
    for name in names:
        path = archive.resource_path(root, name).with_suffix(".arf")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return root


def archived(name: str, root: Path) -> dict:
    return {"scheme": "neurobank", "root": str(root), "resource_name": name}


@pytest.fixture
def fake_registry(monkeypatch, tmp_path):
    """Stands in for the registry searches; records the queries. The
    registered recordings' ARF files are in a neurobank archive on this host."""
    resources = {
        "spikes-pprox": ["A_1_c1", "A_1_c2", "A_10_c1", "B_1_c5", "Z_9_c1", "odd"],
        "spikes-hdf5": ["A_1_c1_spikes", "A_1_c2_spikes", "B_1_c7_spikes"],
    }
    root = local_archive(tmp_path / "archive", "A_1", "A_10", "B_1")
    recordings = {  # Z_9 isn't registered
        name: {"name": name, "locations": [archived(name, root)]}
        for name in ("A_1", "A_10", "B_1")
    }
    queries = []

    def search(registry_url, **params):
        queries.append(params)
        fragment = params.get("name", "")
        for name in resources[params["dtype"]]:
            if fragment in name:
                yield {"name": name, "dtype": params["dtype"]}

    def describe_many(registry_url, *ids):
        return [recordings[name] for name in ids if name in recordings]

    monkeypatch.setattr(kilo_audit.nbank_core, "search", search)
    monkeypatch.setattr(kilo_audit.nbank_core, "describe_many", describe_many)
    return queries


def test_find_units_script(fake_registry, tmp_path):
    """The control file has a line per registered recording with units; waveform
    files without a pprox go in the orphans file in the same format."""
    control, orphans = tmp_path / "audit.tsv", tmp_path / "orphans.tsv"
    kilo_audit.find_units_script(
        [
            "-r",
            "https://registry/",
            "--all",
            "-o",
            str(control),
            "--orphans",
            str(orphans),
        ]
    )
    assert control.read_text() == ("A_1\tA_1_c1,A_1_c2\nA_10\tA_10_c1\nB_1\tB_1_c5\n")
    assert orphans.read_text() == "B_1\tB_1_c7_spikes\n"
    assert [q["dtype"] for q in fake_registry] == ["spikes-pprox", "spikes-hdf5"]


def test_find_units_script_name_filter(fake_registry, capsys):
    """--name is passed to the registry searches; the control file goes to
    standard output by default."""
    kilo_audit.find_units_script(["-r", "https://registry/", "--name", "B_1"])
    assert capsys.readouterr().out == "B_1\tB_1_c5\n"
    assert all(q["name"] == "B_1" for q in fake_registry)


def test_find_units_script_skips_audited(fake_registry, tmp_path, capsys):
    """With --reports, recordings whose report covers the current units are
    skipped; a recording with a new unit is audited again."""
    reports = tmp_path / "reports"
    reports.mkdir()
    for rec, units in {"A_1": ["A_1_c1", "A_1_c2"], "B_1": ["B_1_c4"]}.items():
        (reports / f"{rec}.json").write_text(
            json.dumps({"units": [{"name": u} for u in units]})
        )
    kilo_audit.find_units_script(
        ["-r", "https://registry/", "--all", "--reports", str(reports)]
    )
    assert capsys.readouterr().out == "A_10\tA_10_c1\nB_1\tB_1_c5\n"


def test_read_recordings():
    """The first word of each line, as nbank search prints them, without
    blank lines, comments, or repeats."""
    lines = ["A_1\n", "\n", "# a comment\n", "B_1  extra\n", "A_1\n"]
    assert kilo_audit.read_recordings(lines) == ["A_1", "B_1"]


def test_find_units_for_listed_recordings(fake_registry, tmp_path, capsys, caplog):
    """With a file of recordings, only their units are found, though a name
    search for A_1 also returns A_10's; a recording without units is logged."""
    listed = tmp_path / "recordings.txt"
    listed.write_text("A_1\nC_3\n")
    with caplog.at_level("INFO", logger="dlab"):
        kilo_audit.find_units_script(["-r", "https://registry/", str(listed)])
    assert capsys.readouterr().out == "A_1\tA_1_c1,A_1_c2\n"
    assert {q["name"] for q in fake_registry} == {"A_1", "C_3"}
    assert "C_3: no units found" in caplog.text


def test_find_units_from_stdin(fake_registry, monkeypatch, capsys):
    """'-' reads the recordings from standard input, e.g. piped from nbank
    search."""
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO("B_1\n"))
    kilo_audit.find_units_script(["-r", "https://registry/", "-"])
    assert capsys.readouterr().out == "B_1\tB_1_c5\n"


def test_local_copy(tmp_path):
    """Only a copy in a neurobank archive on this host counts: not one in an
    archive that isn't here, a missing file, or a copy on tape or the web."""
    root = local_archive(tmp_path / "archive", "A_1")
    tape = {"scheme": "tape", "root": "T1:4", "resource_name": "A_1"}
    web = {"scheme": "https", "root": "https://x/", "resource_name": "A_1"}
    elsewhere = archived("A_1", tmp_path / "not_mounted")
    found = kilo_audit.local_copy({"locations": [tape, web, archived("A_1", root)]})
    assert found is not None and found.name == "A_1.arf" and found.exists()
    assert kilo_audit.local_copy({"locations": [tape, web, elsewhere]}) is None
    assert kilo_audit.local_copy({"locations": [archived("B_1", root)]}) is None
    assert kilo_audit.local_copy({"name": "A_1"}) is None


def test_find_units_skips_recordings_not_on_this_host(
    fake_registry, monkeypatch, tmp_path
):
    """Recordings whose ARF files aren't in a neurobank archive on this host
    (e.g. in cold storage) can't be audited here, so they go in the
    --unavailable file instead. Their orphans can still be regenerated (from
    other units' trials, without the ARF file)."""
    root = local_archive(tmp_path / "here", "A_1")
    records = {
        "A_1": {"name": "A_1", "locations": [archived("A_1", root)]},
        "A_10": {
            "name": "A_10",
            "locations": [{"scheme": "tape", "root": "T1:4", "resource_name": "A_10"}],
        },
        "B_1": {"name": "B_1", "locations": [archived("B_1", tmp_path / "elsewhere")]},
    }
    monkeypatch.setattr(
        kilo_audit.nbank_core,
        "describe_many",
        lambda url, *ids: [records[i] for i in ids if i in records],
    )
    control, unavailable, orphans = (tmp_path / n for n in ("a.tsv", "u.tsv", "o.tsv"))
    kilo_audit.find_units_script(
        [
            *("-r", "https://registry/", "--all", "-o", str(control)),
            *("--unavailable", str(unavailable), "--orphans", str(orphans)),
        ]
    )
    assert control.read_text() == "A_1\tA_1_c1,A_1_c2\n"
    assert unavailable.read_text() == "A_10\tA_10_c1\nB_1\tB_1_c5\n"
    assert orphans.read_text() == "B_1\tB_1_c7_spikes\n"


def test_find_units_warns_if_no_archive_here(fake_registry, monkeypatch, caplog):
    """If none of the recordings is on this host, the archive probably isn't
    mounted here, and the script says so."""
    monkeypatch.setattr(
        kilo_audit.nbank_core,
        "describe_many",
        lambda url, *ids: [{"name": i, "locations": []} for i in ids],
    )
    with caplog.at_level("WARNING", logger="dlab"):
        kilo_audit.find_units_script(["-r", "https://registry/", "--name", "B_1"])
    assert "run find-kilo-units (and the audits) on the archive host" in caplog.text


def test_audit_never_downloads(monkeypatch):
    """Resources are only looked up in archives on this host (and the cache):
    the audit scripts never download, and say where they looked."""
    calls = []

    def find_resource(name, registry_url, no_download=False):
        calls.append(no_download)
        raise FileNotFoundError("resource not found")

    monkeypatch.setattr(kilo_audit.nbank, "find_resource", find_resource)
    with pytest.raises(FileNotFoundError, match="not in a neurobank archive on this"):
        kilo_audit.locate("E36_5_1", "https://registry/")
    assert calls == [True]


def test_find_units_needs_a_selection(fake_registry, capsys):
    """Without recordings, --name or --all, the script refuses to run, rather
    than fetching every record in the registry."""
    with pytest.raises(SystemExit) as exc:
        kilo_audit.find_units_script(["-r", "https://registry/"])
    assert exc.value.code == 2
    assert "--all" in capsys.readouterr().err
    assert fake_registry == []  # no queries


def test_find_units_name_or_list(fake_registry):
    """--name and a list of recordings can't both be given."""
    with pytest.raises(SystemExit):
        kilo_audit.find_units_script(["--name", "A", "-"])


def test_httpx_messages_suppressed(fake_registry, capsys):
    """httpx's info messages (one per request) are not shown."""
    import logging

    kilo_audit.find_units_script(["-r", "https://registry/", "--name", "B_1"])
    assert logging.getLogger("httpx").getEffectiveLevel() == logging.WARNING


# --- collection


def make_report(recording, *units, findings=()):
    """A minimal report; units are (name, version, [(check, severity), ...])"""
    unit_reports = [
        {
            "name": name,
            "processed_by": [version],
            "status": kilo_audit.worst({"severity": sev} for _, sev in unit_findings),
            "findings": [
                kilo_audit.finding(check, sev, f"{check} message", [1, 2])
                for check, sev in unit_findings
            ],
        }
        for name, version, unit_findings in units
    ]
    rec_findings = [kilo_audit.finding(c, s, f"{c} message") for c, s in findings]
    return {
        "recording": recording,
        "status": kilo_audit.worst(
            [*rec_findings, *(f for u in unit_reports for f in u["findings"])]
        ),
        "findings": rec_findings,
        "units": unit_reports,
    }


OLD, NEW = "group-kilo-spikes 2026.06.22", "group-kilo-spikes 2026.10.07"
REPORTS = [
    make_report("A_1", ("A_1_c1", NEW, []), ("A_1_c2", NEW, [])),
    make_report(
        "B_1",
        ("B_1_c1", OLD, [("sync-lag", "warn")]),
        (
            "B_1_c2",
            OLD,
            [("sync-lag", "warn"), ("waveforms-before-first-trial", "info")],
        ),
    ),
    make_report(
        "C_1",
        ("C_1_c1", NEW, [("stimulus-labels", "fail")]),
        findings=[("trial-tables", "warn")],
    ),
]


def write_reports(directory, reports=REPORTS):
    directory.mkdir(exist_ok=True)
    for report in reports:
        (directory / f"{report['recording']}.json").write_text(json.dumps(report))
    return directory


def test_load_reports(tmp_path):
    """Reports are read from directories and files and sorted by recording;
    unreadable files and other JSON are listed, not fatal."""
    reports = write_reports(tmp_path / "reports", REPORTS[1:])
    (reports / "broken.json").write_text("{")
    (reports / "other.json").write_text('{"pprox": []}')
    single = tmp_path / "A_1.json"
    single.write_text(json.dumps(REPORTS[0]))
    loaded, errors = kilo_audit.load_reports([reports, single])
    assert [r["recording"] for r in loaded] == ["A_1", "B_1", "C_1"]
    assert [Path(e.split(":")[0]).name for e in errors] == ["broken.json", "other.json"]


def test_summarize():
    """Counts recordings and units by status, findings by check, and units by
    version, and lists the recordings at or above the level, worst first."""
    text = kilo_audit.summarize(REPORTS)
    lines = text.splitlines()
    assert lines[0] == "3 recordings, 5 units"
    status = {line.split()[0]: line.split()[1:] for line in lines[3:7]}
    assert status == {
        "ok": ["1", "2"],
        "info": ["0", "0"],
        "warn": ["1", "2"],
        "fail": ["1", "1"],
    }
    checks = [line.split() for line in lines if line.startswith(("sync-lag", "trial-"))]
    # the recording-level finding counts the recording, but no units
    assert checks == [
        ["sync-lag", "warn", "1", "2"],
        ["trial-tables", "warn", "1", "0"],
    ]
    versions = {
        line.rsplit(None, 4)[0]: line.split()[-4:]
        for line in lines
        if line.startswith("group-kilo-spikes")
    }
    assert versions == {OLD: ["0", "0", "2", "0"], NEW: ["2", "0", "0", "1"]}
    flagged = lines[lines.index("recordings with warn or fail:") + 1 :]
    assert [line.split()[:2] for line in flagged] == [["C_1", "fail"], ["B_1", "warn"]]
    assert flagged[0].endswith("stimulus-labels (fail) x1, trial-tables (warn)")
    assert flagged[1].endswith("sync-lag (warn) x2")


def test_summarize_level():
    """--level fail lists only failing recordings; info findings are never
    listed at the default level."""
    text = kilo_audit.summarize(REPORTS, level="fail")
    flagged = text.splitlines()[text.splitlines().index("recordings with fail:") + 1 :]
    assert [line.split()[0] for line in flagged] == ["C_1"]
    assert "waveforms-before-first-trial (info)" not in kilo_audit.summarize(REPORTS)


def test_write_findings(tmp_path):
    """One row per finding; recording-level findings have no unit or version."""
    out = tmp_path / "findings.tsv"
    kilo_audit.write_findings(out, REPORTS)
    rows = [line.split("\t") for line in out.read_text().splitlines()]
    assert rows[0] == [
        "recording",
        "unit",
        "version",
        "check",
        "severity",
        "trials",
        "message",
    ]
    assert len(rows) == 1 + 5
    assert rows[1] == [
        "B_1",
        "B_1_c1",
        OLD,
        "sync-lag",
        "warn",
        "1,2",
        "sync-lag message",
    ]
    assert ["C_1", "", "", "trial-tables", "warn", "", "trial-tables message"] in rows


def test_collect_script(tmp_path, capsys, caplog):
    """The summary goes to standard output; with --control, recordings without a
    report are logged."""
    reports = write_reports(tmp_path / "reports")
    control = tmp_path / "audit.tsv"
    control.write_text("A_1\tA_1_c1\nD_1\tD_1_c1\n")
    with caplog.at_level("WARNING", logger="dlab"):
        kilo_audit.collect_script(
            ["--control", str(control), "--tsv", str(tmp_path / "f.tsv"), str(reports)]
        )
    assert capsys.readouterr().out.startswith("3 recordings, 5 units\n")
    assert "1 of 2 recordings in" in caplog.text and "D_1" in caplog.text
    assert (tmp_path / "f.tsv").exists()


def test_every_check_is_documented():
    """docs/audit.md has a dictionary entry (and a table row) for each check
    the audit can report, and none for checks it doesn't."""
    import re

    source = Path(kilo_audit.__file__).read_text()
    checks = set(re.findall(r'finding\(\s*"([a-z-]+)"', source))
    doc = (Path(__file__).parent.parent / "docs" / "audit.md").read_text()
    assert set(re.findall(r"^#### `([a-z-]+)`", doc, re.M)) == checks
    assert set(re.findall(r"^\| \[`([a-z-]+)`\]", doc, re.M)) == checks


# --- metadata


def audit_with_registry(output, **kwargs):
    units = kilo_audit.load_units([str(output)], None)
    return kilo_audit.audit_recording(
        E36, units, recording="E36_5_1", registry_url=REGISTRY, **kwargs
    )


def test_metadata_agrees(excerpt_output):
    """The excerpt's ARF attributes, pprox files and registry record agree."""
    report = audit_with_registry(excerpt_output)
    assert report["status"] == "ok"
    assert report["registry"] == REGISTRY


def test_no_registry_no_metadata_checks(excerpt_output, fake_records):
    """Without a registry, only the ARF is checked; the report says so."""
    fake_records["E36_5_1"]["metadata"]["pen"] = 6
    report = audit(excerpt_output)
    assert report["status"] == "ok" and report["registry"] is None


def test_registry_disagrees(excerpt_output, fake_records):
    """A field that differs between the registry and the ARF (and the pprox
    files, which copied it from the registry) is reported for the recording
    and each unit."""
    fake_records["E36_5_1"]["metadata"]["pen"] = 6
    report = audit_with_registry(excerpt_output)
    assert [(f["check"], f["severity"]) for f in report["findings"]] == [
        ("metadata-arf", "warn")
    ]
    assert "pen: 6 (registry), 5 (ARF attributes)" in report["findings"][0]["message"]
    for unit in report["units"]:
        assert checks(unit) == [("metadata-registry", "warn", None)]
        assert "pen: 5 (pprox), 6 (registry)" in unit["findings"][0]["message"]


def test_field_only_in_registry(excerpt_output, fake_records):
    """A field added to the registry after the units were processed is noted."""
    fake_records["E36_5_1"]["metadata"]["hemisphere"] = "R"
    report = audit_with_registry(excerpt_output)
    for unit in report["units"]:
        (f,) = unit["findings"]
        assert (f["check"], f["severity"]) == ("metadata-registry", "info")
        assert "hemisphere (registry only)" in f["message"]


def test_recording_not_registered(excerpt_output, fake_records):
    del fake_records["E36_5_1"]
    report = audit_with_registry(excerpt_output)
    assert [(f["check"], f["severity"]) for f in report["findings"]] == [
        ("registry", "warn")
    ]


def test_unit_resource_disagrees(excerpt_output, fake_records):
    """The registry records of a unit's own resources (pprox or waveform file)
    are checked against the pprox, for the fields they share."""
    fake_records[f"{UNIT}_spikes"] = {
        "name": f"{UNIT}_spikes",
        "metadata": {"site": 2, "note": "anything"},
    }
    unit = unit_report(audit_with_registry(excerpt_output))
    assert checks(unit) == [("metadata-unit", "warn", None)]
    assert f"site: 1 (pprox), 2 ({UNIT}_spikes)" in unit["findings"][0]["message"]


def test_script_uses_registry(excerpt_output, e36_arf, fake_records, tmp_path):
    fake_records["E36_5_1"]["metadata"]["site"] = 3
    out = tmp_path / "report.json"
    kilo_audit.script(
        ["-r", REGISTRY, "--units", str(excerpt_output), "-o", str(out), str(e36_arf)]
    )
    report = json.loads(out.read_text())
    assert [f["check"] for f in report["findings"]] == ["metadata-arf"]


def metadata_arf(tmp_path, name, metadata=None, **attrs):
    """An ARF file with one entry, its attributes, and a metadata message"""
    import arf
    from conftest import add_entry, oeaudio_messages

    path = tmp_path / "meta.arf"
    with arf.open_file(path, "w") as fp:
        add_entry(
            fp,
            name,
            1000.0,
            nsamples=1000,
            messages=oeaudio_messages([], metadata=metadata),
            **attrs,
        )
    return path


def recording_checks(path, recording, record=None):
    with h5py.File(path, "r") as fp:
        entries = [e for _, e in kilo.iter_entries(fp)]
        return kilo_audit.check_recording_metadata(entries, recording, record)


def test_arf_disagrees_with_itself(tmp_path):
    """The ARF attributes and the metadata message are compared even without a
    registry; the message's 'experiment' is the protocol."""
    path = metadata_arf(
        tmp_path,
        "C1_2026-01-01_10-00-00_main",
        metadata={"animal": "C1", "experimenter": "someone", "experiment": "main"},
        experimenter="other",
        protocol="main",
        pen="1",
    )
    (f,) = recording_checks(path, "C1_1_1")
    assert f["check"] == "metadata-arf"
    assert f["message"] == (
        "the ARF file disagrees with itself: "
        "experimenter: other (ARF attributes), someone (ARF metadata message)"
    )


def test_bird_named_in_arf(tmp_path):
    """The bird in the entry name and the metadata message must match the
    recording's id."""
    path = metadata_arf(
        tmp_path, "E79_2026-06-23_12-33-22_chorus", metadata={"animal": "E79"}
    )
    assert recording_checks(path, "E79_1_1b") == []
    found = recording_checks(path, "C180_1_1")
    assert [f["message"] for f in found] == [
        "the ARF metadata message is for bird E79, but the recording is C180_1_1",
        "the ARF entry name is for bird E79, but the recording is C180_1_1",
    ]


@pytest.mark.slow
def test_c180_experimenter_disagrees():
    """C180's metadata message and entry attributes name different
    experimenters."""
    path = EXAMPLES / "C180_1_1.arf"
    if not path.exists():
        pytest.skip("examples/C180_1_1.arf not present")
    (f,) = recording_checks(path, "C180_1_1")
    assert (
        "experimenter: bple (ARF attributes), uac6qw (ARF metadata message)"
        in (f["message"])
    )


# --- aux pulses


@pytest.fixture(scope="module")
def aux_excerpt_output(tmp_path_factory):
    """The excerpt output with the LED channel as aux, checked against the
    condition messages: trial 4 has the one pulse"""
    from test_group_spikes_excerpt import run_excerpt

    return run_excerpt(
        tmp_path_factory.mktemp("audit_aux"), extra=("--aux", "led=ADC4:condition")
    )


@pytest.fixture
def aux_output(aux_excerpt_output, tmp_path):
    return shutil.copytree(aux_excerpt_output, tmp_path / "out")


def aux_checks(report, name=UNIT):
    return [
        (f["check"], f["severity"], f.get("trials"))
        for f in unit_report(report, name)["findings"]
        if f["check"].startswith("aux")
    ]


def modified_arf(tmp_path, fn):
    """A copy of the excerpt ARF with its LED channel (ADC4) changed by fn"""
    path = shutil.copy(E36, tmp_path / "E36_5_1.arf")
    with h5py.File(path, "r+") as fp:
        fn(fp["entry"]["ADC4"])
    return path


def audit_arf(output, arf):
    units = kilo_audit.load_units([str(output)], None)
    return kilo_audit.audit_recording(arf, units, recording="E36_5_1")


def test_aux_output_is_clean(aux_excerpt_output):
    """The LED pulse in trial 4 matches ADC4 and its condition message."""
    report = audit(aux_excerpt_output)
    assert report["status"] == "ok"
    assert unit_report(report)["findings"] == []


def test_aux_without_tracks(aux_output):
    edit_pprox(aux_output, lambda pp: pp.pop("aux_tracks"))
    assert aux_checks(audit(aux_output)) == [("aux-tracks", "warn", [4])]


def test_trial_without_aux_list(aux_output):
    edit_pprox(aux_output, lambda pp: pp["pprox"][1].pop("aux"))
    assert aux_checks(audit(aux_output)) == [("aux-fields", "warn", [1])]


def test_aux_pulse_with_unknown_name(aux_output):
    def rename(pp):
        pp["pprox"][4]["aux"][0]["name"] = "ttl"

    edit_pprox(aux_output, rename)
    found = unit_report(audit(aux_output))["findings"]
    (fields,) = [f for f in found if f["check"] == "aux-fields"]
    assert (
        fields["trials"] == [4] and "names not in aux_tracks: ttl" in fields["message"]
    )
    # and the led pulse on ADC4 is now missing from the pprox
    assert ("aux-pulses", "warn", [4]) in aux_checks(audit(aux_output))


def test_aux_pulse_outside_its_trial(aux_output):
    """A pulse that doesn't start in its trial is reported (it also no longer
    matches the channel)."""

    def move(pp):
        trial = pp["pprox"][3]
        start = trial["interval"][1] + 0.5
        trial["aux"].append({"name": "led", "interval": [start, start + 1.0]})

    edit_pprox(aux_output, move)
    assert ("aux-fields", "warn", [3]) in aux_checks(audit(aux_output))


def test_aux_pulse_missing_from_pprox(aux_output):
    edit_pprox(aux_output, lambda pp: pp["pprox"][4]["aux"].clear())
    report = audit(aux_output)
    (f,) = [f for f in unit_report(report)["findings"] if f["check"] == "aux-pulses"]
    assert f["trials"] == [4] and "missing from the pprox" in f["message"]


def test_aux_pulse_not_on_channel(aux_output):
    """A pulse in the pprox that isn't on the channel is reported; the stream
    check uses the channel, so it is unaffected."""

    def add(pp):
        pp["pprox"][2]["aux"].append({"name": "led", "interval": [0.0, 0.5]})

    edit_pprox(aux_output, add)
    report = audit(aux_output)
    assert aux_checks(report) == [("aux-pulses", "warn", [2])]
    assert "aren't on ADC4" in unit_report(report)["findings"][0]["message"]


def test_aux_pulse_end_differs(aux_output):
    def stretch(pp):
        pp["pprox"][4]["aux"][0]["interval"][1] += 0.1

    edit_pprox(aux_output, stretch)
    report = audit(aux_output)
    assert aux_checks(report) == [("aux-pulses", "warn", [4])]
    assert "end doesn't match" in unit_report(report)["findings"][0]["message"]


def test_aux_channel_not_in_arf(aux_output):
    edit_pprox(aux_output, lambda pp: pp["aux_tracks"]["led"].update(channel="ADC9"))
    assert aux_checks(audit(aux_output)) == [("aux-pulses", "warn", None)]


def test_aux_stream_without_messages(aux_output):
    edit_pprox(aux_output, lambda pp: pp["aux_tracks"]["led"].update(stream="channel3"))
    assert aux_checks(audit(aux_output)) == [("aux-stream", "info", None)]


def test_spurious_aux_pulse(aux_output, tmp_path):
    """A pulse on the channel with no message (here added to the ARF, in trial
    2) is a warning: it may be spurious, and it would be in the trial's aux."""

    def pulse(dset):
        trial = json.loads((aux_output / f"{UNIT}.pprox").read_text())["pprox"][2]
        start = round(trial["offset"] * 30000)
        dset[start : start + 15000] = dset[:].max()

    arf = modified_arf(tmp_path, pulse)
    found = aux_checks(audit_arf(aux_output, arf))
    assert ("aux-stream", "warn", [2]) in found
    assert ("aux-pulses", "warn", [2]) in found  # and the pprox lacks it


def test_message_without_aux_pulse(aux_output, tmp_path):
    """A message whose pulse never came (the LED didn't fire) is only noted:
    the pprox, which follows the channel, records no pulse."""

    def silence(dset):
        x = dset[:]
        for on, off in kilo.detect_pulses(x):
            x[on - 2 : off + 2] = np.median(x)  # including the edges
        dset[:] = x

    edit_pprox(aux_output, lambda pp: pp["pprox"][4]["aux"].clear())
    found = aux_checks(audit_arf(aux_output, modified_arf(tmp_path, silence)))
    assert found == [("aux-stream", "info", [4])]


@pytest.mark.slow
def test_e36_aux_pulses_clean():
    """The full E36 recording's 650 LED pulses, as group-kilo-spikes records
    them (here in a unit without spikes or a waveform file), agree with ADC4
    and the condition messages."""
    import pandas as pd
    from conftest import StubFinder
    from test_group_spikes_examples import stimulus_durations

    ex = EXAMPLES / "E36_5_1"
    if not (ex / "output").exists():
        pytest.skip("examples/E36_5_1 not present")
    finder = StubFinder(stimulus_durations(ex / "output"))
    with h5py.File(ex / "E36_5_1.arf", "r") as fp:
        trials = kilo.arf_to_trials(
            fp, finder, "ADC3", prepad=0.5, oeaudio_log=None, aux={"led": "ADC4"}
        )
    pprox = {
        "$schema": "https://meliza.org/spec:2/stimtrial.json#",
        "recording": "https://registry/resources/E36_5_1/",
        "aux_tracks": {"led": {"channel": "ADC4", "stream": "condition"}},
        "pprox": list(
            kilo.trials_to_pprox(pd.DataFrame(trials).assign(events=np.nan), 30000.0)
        ),
    }
    pprox = json.loads(json.dumps(pprox, default=list))
    unit = kilo_audit.Unit("E36_5_1_c0", Path("E36_5_1_c0.pprox"), pprox)
    report = kilo_audit.audit_recording(ex / "E36_5_1.arf", [unit])
    assert [f["check"] for f in unit_report(report, "E36_5_1_c0")["findings"]] == [
        "waveforms"
    ]
