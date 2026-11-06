# -*- mode: python -*-
"""Tests for prep-oephys. The bird colony and the registry are faked with an
httpx.MockTransport, and the user's answers are scripted."""

import datetime
import json
import sys
import time
from pathlib import Path

import httpx
import pytest

from dlab import oephys_prep as prep
from dlab.abfzip import file_sha1

REGISTRY = "https://example.org/neurobank/"
COLONY = prep.colony_url(REGISTRY)
SESSION_DIR = "C182_2026-02-06_11-27-05_chorus"
# what oeaudio-present logs (arfx's presenter_metadata leaves out the animal)
PRESENTER = {
    "experimenter": "tbyron",
    "protocol": "chorus",
    "hemisphere": "L",
    "pen": 1,
    "site": 4,
    "x": 893.4,
    "y": -749.5,
    "z": -2500,
}
COBALT = {
    "name": "zebf_cobalt_182",
    "uuid": "a3e6c1b2-0000-4000-8000-000000000001",
    "species": "Zebra finch",
    "sex": "M",
    "alive": False,
    "last_location": "Gilmer 478/Box 9",
    "band_color": "cobalt",
    "band_number": 182,
}
PINK = {
    **COBALT,
    "name": "zebf_pink_182",
    "uuid": "f00dcafe-0000-4000-8000-000000000002",
    "band_color": "pink",
}
EVENTS = [
    {"date": "2025-10-01", "status": "hatched", "description": ""},
    {"date": "2026-01-20", "status": "moved", "description": "to Box 9"},
    {"date": "2026-02-06", "status": "died", "description": "terminal recording"},
]


def make_session(tmp_path, *recordings, name=SESSION_DIR):
    session = tmp_path / name
    for recording in recordings or ("experiment1/recording1",):
        path = session / "Record Node 101" / recording
        path.mkdir(parents=True)
        (path / "structure.oebin").write_text("{}")
    return session


class Fake:
    """The colony and the registry. Records the bodies checked; check_errors are
    the registry's objections to each check, in turn; resources are the
    registered resources, by name."""

    def __init__(self, animals=(COBALT, PINK), check_errors=()):
        self.animals = list(animals)
        self.check_errors = list(check_errors)
        self.checked = []
        self.resources = {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        params = request.url.params
        if path == "/birds/api/animals/":
            if "band" in params:
                found = [
                    a for a in self.animals if str(a["band_number"]) == params["band"]
                ]
            else:
                found = [
                    a for a in self.animals if a["uuid"].startswith(params["uuid"])
                ]
            return httpx.Response(200, json=found)
        for animal in self.animals:
            base = f"/birds/api/animals/{animal['uuid']}/"
            if path == base:
                return httpx.Response(200, json=animal)
            if path == base + "events/":
                return httpx.Response(200, json=EVENTS)
        if path == "/neurobank/check/resources/":
            self.checked.append(json.loads(request.content))
            errors = self.check_errors.pop(0) if self.check_errors else {}
            return httpx.Response(200, json=[errors])
        for name, record in self.resources.items():
            if path == f"/neurobank/resources/{name}/":
                return httpx.Response(200, json=record)
        return httpx.Response(404, json={"detail": "not found"})


real_arfx_oephys = prep.arfx_oephys


@pytest.fixture(autouse=True)
def arfx_oephys(monkeypatch):
    """arfx-oephys is run by name, whatever this environment has installed"""
    monkeypatch.setattr(prep, "arfx_oephys", lambda: "arfx-oephys")


def test_arfx_oephys_from_this_environment(tmp_path, monkeypatch):
    script = tmp_path / "arfx-oephys"
    script.write_text("#!/bin/sh\n")
    script.chmod(0o755)
    monkeypatch.setattr(prep.sysconfig, "get_path", lambda name: str(tmp_path))
    assert real_arfx_oephys() == str(script)


def test_arfx_oephys_from_the_path(tmp_path, monkeypatch):
    monkeypatch.setattr(prep.sysconfig, "get_path", lambda name: str(tmp_path))
    assert real_arfx_oephys() == "arfx-oephys"


@pytest.fixture
def fake():
    return Fake()


@pytest.fixture
def client(fake):
    with httpx.Client(transport=httpx.MockTransport(fake)) as client:
        yield client


class Answers:
    """Scripted answers to the prompts, which are recorded"""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.prompts = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answers.pop(0)


# answers that accept every default, choosing cobalt 182 from the two animals
ACCEPT_ALL = ("", "", "1", "y", "", "", "", "", "", "", "", "", "", "")


def run(tmp_path, client, answers, presenter=PRESENTER):
    session = prep.find_session(make_session(tmp_path))
    return prep.prepare(answers, client, COLONY, client, REGISTRY, session, presenter)


def test_find_session(tmp_path):
    path = make_session(tmp_path)
    session = prep.find_session(path)
    assert session.path == path
    assert session.recording == path / "Record Node 101/experiment1/recording1"
    assert session.animal == "C182"
    assert session.time == datetime.datetime(2026, 2, 6, 11, 27, 5)
    assert session.protocol == "chorus"


def test_find_session_from_the_recording_directory(tmp_path):
    path = make_session(tmp_path)
    recording = path / "Record Node 101/experiment1/recording1"
    assert prep.find_session(recording).path == path


def test_find_session_refuses_several_recordings(tmp_path):
    path = make_session(tmp_path, "experiment1/recording1", "experiment1/recording2")
    with pytest.raises(ValueError, match=r"holds 2 recordings.*still working out"):
        prep.find_session(path)


def test_find_session_without_a_recording(tmp_path):
    with pytest.raises(ValueError, match="no open-ephys recording"):
        prep.find_session(tmp_path)


def test_find_session_needs_the_session_directory_name(tmp_path):
    path = make_session(tmp_path, name="C182_chorus")
    with pytest.raises(ValueError, match="no session directory"):
        prep.find_session(path)


def test_recorded_values_from_the_directory_name(tmp_path):
    session = prep.find_session(make_session(tmp_path))
    assert prep.recorded_values(session, {}) == {
        "animal": "C182",
        "protocol": "chorus",
    }


def test_recorded_protocol_is_the_presenters(tmp_path, caplog):
    session = prep.find_session(make_session(tmp_path))
    values = prep.recorded_values(session, {"protocol": "song"})
    assert values["protocol"] == "song"
    assert "named for 'chorus'" in caplog.text


@pytest.mark.parametrize(
    ("animal", "params"),
    [
        ("C182", {"band": "182"}),
        ("182", {"band": "182"}),
        ("a3e6c1b2", {"uuid": "a3e6c1b2"}),
    ],
)
def test_find_animals_by_band_number_or_uuid(animal, params):
    seen = []

    def handler(request):
        seen.append(dict(request.url.params))
        return httpx.Response(200, json=[])

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        prep.find_animals(client, COLONY, animal)
    assert seen == [params]


@pytest.mark.parametrize(
    ("animal", "name"),
    [
        ("c182", "C182_1_4"),
        ("A3E6C1B2-0000", "a3e6c1b2_1_4"),
        ("a3e6c1b2", "a3e6c1b2_1_4"),
    ],
)
def test_default_name(animal, name):
    assert prep.default_name(animal, 1, 4) == name


def test_animal_summary_shows_the_events_nearest_the_recording(client):
    when = datetime.datetime(2026, 2, 6, 11, 27, 5)
    summary = prep.animal_summary(client, COLONY, COBALT["uuid"], when)
    assert "zebf_cobalt_182" in summary
    assert "not alive" in summary
    assert "2026-02-06: died (terminal recording)" in summary
    assert summary.endswith("2026-02-06: this recording")


def test_accepting_what_was_recorded(tmp_path, client, fake):
    answers = Answers(*ACCEPT_ALL)
    values, revised = run(tmp_path, client, answers)
    assert revised == []
    assert values["animal_uuid"] == COBALT["uuid"]
    assert values["name"] == "C182_1_4"
    assert "experimenter (registry username) [tbyron]: " in answers.prompts
    [[body]] = fake.checked
    assert body == {
        "name": "C182_1_4",
        "dtype": "extracellular-arf",
        "experimenter": "tbyron",
        "animal": COBALT["uuid"],
        "metadata": {
            "protocol": "chorus",
            "hemisphere": "L",
            "pen": 1,
            "site": 4,
            "x": 893.4,
            "y": -749.5,
            "z": -2500,
        },
    }


def test_revised_values_are_listed(tmp_path, client):
    # experimenter, animal (pink 182 instead of cobalt), pen and site changed
    answers = Answers("dmeliza", "", "2", "y", "", "2", "5", "", "", "", "", "", "", "")
    values, revised = run(tmp_path, client, answers)
    assert values["animal_uuid"] == PINK["uuid"]
    assert values["name"] == "C182_2_5"
    # the animal as named is the same; the band's color isn't checked
    assert revised == ["experimenter", "pen", "site"]


def test_a_different_animal_is_revised_and_renames(tmp_path, client, fake):
    fake.animals.append({**COBALT, "uuid": "0123abcd-0000-4000-8000-0000000000ff"})
    answers = Answers("", "0123abcd", "y", "", "", "", "", "", "", "", "", "", "")
    values, revised = run(tmp_path, client, answers)
    assert values["name"] == "0123abcd_1_4"
    assert revised == ["animal"]


def test_values_entered_for_jpresent_are_not_revisions(tmp_path, client):
    answers = Answers(
        "tbyron", "", "1", "y", "", "3", "1", "A1x32", "L", "NCM", "", "", "-2100", ""
    )
    values, revised = run(tmp_path, client, answers, presenter={})
    assert values["probe"] == "A1x32"
    assert values["hemisphere"] == "L"
    assert values["location"] == "NCM"
    assert "location (brain area; provisional until histology): " in answers.prompts
    assert values["z"] == -2100
    assert values["name"] == "C182_3_1"
    assert revised == []


def test_required_values_are_asked_again(tmp_path, client):
    answers = Answers(
        "", "tbyron", "", "1", "y", "", "", "1", "x", "2", "", "", "", "", "", "", ""
    )
    values, _ = run(tmp_path, client, answers, presenter={})
    assert values["pen"] == 1
    assert values["site"] == 2


def test_an_animal_that_isnt_found_is_asked_again(tmp_path, client):
    answers = Answers("", "Y999", "C182", "1", "y", *ACCEPT_ALL[4:])
    values, _ = run(tmp_path, client, answers)
    assert values["animal_uuid"] == COBALT["uuid"]


def test_registry_objections_ask_again_with_the_answers_as_defaults(
    tmp_path, client, fake
):
    fake.check_errors = [{"experimenter": ["no such user 'tbyrn'"], "sha1": ["x"]}]
    first = ("tbyrn", "", "1", "y", "", "", "", "", "", "", "", "", "", "")
    second = ("tbyron", "", "1", "y", "", "", "", "", "", "", "", "", "", "")
    answers = Answers(*first, *second)
    values, revised = run(tmp_path, client, answers)
    assert values["experimenter"] == "tbyron"
    assert revised == []
    assert len(fake.checked) == 2
    assert "experimenter (registry username) [tbyrn]: " in answers.prompts


def test_a_name_the_user_gave_is_kept_when_asked_again(tmp_path, client, fake):
    fake.check_errors = [{"name": ["taken"]}]
    first = ("", "", "1", "y", "", "", "", "", "", "", "", "", "", "C182_1_4b")
    second = ("", "", "1", "y", "", "", "", "", "", "", "", "", "", "")
    values, _ = run(tmp_path, client, Answers(*first, *second))
    assert values["name"] == "C182_1_4b"


def test_arfx_command(tmp_path):
    session = prep.find_session(make_session(tmp_path))
    values = {
        "experimenter": "tbyron",
        "animal_uuid": COBALT["uuid"],
        "protocol": "chorus",
        "pen": 2,
        "site": 4,
        "probe": "64",
        "x": None,
        "y": None,
        "z": -2500,
    }
    cmd = prep.arfx_command(session, values, ["pen"], PRESENTER, tmp_path / "a.arf")
    assert cmd == [
        "arfx-oephys",
        "-f",
        str(tmp_path / "a.arf"),
        "-T",
        "EXTRAC_RAW",
        "--experimenter",
        "tbyron",
        "--animal",
        COBALT["uuid"],
        "-k",
        'protocol="chorus"',
        "-k",
        "pen=2",
        "-k",
        "site=4",
        "-k",
        'probe="64"',
        "-k",
        "z=-2500",
        "-k",
        'revised_after_recording=["pen"]',
        "--override-presenter-metadata",
        str(session.path),
    ]


def test_arfx_command_datatype(tmp_path):
    session = prep.find_session(make_session(tmp_path))
    values = {"experimenter": "tbyron", "animal_uuid": COBALT["uuid"]}
    cmd = prep.arfx_command(session, values, [], {}, tmp_path / "a.arf", "EXTRAC_HP")
    assert cmd[3:5] == ["-T", "EXTRAC_HP"]


def test_arfx_command_overrides_only_the_presenters_values(tmp_path):
    session = prep.find_session(make_session(tmp_path))
    values = {"experimenter": "tbyron", "animal_uuid": COBALT["uuid"], "pen": 1}
    cmd = prep.arfx_command(session, values, ["protocol"], {}, tmp_path / "a.arf")
    assert "--override-presenter-metadata" not in cmd
    assert 'revised_after_recording=["protocol"]' in cmd


# the ARF's metadata, as arfx's reader returns them for the confirmed values
ARF_METADATA = {
    "experimenter": "tbyron",
    "animal": COBALT["uuid"],
    "processed_by": "org.meliza.arfx/arfx-oephys 3.1.1",
    **{k: v for k, v in PRESENTER.items() if k != "experimenter"},
}
CONFIRMED = {
    "name": "C182_1_4",
    "experimenter": "tbyron",
    "animal": "C182",
    "animal_uuid": COBALT["uuid"],
    "protocol": "chorus",
    "pen": 1,
    "site": 4,
    "probe": None,
    "x": 893.4,
    "y": -749.5,
    "z": -2500,
}


@pytest.fixture
def arf_metadata(monkeypatch):
    metadata = dict(ARF_METADATA)
    monkeypatch.setattr(prep, "read_arf_metadata", lambda path: dict(metadata))
    return metadata


def make_outputs(tmp_path):
    recording = prep.find_session(make_session(tmp_path))
    staging = tmp_path / "staging"
    target = tmp_path / "share"
    staging.mkdir()
    target.mkdir()
    outputs = prep.Outputs.for_recording(recording, "C182_1_4", staging, target)
    return recording, outputs


def test_outputs(tmp_path):
    _, outputs = make_outputs(tmp_path)
    assert outputs.arf == tmp_path / "staging/C182_1_4.arf"
    assert outputs.conversion_log == tmp_path / "staging/C182_1_4.arfx-oephys.log"
    assert outputs.hashes == tmp_path / "staging/C182_1_4.session.json"
    assert outputs.target_session == tmp_path / "share" / SESSION_DIR
    assert outputs.target_arf == tmp_path / "share/C182_1_4.arf"
    assert outputs.manifest == tmp_path / "share/C182_1_4.json"


def test_space_warnings(tmp_path, monkeypatch):
    recording, _ = make_outputs(tmp_path)
    (recording.recording / "continuous.dat").write_bytes(b"x" * 1000)
    usage = prep.shutil.disk_usage(tmp_path)
    monkeypatch.setattr(
        prep.shutil, "disk_usage", lambda path: usage._replace(free=1500)
    )
    warnings = prep.space_warnings(recording, tmp_path / "staging", tmp_path / "share")
    # staging is on the recording's drive here; the target needs room for two
    assert len(warnings) == 2
    assert "same drive" in warnings[0]
    assert str(tmp_path / "share") in warnings[1]


def test_lower_priority(monkeypatch):
    monkeypatch.setattr(prep.shutil, "which", lambda name: f"/usr/bin/{name}")
    assert prep.lower_priority(["cmd"]) == [
        "ionice",
        "-c2",
        "-n7",
        "nice",
        "-n",
        "10",
        "cmd",
    ]
    monkeypatch.setattr(prep.shutil, "which", lambda name: None)
    assert prep.lower_priority(["cmd"]) == ["cmd"]


def write_with(arffile, *, status=0, then="pass"):
    """A command that writes 'arf' to arffile's temporary name, says so, runs
    then, and exits"""
    path = str(prep.partial(arffile))
    script = (
        f"open({path!r}, 'a').write('arf'); print('wrote it', flush=True); "
        f"{then}; raise SystemExit({status})"
    )
    return [sys.executable, "-c", script]


@pytest.fixture
def conversion(tmp_path, monkeypatch):
    """Starts a conversion with a command, at the usual priority"""
    monkeypatch.setattr(prep, "lower_priority", lambda cmd: cmd)
    arffile = tmp_path / "a.arf"
    return lambda cmd: prep.Conversion(cmd, arffile, tmp_path / "a.log")


def test_conversion(tmp_path, conversion):
    arffile = tmp_path / "a.arf"
    conversion(write_with(arffile)).finish()
    assert arffile.read_text() == "arf"
    assert not prep.partial(arffile).exists()
    assert (tmp_path / "a.log").read_text() == "wrote it\n"


def test_a_conversion_starts_over_from_a_partial_arf(tmp_path, conversion):
    arffile = tmp_path / "a.arf"
    prep.partial(arffile).write_text("half an ")
    conversion(write_with(arffile)).finish()
    assert arffile.read_text() == "arf"


def test_a_failed_conversion_leaves_its_log(tmp_path, conversion):
    arffile = tmp_path / "a.arf"
    running = conversion(write_with(arffile, status=2))
    with pytest.raises(RuntimeError, match=r"exit status 2\).*ends:\nwrote it"):
        running.finish()
    assert list(tmp_path.iterdir()) == [tmp_path / "a.log"]


def test_a_conversion_without_arfx(tmp_path, conversion):
    with pytest.raises(RuntimeError, match="isn't installed"):
        conversion(["no-such-command-for-prep-oephys"])
    assert list(tmp_path.iterdir()) == []


def test_stopping_a_conversion(tmp_path, conversion):
    arffile = tmp_path / "a.arf"
    running = conversion(write_with(arffile, then="import time; time.sleep(60)"))
    deadline = time.monotonic() + 10
    while "wrote it" not in (tmp_path / "a.log").read_text():
        assert time.monotonic() < deadline
        time.sleep(0.05)
    running.stop()
    assert running.process.poll() is not None
    assert not prep.partial(arffile).exists()
    assert not arffile.exists()


def test_check_arf(tmp_path, client, fake, arf_metadata):
    arffile = tmp_path / "a.arf"
    arffile.write_text("arf")
    resource = prep.check_arf(client, REGISTRY, arffile, CONFIRMED)
    metadata = {
        k: v for k, v in ARF_METADATA.items() if k not in ("experimenter", "animal")
    }
    assert resource == {
        "name": "C182_1_4",
        "dtype": "extracellular-arf",
        "sha1": file_sha1(arffile),
        "experimenter": "tbyron",
        "animal": COBALT["uuid"],
        "metadata": metadata,
    }
    assert fake.checked == [[resource]]


def test_check_arf_needs_the_confirmed_values(tmp_path, client, arf_metadata):
    arffile = tmp_path / "a.arf"
    arffile.write_text("arf")
    arf_metadata["pen"] = 2
    with pytest.raises(ValueError, match=r"differ from the confirmed values \(pen\)"):
        prep.check_arf(client, REGISTRY, arffile, CONFIRMED)


def test_check_arf_reports_the_registrys_objections(
    tmp_path, client, fake, arf_metadata
):
    arffile = tmp_path / "a.arf"
    arffile.write_text("arf")
    fake.check_errors = [
        {"sha1": ["already registered as C182_1_3"], "locations": ["x"]}
    ]
    with pytest.raises(ValueError, match="sha1: already registered") as err:
        prep.check_arf(client, REGISTRY, arffile, CONFIRMED)
    assert "locations" not in str(err.value)


DATA = "Record Node 101/experiment1/recording1/structure.oebin"


def test_copy_session(tmp_path):
    recording, outputs = make_outputs(tmp_path)
    hashes = prep.copy_session(recording, outputs)
    assert hashes == {DATA: file_sha1(recording.path / DATA)}
    assert (outputs.target_session / DATA).read_text() == "{}"
    assert not prep.partial(outputs.target_session).exists()
    kept = json.loads(outputs.hashes.read_text())
    assert kept == {"session": str(recording.path.resolve()), "files": hashes}


def test_copy_to_a_share_that_refuses_permissions_and_times(tmp_path, monkeypatch):
    # e.g. a CIFS mount without Unix extensions
    def refuse(*args, **kwargs):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr("os.chmod", refuse)
    monkeypatch.setattr("os.utime", refuse)
    recording, outputs = make_outputs(tmp_path)
    prep.copy_session(recording, outputs)
    assert (outputs.target_session / DATA).read_text() == "{}"


def corrupt_copies(monkeypatch, under: Path):
    """Files read back from under hash differently from what was written"""
    real = prep.file_sha1

    def sha1(path):
        return "0" * 40 if Path(path).is_relative_to(under) else real(path)

    monkeypatch.setattr(prep, "file_sha1", sha1)


def test_a_session_copy_that_differs_is_removed(tmp_path, monkeypatch):
    recording, outputs = make_outputs(tmp_path)
    corrupt_copies(monkeypatch, tmp_path / "share")
    with pytest.raises(OSError, match="doesn't match"):
        prep.copy_session(recording, outputs)
    assert list((tmp_path / "share").iterdir()) == []
    assert not outputs.hashes.exists()


def test_a_session_copy_is_done_once(tmp_path, monkeypatch):
    recording, outputs = make_outputs(tmp_path)
    hashes = prep.copy_session(recording, outputs)
    monkeypatch.setattr(prep, "copy_tree", None)
    assert prep.copy_session(recording, outputs) == hashes


def test_a_partial_session_copy_is_redone(tmp_path):
    recording, outputs = make_outputs(tmp_path)
    stale = prep.partial(outputs.target_session)
    stale.mkdir()
    (stale / "junk").write_text("")
    prep.copy_session(recording, outputs)
    assert not stale.exists()
    assert not (outputs.target_session / "junk").exists()


def test_a_session_copy_from_elsewhere_isnt_used(tmp_path):
    recording, outputs = make_outputs(tmp_path)
    outputs.target_session.mkdir()
    with pytest.raises(OSError, match="didn't copy it"):
        prep.copy_session(recording, outputs)
    outputs.hashes.write_text(json.dumps({"session": "/elsewhere", "files": {}}))
    with pytest.raises(OSError, match="copied from /elsewhere"):
        prep.copy_session(recording, outputs)


def test_copy_arf(tmp_path):
    _, outputs = make_outputs(tmp_path)
    outputs.arf.write_text("arf")
    prep.copy_arf(outputs, file_sha1(outputs.arf))
    assert outputs.target_arf.read_text() == "arf"
    assert not prep.partial(outputs.target_arf).exists()


def test_an_arf_copy_that_differs_is_removed(tmp_path, monkeypatch):
    _, outputs = make_outputs(tmp_path)
    outputs.arf.write_text("arf")
    sha1 = file_sha1(outputs.arf)
    corrupt_copies(monkeypatch, tmp_path / "share")
    with pytest.raises(OSError, match="doesn't match"):
        prep.copy_arf(outputs, sha1)
    assert list((tmp_path / "share").iterdir()) == []


def test_an_arf_already_in_the_target_must_match(tmp_path):
    _, outputs = make_outputs(tmp_path)
    outputs.arf.write_text("arf")
    outputs.target_arf.write_text("other")
    with pytest.raises(OSError, match="already there"):
        prep.copy_arf(outputs, file_sha1(outputs.arf))


def test_write_manifest(tmp_path):
    recording, outputs = make_outputs(tmp_path)
    resource = {"name": "C182_1_4", "sha1": "ab" * 20}
    prep.write_manifest(recording, outputs, resource, {DATA: "cd" * 20})
    manifest = json.loads(outputs.manifest.read_text())
    assert manifest["name"] == "C182_1_4"
    assert manifest["sha1"] == "ab" * 20
    assert manifest["source"]["session"] == str(recording.path.resolve())
    assert manifest["source"]["arf"] == str(outputs.arf.resolve())
    assert manifest["session"] == SESSION_DIR
    assert manifest["session_sha1"] == {DATA: "cd" * 20}
    assert manifest["prepared_by"].startswith("prep-oephys ")


def test_remove_session(tmp_path):
    recording, outputs = make_outputs(tmp_path)
    hashes = prep.copy_session(recording, outputs)
    prep.remove_session(recording, outputs, hashes)
    assert not recording.path.exists()


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        (
            lambda path: (
                (path / "experiment2").mkdir()
                or (path / "experiment2/x").write_text("")
            ),
            "weren't copied",
        ),
        (lambda path: (path / DATA).write_text("{} "), "has changed"),
    ],
)
def test_a_session_that_changed_isnt_removed(tmp_path, change, problem):
    recording, outputs = make_outputs(tmp_path)
    hashes = prep.copy_session(recording, outputs)
    change(recording.path)
    with pytest.raises(OSError, match=problem):
        prep.remove_session(recording, outputs, hashes)
    assert (recording.path / DATA).exists()


class FakeProcess:
    """arfx-oephys, which has written 'arf' and finished"""

    def __init__(self, cmd):
        arffile = Path(cmd[cmd.index("-f") + 1])
        arffile.write_text("arf")
        self.returncode = 0

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode


def arfx_args(cmd):
    """cmd, without what lowers its priority"""
    return cmd[cmd.index("arfx-oephys") :]


@pytest.fixture
def script_env(tmp_path, monkeypatch, fake, arf_metadata):
    """Runs the script against the fakes: the presenter logged PRESENTER, the
    user accepts every default, and arfx-oephys writes 'arf'. Returns the
    session directory, the staging directory, the target, the arfx-oephys
    commands run, and the argument list."""
    import nbank.util

    real_client = httpx.Client

    def fake_client(*args, **kwargs):
        return real_client(*args, transport=httpx.MockTransport(fake), **kwargs)

    monkeypatch.setattr(prep.httpx, "Client", fake_client)
    monkeypatch.setattr(nbank.util, "Client", fake_client)
    monkeypatch.setattr(nbank.util, "check_registry_version", lambda *a: None)
    monkeypatch.setattr(prep, "read_presenter_metadata", lambda path: PRESENTER)
    monkeypatch.setattr("builtins.input", Answers(*ACCEPT_ALL))
    # nice, ionice and arfx-oephys are found, whether or not they're installed
    monkeypatch.setattr(prep.shutil, "which", lambda name: f"/usr/bin/{name}")
    commands = []

    class Process(FakeProcess):
        def __init__(self, cmd, **kwargs):
            commands.append(cmd)
            super().__init__(cmd)

    monkeypatch.setattr(prep.subprocess, "Popen", Process)
    session = make_session(tmp_path)
    staging = tmp_path / "staging"
    target = tmp_path / "share"
    staging.mkdir()
    target.mkdir()
    argv = ["-r", REGISTRY, "convert", "--staging", str(staging)]
    return session, staging, target, commands, argv


def test_script(script_env, tmp_path):
    session, staging, target, commands, argv = script_env
    assert prep.script([*argv, str(session), str(target)]) == 0
    [cmd] = commands
    assert cmd[:6] == ["ionice", "-c2", "-n7", "nice", "-n", "10"]
    assert arfx_args(cmd)[:5] == [
        "arfx-oephys",
        "-f",
        str(staging / "C182_1_4.arf.partial"),
        "-T",
        "EXTRAC_RAW",
    ]
    assert (staging / "C182_1_4.arf").read_text() == "arf"
    assert (staging / "C182_1_4.arfx-oephys.log").exists()
    assert (target / "C182_1_4.arf").read_text() == "arf"
    assert (target / SESSION_DIR / DATA).read_text() == "{}"
    manifest = json.loads((target / "C182_1_4.json").read_text())
    assert manifest["sha1"] == file_sha1(target / "C182_1_4.arf")
    assert manifest["session_sha1"] == {DATA: file_sha1(target / SESSION_DIR / DATA)}
    assert not session.exists()


def test_script_keep_session(script_env):
    session, _, target, _, argv = script_env
    assert prep.script([*argv, "--keep-session", str(session), str(target)]) == 0
    assert (target / "C182_1_4.json").exists()
    assert (session / DATA).exists()


def test_script_dry_run_writes_nothing(script_env, capsys):
    session, staging, target, commands, argv = script_env
    assert prep.script([*argv, "--dry-run", str(session), str(target)]) == 0
    assert commands == []
    assert list(staging.iterdir()) == []
    assert list(target.iterdir()) == []
    assert capsys.readouterr().out.rstrip().endswith(str(session))


def test_script_datatype(script_env):
    session, _, target, commands, argv = script_env
    assert prep.script([*argv, "-T", "EXTRAC_HP", str(session), str(target)]) == 0
    [cmd] = commands
    assert arfx_args(cmd)[3:5] == ["-T", "EXTRAC_HP"]


@pytest.mark.parametrize("missing", ["staging", "share"])
def test_script_needs_the_directories(script_env, tmp_path, missing):
    session, _, target, _, argv = script_env
    (tmp_path / missing).rmdir()
    assert prep.script([*argv, str(session), str(target)]) == 1


def test_script_wont_prepare_twice(script_env):
    session, _, target, commands, argv = script_env
    (target / "C182_1_4.json").write_text("{}")
    assert prep.script([*argv, str(session), str(target)]) == 1
    assert commands == []
    assert session.exists()


def test_script_removes_an_arf_the_registry_refuses(script_env, fake):
    session, staging, target, _, argv = script_env
    # the first check (of the values) passes, the second (of the ARF) doesn't
    fake.check_errors = [{}, {"sha1": ["already registered as C182_1_3"]}]
    assert prep.script([*argv, str(session), str(target)]) == 1
    assert not (staging / "C182_1_4.arf").exists()
    # the session's copy can still be sorted
    assert (target / SESSION_DIR).is_dir()
    assert not (target / "C182_1_4.arf").exists()
    assert session.exists()


def test_script_resumes_after_a_failed_copy(script_env, monkeypatch):
    session, staging, target, commands, argv = script_env
    real_copy_arf = prep.copy_arf
    calls = []

    def fail_once(*args):
        calls.append(args)
        if len(calls) == 1:
            raise OSError("the share went away")
        return real_copy_arf(*args)

    monkeypatch.setattr(prep, "copy_arf", fail_once)
    assert prep.script([*argv, str(session), str(target)]) == 1
    assert (staging / "C182_1_4.arf").exists()
    assert session.exists()
    # the second run neither copies the session nor converts it again
    monkeypatch.setattr("builtins.input", Answers(*ACCEPT_ALL))
    monkeypatch.setattr(prep, "copy_tree", None)
    assert prep.script([*argv, str(session), str(target)]) == 0
    assert len(commands) == 1
    assert (target / "C182_1_4.arf").read_text() == "arf"
    assert not session.exists()


def test_script_keeps_a_reused_arf_the_registry_refuses(script_env, fake):
    session, staging, target, commands, argv = script_env
    (staging / "C182_1_4.arf").write_text("arf")
    fake.check_errors = [{}, {"sha1": ["already registered as C182_1_3"]}]
    assert prep.script([*argv, str(session), str(target)]) == 1
    assert commands == []
    assert (staging / "C182_1_4.arf").exists()


def test_script_converts_even_if_the_session_copy_fails(script_env, monkeypatch):
    session, staging, target, commands, argv = script_env
    real_copy_session = prep.copy_session

    def fail(*args):
        raise OSError("the share went away")

    monkeypatch.setattr(prep, "copy_session", fail)
    assert prep.script([*argv, str(session), str(target)]) == 1
    assert (staging / "C182_1_4.arf").exists()
    assert list(target.iterdir()) == []
    # the next run copies the session and uses the ARF
    monkeypatch.setattr(prep, "copy_session", real_copy_session)
    monkeypatch.setattr("builtins.input", Answers(*ACCEPT_ALL))
    assert prep.script([*argv, str(session), str(target)]) == 0
    assert len(commands) == 1
    assert (target / SESSION_DIR / DATA).exists()
    assert not session.exists()


def test_script_interrupted_stops_the_conversion(script_env, monkeypatch):
    session, staging, target, _, argv = script_env
    stopped = []

    class Running(FakeProcess):
        def __init__(self, cmd, **kwargs):
            super().__init__(cmd)
            self.returncode = None

        def terminate(self):
            stopped.append(True)
            self.returncode = -15

    def interrupt(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(prep.subprocess, "Popen", Running)
    monkeypatch.setattr(prep, "copy_session", interrupt)
    with pytest.raises(KeyboardInterrupt):
        prep.script([*argv, str(session), str(target)])
    assert stopped == [True]
    assert list(staging.iterdir()) == [staging / "C182_1_4.arfx-oephys.log"]
    assert session.exists()
