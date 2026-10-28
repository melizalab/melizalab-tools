# -*- mode: python -*-
"""Tests for prep-oephys. The bird colony and the registry are faked with an
httpx.MockTransport, and the user's answers are scripted."""

import datetime
import json
import subprocess
import sys
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
    target = tmp_path / "share"
    target.mkdir()
    return recording, prep.Outputs.for_recording(recording, "C182_1_4", target)


def test_outputs(tmp_path):
    _, outputs = make_outputs(tmp_path)
    assert outputs.arf == tmp_path / "C182_1_4.arf"
    assert outputs.target_arf == tmp_path / "share/C182_1_4.arf"
    assert outputs.target_session == tmp_path / "share" / SESSION_DIR
    assert outputs.manifest == tmp_path / "share/C182_1_4.json"
    assert outputs.existing() == []
    outputs.manifest.write_text("{}")
    assert outputs.existing() == [outputs.manifest]


def test_convert_runs_the_command(tmp_path):
    arffile = tmp_path / "a.arf"
    cmd = [sys.executable, "-c", f"open({str(arffile)!r}, 'w').write('arf')"]
    prep.convert(cmd, arffile)
    assert arffile.read_text() == "arf"


def test_a_failed_conversion_leaves_nothing(tmp_path):
    arffile = tmp_path / "a.arf"
    script = f"open({str(arffile)!r}, 'w').write('arf'); raise SystemExit(2)"
    with pytest.raises(RuntimeError, match="exit status 2"):
        prep.convert([sys.executable, "-c", script], arffile)
    assert not arffile.exists()


def test_convert_without_arfx(tmp_path):
    with pytest.raises(RuntimeError, match="isn't installed"):
        prep.convert(["no-such-command-for-prep-oephys"], tmp_path / "a.arf")


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


def test_copy_to_target(tmp_path):
    recording, outputs = make_outputs(tmp_path)
    outputs.arf.write_text("arf")
    prep.copy_to_target(recording, outputs, file_sha1(outputs.arf))
    assert outputs.target_arf.read_text() == "arf"
    copied = outputs.target_session / "Record Node 101/experiment1/recording1"
    assert (copied / "structure.oebin").read_text() == "{}"


def test_a_copy_that_differs_is_removed(tmp_path):
    recording, outputs = make_outputs(tmp_path)
    outputs.arf.write_text("arf")
    with pytest.raises(OSError, match="doesn't match"):
        prep.copy_to_target(recording, outputs, "0" * 40)
    assert outputs.existing() == [outputs.arf]


def test_write_manifest(tmp_path):
    recording, outputs = make_outputs(tmp_path)
    resource = {"name": "C182_1_4", "sha1": "ab" * 20}
    prep.write_manifest(recording, outputs, resource)
    manifest = json.loads(outputs.manifest.read_text())
    assert manifest["name"] == "C182_1_4"
    assert manifest["sha1"] == "ab" * 20
    assert manifest["source"]["session"] == str(recording.path.resolve())
    assert manifest["source"]["arf"] == str(outputs.arf.resolve())
    assert manifest["session"] == SESSION_DIR
    assert manifest["prepared_by"].startswith("prep-oephys ")


@pytest.fixture
def script_env(tmp_path, monkeypatch, fake, arf_metadata):
    """Runs the script against the fakes: the presenter logged PRESENTER, the
    user accepts every default, and arfx-oephys writes 'arf'."""
    import nbank.util

    real_client = httpx.Client

    def fake_client(*args, **kwargs):
        return real_client(*args, transport=httpx.MockTransport(fake), **kwargs)

    monkeypatch.setattr(prep.httpx, "Client", fake_client)
    monkeypatch.setattr(nbank.util, "Client", fake_client)
    monkeypatch.setattr(nbank.util, "check_registry_version", lambda *a: None)
    monkeypatch.setattr(prep, "read_presenter_metadata", lambda path: PRESENTER)
    monkeypatch.setattr("builtins.input", Answers(*ACCEPT_ALL))
    commands = []

    def run(cmd):
        commands.append(cmd)
        Path(cmd[2]).write_text("arf")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(prep.subprocess, "run", run)
    session = make_session(tmp_path)
    target = tmp_path / "share"
    target.mkdir()
    return session, target, commands


def test_script(script_env, tmp_path):
    session, target, commands = script_env
    assert prep.script(["-r", REGISTRY, "convert", str(session), str(target)]) == 0
    [cmd] = commands
    assert cmd[:5] == [
        "arfx-oephys",
        "-f",
        str(tmp_path / "C182_1_4.arf"),
        "-T",
        "EXTRAC_RAW",
    ]
    assert (target / "C182_1_4.arf").read_text() == "arf"
    assert (target / SESSION_DIR).is_dir()
    manifest = json.loads((target / "C182_1_4.json").read_text())
    assert manifest["sha1"] == file_sha1(target / "C182_1_4.arf")


def test_script_dry_run_writes_nothing(script_env, tmp_path, capsys):
    session, target, commands = script_env
    assert (
        prep.script(["-r", REGISTRY, "convert", "--dry-run", str(session), str(target)])
        == 0
    )
    assert commands == []
    assert list(target.iterdir()) == []
    assert capsys.readouterr().out.rstrip().endswith(str(session))


def test_script_datatype(script_env):
    session, target, commands = script_env
    argv = ["-r", REGISTRY, "convert", "-T", "EXTRAC_HP", str(session), str(target)]
    assert prep.script(argv) == 0
    [cmd] = commands
    assert cmd[3:5] == ["-T", "EXTRAC_HP"]


def test_script_wont_overwrite(script_env, tmp_path):
    session, target, commands = script_env
    (target / "C182_1_4.json").write_text("{}")
    assert prep.script(["-r", REGISTRY, "convert", str(session), str(target)]) == 1
    assert commands == []


def test_script_removes_an_arf_the_registry_refuses(script_env, tmp_path, fake):
    session, target, _ = script_env
    # the first check (of the values) passes, the second (of the ARF) doesn't
    fake.check_errors = [{}, {"sha1": ["already registered as C182_1_3"]}]
    assert prep.script(["-r", REGISTRY, "convert", str(session), str(target)]) == 1
    assert not (tmp_path / "C182_1_4.arf").exists()
    assert list(target.iterdir()) == []


def test_script_reuses_the_arf_of_a_failed_copy(script_env, tmp_path):
    session, target, commands = script_env
    (tmp_path / "C182_1_4.arf").write_text("arf")
    assert prep.script(["-r", REGISTRY, "convert", str(session), str(target)]) == 0
    assert commands == []
    assert (target / "C182_1_4.arf").read_text() == "arf"
    assert (target / "C182_1_4.json").exists()


def test_script_keeps_a_reused_arf_the_registry_refuses(script_env, tmp_path, fake):
    session, target, commands = script_env
    (tmp_path / "C182_1_4.arf").write_text("arf")
    fake.check_errors = [{}, {"sha1": ["already registered as C182_1_3"]}]
    assert prep.script(["-r", REGISTRY, "convert", str(session), str(target)]) == 1
    assert commands == []
    assert (tmp_path / "C182_1_4.arf").exists()


@pytest.fixture
def prepared(script_env, tmp_path, fake):
    """A recording that prep-oephys has converted and copied, and whose ARF is
    registered, in an archive. Returns the manifest's path."""
    session, target, _ = script_env
    assert prep.script(["-r", REGISTRY, "convert", str(session), str(target)]) == 0
    manifest = target / "C182_1_4.json"
    sha1 = json.loads(manifest.read_text())["sha1"]
    fake.resources["C182_1_4"] = {
        "name": "C182_1_4",
        "sha1": sha1,
        "locations": ["chorus"],
    }
    return manifest


def cleanup(manifest, answer="y"):
    return prep.cleanup(Answers(answer), REGISTRY, manifest)


def test_cleanup(prepared, tmp_path):
    assert cleanup(prepared) == 0
    assert not (tmp_path / "C182_1_4.arf").exists()
    assert not (tmp_path / SESSION_DIR).exists()
    # the copies in the target are left for the sorting workstation
    assert (tmp_path / "share" / SESSION_DIR).is_dir()


def test_cleanup_asks_first(prepared, tmp_path):
    assert cleanup(prepared, answer="n") == 1
    assert (tmp_path / "C182_1_4.arf").exists()
    assert (tmp_path / SESSION_DIR).is_dir()


def test_cleanup_after_the_arf_is_gone(prepared, tmp_path):
    (tmp_path / "C182_1_4.arf").unlink()
    assert cleanup(prepared) == 0
    assert not (tmp_path / SESSION_DIR).exists()


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        (lambda fake, tmp: fake.resources.clear(), "isn't registered"),
        (
            lambda fake, tmp: fake.resources["C182_1_4"].update(sha1="0" * 40),
            "different hash",
        ),
        (
            lambda fake, tmp: fake.resources["C182_1_4"].update(locations=[]),
            "no location",
        ),
        (
            lambda fake, tmp: (tmp / "C182_1_4.arf").write_text("other"),
            "isn't the ARF that was prepared",
        ),
    ],
)
def test_cleanup_needs_the_arf_registered(
    prepared, tmp_path, fake, caplog, change, problem
):
    change(fake, tmp_path)
    assert cleanup(prepared) == 1
    assert problem in caplog.text
    assert (tmp_path / SESSION_DIR).is_dir()


def test_cleanup_only_on_the_recording_computer(prepared, tmp_path, caplog):
    manifest = json.loads(prepared.read_text())
    manifest["source"]["host"] = "elsewhere"
    prepared.write_text(json.dumps(manifest))
    assert cleanup(prepared) == 1
    assert "prepared on elsewhere" in caplog.text
    assert (tmp_path / SESSION_DIR).is_dir()


def test_cleanup_command(prepared, tmp_path, monkeypatch):
    monkeypatch.setattr("builtins.input", Answers("y"))
    assert prep.script(["-r", REGISTRY, "cleanup", str(prepared)]) == 0
    assert not (tmp_path / SESSION_DIR).exists()
