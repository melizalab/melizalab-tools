# -*- mode: python -*-
"""Tests for prep-oephys. The bird colony and the registry are faked with an
httpx.MockTransport, and the user's answers are scripted."""

import datetime
import json

import httpx
import pytest

from dlab import oephys_prep as prep

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
    the registry's objections to each check, in turn."""

    def __init__(self, animals=(COBALT, PINK), check_errors=()):
        self.animals = list(animals)
        self.check_errors = list(check_errors)
        self.checked = []

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
        return httpx.Response(404, json={"detail": "not found"})


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
ACCEPT_ALL = ("", "", "1", "y", "", "", "", "", "", "", "", "")


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
    with pytest.raises(ValueError, match="holds 2 recordings"):
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
    answers = Answers("dmeliza", "", "2", "y", "", "2", "5", "", "", "", "", "")
    values, revised = run(tmp_path, client, answers)
    assert values["animal_uuid"] == PINK["uuid"]
    assert values["name"] == "C182_2_5"
    # the animal as named is the same; the band's color isn't checked
    assert revised == ["experimenter", "pen", "site"]


def test_a_different_animal_is_revised_and_renames(tmp_path, client, fake):
    fake.animals.append({**COBALT, "uuid": "0123abcd-0000-4000-8000-0000000000ff"})
    answers = Answers("", "0123abcd", "y", "", "", "", "", "", "", "", "")
    values, revised = run(tmp_path, client, answers)
    assert values["name"] == "0123abcd_1_4"
    assert revised == ["animal"]


def test_values_entered_for_jpresent_are_not_revisions(tmp_path, client):
    answers = Answers(
        "tbyron", "", "1", "y", "", "3", "1", "A1x32", "", "", "-2100", ""
    )
    values, revised = run(tmp_path, client, answers, presenter={})
    assert values["probe"] == "A1x32"
    assert values["z"] == -2100
    assert values["name"] == "C182_3_1"
    assert revised == []


def test_required_values_are_asked_again(tmp_path, client):
    answers = Answers(
        "", "tbyron", "", "1", "y", "", "", "1", "x", "2", "", "", "", "", ""
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
    first = ("tbyrn", "", "1", "y", "", "", "", "", "", "", "", "")
    second = ("tbyron", "", "1", "y", "", "", "", "", "", "", "", "")
    answers = Answers(*first, *second)
    values, revised = run(tmp_path, client, answers)
    assert values["experimenter"] == "tbyron"
    assert revised == []
    assert len(fake.checked) == 2
    assert "experimenter (registry username) [tbyrn]: " in answers.prompts


def test_a_name_the_user_gave_is_kept_when_asked_again(tmp_path, client, fake):
    fake.check_errors = [{"name": ["taken"]}]
    first = ("", "", "1", "y", "", "", "", "", "", "", "", "C182_1_4b")
    second = ("", "", "1", "y", "", "", "", "", "", "", "", "")
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


def test_arfx_command_overrides_only_the_presenters_values(tmp_path):
    session = prep.find_session(make_session(tmp_path))
    values = {"experimenter": "tbyron", "animal_uuid": COBALT["uuid"], "pen": 1}
    cmd = prep.arfx_command(session, values, ["protocol"], {}, tmp_path / "a.arf")
    assert "--override-presenter-metadata" not in cmd
    assert 'revised_after_recording=["protocol"]' in cmd
