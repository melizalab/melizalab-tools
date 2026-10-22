# -*- mode: python -*-
"""Tests for dlab.abfzip (abf-zip), on three small ABF2 files from the
intracellular archive (tests/data/abf: header versions 2.4, 2.6 and 2.9)."""

import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path

import pytest

from dlab import abfzip

DATA = Path(__file__).parent / "data" / "abf"
# the catalog of the test files, in the order they were recorded
CATALOG = [
    {
        "path": "2014_09_11_0013.abf",
        "protocol": "gapfree-2",
        "start": "2014-09-11T14:54:49.800",
        "mode": "gap-free",
        "sampling_rate": 50000,
        "channels": [
            {"name": "V2_pri", "units": "pA", "clamp": "voltage"},
            {"name": "I2_sec", "units": "mV", "clamp": "voltage"},
        ],
    },
    {
        "path": "2017_06_07_0044.abf",
        "protocol": "ch2-vclamp",
        "start": "2017-06-07T17:36:25.502",
        "mode": "episodic",
        "sampling_rate": 10000,
        "sweeps": 1,
        "channels": [
            {"name": "V2_pri", "units": "pA", "clamp": "voltage"},
            {"name": "I2_sec", "units": "mV", "clamp": "voltage"},
            {"name": "T_bath", "units": "C"},
            {"name": "ValveLink", "units": "V"},
        ],
    },
    {
        "path": "2025_05_07_0003.abf",
        "protocol": "250506_ramp",
        "start": "2025-05-07T18:43:46.204",
        "mode": "episodic",
        "sampling_rate": 10000,
        "sweeps": 2,
        # recorded without the amplifier's telegraphs
        "channels": [{"name": "IN 0", "units": "A"}],
    },
]
# the software that wrote each test file
SOFTWARE = {
    "2014_09_11_0013.abf": "Clampex 10.4.1.10",
    "2017_06_07_0044.abf": "Clampex 10.7.0.3",
    "2025_05_07_0003.abf": "Clampex 11.2.2.17",
}
METADATA = {
    "experimenter": "anc4kj",
    "animal": "ca7cb913-0888-4861-8b3b-a954b16f42ce",
    "cell": 5,
}


@pytest.fixture
def single(tmp_path):
    """A recording directory with two files written by the same software"""
    src = tmp_path / "rec"
    src.mkdir()
    shutil.copy(DATA / "2017_06_07_0044.abf", src)
    shutil.copy(DATA / "2017_06_07_0044.abf", src / "2017_06_07_0045.abf")
    (src / "notes.txt").write_text("cell 5\n")
    return src


@pytest.fixture
def recording(tmp_path):
    """A recording directory with the test files (the newest in a
    subdirectory), notes, and an image stack"""
    src = tmp_path / "rec"
    (src / "later").mkdir(parents=True)
    (src / "_1").mkdir()
    shutil.copy(DATA / "2014_09_11_0013.abf", src)
    shutil.copy(DATA / "2017_06_07_0044.abf", src)
    shutil.copy(DATA / "2025_05_07_0003.abf", src / "later")
    (src / "notes.txt").write_text("cell 5\n")
    (src / "_1" / "img_MMStack_Pos0_metadata.txt").write_text("{}\n")
    return src


def sha1(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def catalog_in(src):
    """The catalog of the recording fixture: CATALOG with its paths, then its
    other files by path, each with its hash"""
    out = [dict(r) for r in CATALOG]
    out[2]["path"] = "later/" + out[2]["path"]
    out += [{"path": "_1/img_MMStack_Pos0_metadata.txt"}, {"path": "notes.txt"}]
    for record in out:
        record["sha1"] = sha1((src / record["path"]).read_bytes())
    return out


@pytest.mark.parametrize(
    "path,expected",
    [
        (
            "C:\\Users\\Meliza Lab\\Documents\\Molecular Devices\\pCLAMP\\Params"
            "\\Redesign\\ch2-episode-01-spiketest-redux.pro",
            "ch2-episode-01-spiketest-redux",
        ),
        ("C:/Params/IV_steps_500ms.PRO", "IV_steps_500ms"),
        ("ramp", "ramp"),
        ("", None),
    ],
)
def test_header_to_protocol(path, expected):
    """The stem of the protocol file, from a Windows or POSIX path"""
    assert abfzip.header_to_protocol({"sProtocolPath": path}) == expected


@pytest.mark.parametrize("record", CATALOG, ids=lambda r: r["path"])
def test_header_to_record(record):
    """Protocol, start time, and settings from the header; the start time is
    exact to the millisecond (neo's datetime has float error, e.g. .203999)"""
    with open(DATA / record["path"], "rb") as fp:
        header = abfzip.read_header(fp)
    assert abfzip.header_to_record(header, record["path"]) == record


@pytest.mark.parametrize(
    "name,version,expected",
    [
        ("Clampex", 0x0B020211, "Clampex 11.2.2.17"),
        ("Clampex", 0, "Clampex"),
        ("", 0x0B020211, None),
    ],
)
def test_header_to_software(name, version, expected):
    header = {"sCreatorName": name, "uCreatorVersion": version}
    assert abfzip.header_to_software(header) == expected


@pytest.mark.parametrize("record", CATALOG, ids=lambda r: r["path"])
def test_header_software(record):
    with open(DATA / record["path"], "rb") as fp:
        header = abfzip.read_header(fp)
    assert abfzip.header_to_software(header) == SOFTWARE[record["path"]]


def test_headers_to_software(single):
    headers = abfzip.directory_to_headers(single)
    assert abfzip.headers_to_software(headers) == "Clampex 10.7.0.3"


def test_headers_to_software_mixed(recording):
    """The files of a recording must be written by the same software"""
    headers = abfzip.directory_to_headers(recording)
    with pytest.raises(ValueError, match=r"different software: Clampex 10\.4"):
        abfzip.headers_to_software(headers)


def test_headers_to_software_none():
    header = {"sCreatorName": "", "uCreatorVersion": 0}
    assert abfzip.headers_to_software([("a.abf", header)]) is None


def test_header_without_date():
    header = {"sProtocolPath": "", "rec_datetime": None, "uFileStartDate": 0}
    assert abfzip.header_to_record(header, "a.abf") == {"path": "a.abf"}


@pytest.mark.parametrize(
    "contents", [b"", b"ABF2", b"ABF2" + bytes(100), b"PK\x03\x04"]
)
def test_read_header_invalid(tmp_path, contents):
    path = tmp_path / "bad.abf"
    path.write_bytes(contents)
    with open(path, "rb") as fp, pytest.raises(ValueError, match="ABF"):
        abfzip.read_header(fp)


def test_directory_to_catalog(recording):
    """Every file, with its hash: ABF files anywhere in the directory in the
    order they were recorded, then the others by path."""
    assert abfzip.directory_to_catalog(recording) == catalog_in(recording)


def test_catalog_hash_is_of_the_file(recording):
    catalog = abfzip.directory_to_catalog(recording)
    record = next(r for r in catalog if r["path"] == "later/2025_05_07_0003.abf")
    assert record["sha1"] == sha1((DATA / "2025_05_07_0003.abf").read_bytes())


def test_directory_to_catalog_uppercase(recording):
    (recording / "2017_06_07_0044.abf").rename(recording / "2017_06_07_0044.ABF")
    paths = [r["path"] for r in abfzip.directory_to_catalog(recording)]
    assert "2017_06_07_0044.ABF" in paths


def test_directory_to_catalog_no_abf(tmp_path):
    (tmp_path / "img.tif").write_bytes(b"II*\x00")
    with pytest.raises(ValueError, match="no ABF files"):
        abfzip.directory_to_catalog(tmp_path)


def test_directory_to_catalog_bad_abf(recording):
    (recording / "notes.abf").write_text("not an abf\n")
    with pytest.raises(ValueError, match=r"notes\.abf"):
        abfzip.directory_to_catalog(recording)


def test_build(recording, tmp_path):
    """The archive holds every file in the directory, stored, and
    metadata.json at the top level."""
    doc = abfzip.metadata_document(METADATA, catalog_in(recording))
    out = abfzip.build(recording, doc, tmp_path / "rec.zip")
    with zipfile.ZipFile(out) as zf:
        assert sorted(zf.namelist()) == sorted(
            [*abfzip.directory_files(recording), "metadata.json"]
        )
        assert all(i.compress_type == zipfile.ZIP_STORED for i in zf.infolist())
        assert json.loads(zf.read("metadata.json")) == doc
        assert (
            zf.read("later/2025_05_07_0003.abf")
            == (DATA / "2025_05_07_0003.abf").read_bytes()
        )


def test_build_is_reproducible(recording, tmp_path):
    """The same files and metadata give the same bytes, whatever the files'
    timestamps."""
    doc = abfzip.metadata_document(METADATA, catalog_in(recording))
    first = abfzip.build(recording, doc, tmp_path / "a.zip").read_bytes()
    os.utime(recording / "notes.txt", (0, 0))
    assert abfzip.build(recording, doc, tmp_path / "b.zip").read_bytes() == first


def test_build_file_changed_since_cataloged(recording, tmp_path):
    """A file whose hash isn't its record's has changed since the catalog was
    written; nothing is built."""
    doc = abfzip.metadata_document(METADATA, catalog_in(recording))
    (recording / "notes.txt").write_text("cell 6\n")
    with pytest.raises(ValueError, match=r"notes\.txt has changed"):
        abfzip.build(recording, doc, tmp_path / "rec.zip")
    assert not (tmp_path / "rec.zip").exists()


def test_build_own_metadata(recording, tmp_path):
    (recording / "metadata.json").write_text("{}")
    doc = abfzip.metadata_document(METADATA, catalog_in(recording))
    with pytest.raises(ValueError, match=r"metadata\.json of its own"):
        abfzip.build(recording, doc, tmp_path / "rec.zip")
    assert not (tmp_path / "rec.zip").exists()


def test_build_catalog_path_missing(recording, tmp_path):
    """A catalog path must name a file in the directory, exactly."""
    doc = abfzip.metadata_document(METADATA, CATALOG)
    with pytest.raises(ValueError, match=r"2025_05_07_0003\.abf.*isn't in"):
        abfzip.build(recording, doc, tmp_path / "rec.zip")


def test_build_no_abf(tmp_path):
    src = tmp_path / "rec"
    src.mkdir()
    (src / "img.tif").write_bytes(b"II*\x00")
    with pytest.raises(ValueError, match="no ABF files"):
        abfzip.build(src, abfzip.metadata_document({}, []), tmp_path / "rec.zip")


@pytest.mark.parametrize("version", ["1.0", "1.3"])
def test_check_metadata_minor_versions(version):
    doc = abfzip.metadata_document(METADATA, CATALOG) | {"version": version}
    abfzip.check_metadata(doc, [r["path"] for r in CATALOG])


@pytest.mark.parametrize(
    "change,message",
    [
        ({"version": "2.0"}, "only 1.x"),
        ({"$schema": "https://meliza.org/spec:4/abfzip.json"}, r"\$schema"),
        ({"metadata": {"pen": None}}, "invalid at metadata/pen"),
        ({"catalog": [{"protocol": "ramp"}]}, "invalid at catalog/0"),
        ({"catalog": [CATALOG[0], CATALOG[0]]}, "more than once"),
        ({"catalog": [{"path": "metadata.json"}]}, "invalid at catalog/0/path"),
    ],
)
def test_check_metadata_invalid(change, message):
    doc = abfzip.metadata_document(METADATA, CATALOG) | change
    with pytest.raises(ValueError, match=message):
        abfzip.check_metadata(doc, [r["path"] for r in CATALOG])


def test_script(single):
    """catalog writes <directory>.json beside the directory, with the given
    metadata (typed as nbank deposit types it) and the software that wrote the
    files; build writes <directory>.zip from the reviewed file."""
    abfzip.script(
        [
            "catalog",
            "--experimenter",
            "anc4kj",
            "--animal",
            "ca7cb913-0888-4861-8b3b-a954b16f42ce",
            "-k",
            "cell=5",
            str(single),
        ]
    )
    reviewed = single.parent / "rec.json"
    doc = json.loads(reviewed.read_text())
    abf_sha1 = sha1((DATA / "2017_06_07_0044.abf").read_bytes())
    assert doc == abfzip.metadata_document(
        METADATA
        | {"software": "Clampex 10.7.0.3", "processed_by": abfzip.PROCESSED_BY},
        [
            CATALOG[1] | {"sha1": abf_sha1},
            CATALOG[1] | {"path": "2017_06_07_0045.abf", "sha1": abf_sha1},
            {"path": "notes.txt", "sha1": sha1(b"cell 5\n")},
        ],
    )
    assert abfzip.PROCESSED_BY.startswith("org.meliza.melizalab-tools/abf-zip ")
    # the reviewer adds a note to a file
    doc["catalog"][1]["note"] = "after bath application"
    reviewed.write_text(json.dumps(doc))
    abfzip.script(["build", "--metadata", str(reviewed), str(single)])
    with zipfile.ZipFile(single.parent / "rec.zip") as zf:
        assert json.loads(zf.read("metadata.json")) == doc


def test_script_catalog_name_with_underscores(single):
    """A recording name like 20140603_1_1 stays a string (nbank's -k parser
    used to read it as the number 2014060311)"""
    abfzip.script(["catalog", "-k", "name=20140603_1_1", str(single)])
    doc = json.loads((single.parent / "rec.json").read_text())
    assert doc["metadata"]["name"] == "20140603_1_1"


def test_script_catalog_processed_by_with_k(single):
    """processed_by is abf-zip's to set"""
    with pytest.raises(ValueError, match="processed_by is set by abf-zip"):
        abfzip.script(["catalog", "-k", "processed_by=me", str(single)])
    assert not (single.parent / "rec.json").exists()


def test_script_catalog_exists(recording):
    (recording.parent / "rec.json").write_text("{}")
    with pytest.raises(FileExistsError):
        abfzip.script(["catalog", str(recording)])


def test_script_catalog_software_given(single, caplog):
    """A software value given with -k is kept, with a warning if the files
    disagree"""
    abfzip.script(["catalog", "-k", "software=pCLAMP 11", str(single)])
    doc = json.loads((single.parent / "rec.json").read_text())
    assert doc["metadata"]["software"] == "pCLAMP 11"
    assert "Clampex 10.7.0.3" in caplog.text


def test_script_catalog_mixed_software(recording):
    with pytest.raises(ValueError, match="different software"):
        abfzip.script(["catalog", str(recording)])
    assert not (recording.parent / "rec.json").exists()


def test_script_catalog_core_fields_with_k(recording):
    """experimenter and animal are options of their own, as in nbank."""
    with pytest.raises(SystemExit):
        abfzip.script(["catalog", "-k", "animal=x", str(recording)])
