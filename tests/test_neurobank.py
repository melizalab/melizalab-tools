# -*- mode: python -*-
"""Tests for dlab.neurobank

nbank has its own test suite, so these tests exercise how dlab.neurobank uses
it: the order in which locations are searched, how the local cache is used, and
how failures are reported. The registry is faked with an httpx.MockTransport, so
the real nbank request/response handling is still in the loop.
"""

import json
import logging
from pathlib import Path

import httpx
import pytest

from dlab import cache, neurobank

REGISTRY = "http://registry.test/neurobank/"
REMOTE_ROOT = "archive.test/neurobank/resources"


class FakeRegistry:
    """Stands in for a neurobank registry and its http archive.

    resources maps a name to a record as returned by the bulk locations endpoint.
    Files served from the archive are in `files`, keyed by resource name.
    """

    def __init__(self):
        self.resources: dict[str, dict] = {}
        self.files: dict[str, bytes] = {}
        self.mirror_files: dict[str, bytes] = {}  # served from mirror.test
        self.requests: list[httpx.Request] = []
        self.registry_down = False

    def add_http(self, name, content=b"data", *, filename=None, served=True):
        record = {"name": name, "locations": [self.http_location(name)]}
        if filename is not None:
            record["filename"] = filename
        self.resources[name] = record
        if served:
            self.files[name] = content
        return record

    @staticmethod
    def http_location(name, scheme="http"):
        return {"scheme": scheme, "root": REMOTE_ROOT, "resource_name": name}

    @property
    def registry_posts(self):
        return [r for r in self.requests if r.method == "POST"]

    @property
    def downloads(self):
        return [r for r in self.requests if r.method == "GET"]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.registry_down:
            raise httpx.ConnectError("registry unreachable", request=request)
        if request.method == "POST":
            names = json.loads(request.content)["names"]
            lines = [
                json.dumps(self.resources[n]) for n in names if n in self.resources
            ]
            return httpx.Response(200, text="\n".join(lines))
        # GET on the http archive: http://archive.test/neurobank/resources/<name>/
        name = request.url.path.rstrip("/").rsplit("/", 1)[-1]
        files = self.mirror_files if request.url.host == "mirror.test" else self.files
        if name in files:
            return httpx.Response(200, content=files[name])
        return httpx.Response(404)


@pytest.fixture
def registry(monkeypatch):
    fake = FakeRegistry()
    real_client = httpx.Client
    monkeypatch.setattr(
        neurobank,
        "Client",
        lambda *a, **kw: real_client(*a, transport=httpx.MockTransport(fake), **kw),
    )
    # describe_many builds its own client inside nbank.core
    import nbank.core

    monkeypatch.setattr(
        nbank.core.httpx,
        "Client",
        lambda *a, **kw: real_client(*a, transport=httpx.MockTransport(fake), **kw),
    )
    return fake


@pytest.fixture(autouse=True)
def cache_dir(tmp_path, monkeypatch):
    """Keep the tests away from the real user cache"""
    path = tmp_path / "cache"
    monkeypatch.setattr(cache, "user_dir", str(path))
    return path


@pytest.fixture
def remote_cache(cache_dir):
    """Where resources fetched from REMOTE_ROOT's host are cached"""
    return cache_dir / "archive.test" / "resources"


def find(*ids, **kwargs):
    return dict(neurobank.find_resources(*ids, registry_url=REGISTRY, **kwargs))


def make_archive(base: Path, *names: str, suffix: str = "") -> Path:
    """Creates files directly under base, as alt_base expects"""
    base.mkdir(parents=True, exist_ok=True)
    for name in names:
        (base / f"{name}{suffix}").write_text(name)
    return base


def make_local_archive(tmp_path: Path, *names: str) -> Path:
    """Creates a neurobank archive layout: <root>/resources/<id[:2]>/<id>"""
    root = tmp_path / "archive"
    for name in names:
        d = root / "resources" / name[:2]
        d.mkdir(parents=True, exist_ok=True)
        (d / name).write_text(name)
    return root


# find_resources: alt_base


def test_alt_base_hit_does_not_contact_registry(tmp_path, registry):
    base = make_archive(tmp_path / "base", "abcd1234")
    result = find("abcd1234", alt_base=base)
    assert result == {"abcd1234": base / "abcd1234"}
    assert registry.requests == []


def test_alt_base_resolves_missing_extension(tmp_path, registry):
    base = make_archive(tmp_path / "base", "abcd1234", suffix=".wav")
    result = find("abcd1234", alt_base=str(base))  # str is accepted
    assert result == {"abcd1234": base / "abcd1234.wav"}
    assert registry.requests == []


def test_alt_base_found_items_are_yielded_before_registry_is_queried(
    tmp_path, registry
):
    base = make_archive(tmp_path / "base", "abcd1234")
    registry.add_http("efgh5678")
    gen = neurobank.find_resources(
        "abcd1234", "efgh5678", registry_url=REGISTRY, alt_base=base
    )
    assert next(gen) == ("abcd1234", base / "abcd1234")
    assert registry.requests == []
    name, _ = next(gen)
    assert name == "efgh5678"
    assert len(registry.registry_posts) == 1


def test_only_ids_missing_from_alt_base_are_sent_to_registry(tmp_path, registry):
    base = make_archive(tmp_path / "base", "abcd1234")
    registry.add_http("efgh5678")
    find("abcd1234", "efgh5678", alt_base=base)
    (post,) = registry.registry_posts
    assert json.loads(post.content)["names"] == ["efgh5678"]


# find_resources: locations


def test_local_archive_location(tmp_path, registry):
    root = make_local_archive(tmp_path, "abcd1234")
    registry.resources["abcd1234"] = {
        "name": "abcd1234",
        "locations": [
            {"scheme": "neurobank", "root": str(root), "resource_name": "abcd1234"}
        ],
    }
    result = find("abcd1234")
    assert result["abcd1234"] == root / "resources" / "ab" / "abcd1234"
    assert registry.downloads == []


def test_local_archive_location_remapped_by_alt_base(tmp_path, registry):
    # the registry knows the archive at a path that doesn't exist here
    copy = make_local_archive(tmp_path / "mount", "abcd1234")
    registry.resources["abcd1234"] = {
        "name": "abcd1234",
        "locations": [
            {
                "scheme": "neurobank",
                "root": f"/nonexistent/{copy.name}",
                "resource_name": "abcd1234",
            }
        ],
    }
    result = find("abcd1234", alt_base=copy.parent)
    assert result["abcd1234"] == copy / "resources" / "ab" / "abcd1234"


def test_local_archive_preferred_over_download(tmp_path, registry):
    root = make_local_archive(tmp_path, "abcd1234")
    registry.add_http("abcd1234")
    registry.resources["abcd1234"]["locations"].insert(
        0, {"scheme": "neurobank", "root": str(root), "resource_name": "abcd1234"}
    )
    result = find("abcd1234")
    assert result["abcd1234"] == root / "resources" / "ab" / "abcd1234"
    assert registry.downloads == []


def test_unavailable_local_location_falls_through_to_remote(tmp_path, registry):
    registry.add_http("abcd1234", b"remote")
    registry.resources["abcd1234"]["locations"].insert(
        0,
        {
            "scheme": "neurobank",
            "root": str(tmp_path / "not-mounted"),
            "resource_name": "abcd1234",
        },
    )
    result = find("abcd1234")
    assert result["abcd1234"].read_bytes() == b"remote"


def test_unknown_scheme_is_skipped(registry):
    registry.add_http("abcd1234", b"remote")
    registry.resources["abcd1234"]["locations"].insert(
        0, {"scheme": "carrier-pigeon", "root": "x", "resource_name": "abcd1234"}
    )
    assert find("abcd1234")["abcd1234"].read_bytes() == b"remote"


def test_unregistered_id_is_reported_not_raised(registry):
    result = find("nonexistent")
    assert isinstance(result["nonexistent"], FileNotFoundError)
    assert "nonexistent" in str(result["nonexistent"])


def test_resource_with_no_locations(registry):
    registry.resources["abcd1234"] = {"name": "abcd1234", "locations": []}
    assert isinstance(find("abcd1234")["abcd1234"], FileNotFoundError)


def test_mixed_results_keyed_by_requested_id(tmp_path, registry):
    base = make_archive(tmp_path / "base", "local001")
    registry.add_http("remote01", b"r")
    result = find("local001", "remote01", "missing1", alt_base=base)
    assert set(result) == {"local001", "remote01", "missing1"}
    assert result["local001"] == base / "local001"
    assert result["remote01"].read_bytes() == b"r"
    assert isinstance(result["missing1"], FileNotFoundError)


def test_each_id_is_reported_once(tmp_path, registry):
    base = make_archive(tmp_path / "base", "local001")
    registry.add_http("remote01")
    results = list(
        neurobank.find_resources(
            "local001", "remote01", "missing1", registry_url=REGISTRY, alt_base=base
        )
    )
    assert sorted(name for name, _ in results) == ["local001", "missing1", "remote01"]


# find_resources: downloading and the cache


def test_download_is_stored_in_cache(registry, remote_cache):
    registry.add_http("abcd1234", b"payload")
    path = find("abcd1234")["abcd1234"]
    assert path == remote_cache / "abcd1234"
    assert path.read_bytes() == b"payload"


def test_cache_dir_is_keyed_by_host_of_the_resource_url(registry, cache_dir):
    registry.add_http("abcd1234")
    find("abcd1234")
    assert (cache_dir / "archive.test" / "resources" / "abcd1234").exists()
    assert not (cache_dir / "registry.test").exists()


def test_cached_resource_is_not_downloaded_again(registry):
    registry.add_http("abcd1234", b"v1")
    first = find("abcd1234")["abcd1234"]
    assert len(registry.downloads) == 1
    registry.files["abcd1234"] = b"v2"
    second = find("abcd1234")["abcd1234"]
    assert second == first
    assert second.read_bytes() == b"v1"
    assert len(registry.downloads) == 1


def test_filename_field_determines_cached_name(registry, remote_cache):
    registry.add_http("abcd1234", b"x", filename="abcd1234.wav")
    path = find("abcd1234")["abcd1234"]
    assert path == remote_cache / "abcd1234.wav"
    assert path.exists()
    # and the cached copy is found again under that name
    find("abcd1234")
    assert len(registry.downloads) == 1


def test_no_download_returns_cached_file(registry, remote_cache):
    registry.add_http("abcd1234", b"cached")
    find("abcd1234")
    registry.requests.clear()
    path = find("abcd1234", no_download=True)["abcd1234"]
    assert path.read_bytes() == b"cached"
    assert registry.downloads == []


def test_no_download_does_not_fetch_uncached_resource(registry, remote_cache):
    registry.add_http("abcd1234")
    result = find("abcd1234", no_download=True)
    assert isinstance(result["abcd1234"], FileNotFoundError)
    assert registry.downloads == []
    assert not (remote_cache / "abcd1234").exists()


def test_no_download_still_uses_local_archive(tmp_path, registry):
    root = make_local_archive(tmp_path, "abcd1234")
    registry.add_http("abcd1234")
    registry.resources["abcd1234"]["locations"].insert(
        0, {"scheme": "neurobank", "root": str(root), "resource_name": "abcd1234"}
    )
    path = find("abcd1234", no_download=True)["abcd1234"]
    assert path.parent == root / "resources" / "ab"


def test_http_error_is_reported_and_logged(registry, caplog):
    registry.add_http("abcd1234", served=False)
    with caplog.at_level(logging.WARNING, logger="dlab.neurobank"):
        result = find("abcd1234")
    assert isinstance(result["abcd1234"], FileNotFoundError)
    assert "abcd1234" in caplog.text
    assert "404" in caplog.text


def test_failed_download_leaves_no_cache_entry(registry, remote_cache):
    registry.add_http("abcd1234", served=False)
    find("abcd1234")
    # a leftover file would be mistaken for a cache hit on the next call
    assert not (remote_cache / "abcd1234").exists()


def test_failed_location_falls_through_to_next(registry):
    registry.add_http("abcd1234", served=False)
    registry.resources["abcd1234"]["locations"].append(
        {
            "scheme": "https",
            "root": "mirror.test/resources",
            "resource_name": "abcd1234",
        }
    )
    registry.mirror_files["abcd1234"] = b"from mirror"
    path = find("abcd1234")["abcd1234"]
    assert path.read_bytes() == b"from mirror"
    assert path.parent.parent.name == "mirror.test"


def test_registry_failure_propagates(registry):
    registry.registry_down = True
    with pytest.raises(httpx.ConnectError):
        find("abcd1234")


# find_resource


def test_find_resource_returns_path(tmp_path, registry):
    base = make_archive(tmp_path / "base", "abcd1234")
    assert (
        neurobank.find_resource("abcd1234", registry_url=REGISTRY, alt_base=base)
        == base / "abcd1234"
    )


def test_find_resource_raises_when_missing(registry):
    with pytest.raises(FileNotFoundError):
        neurobank.find_resource("nonexistent", registry_url=REGISTRY)


def test_find_resource_passes_no_download(registry):
    registry.add_http("abcd1234")
    with pytest.raises(FileNotFoundError):
        neurobank.find_resource("abcd1234", registry_url=REGISTRY, no_download=True)
    assert registry.downloads == []


# fetch_resource


def test_fetch_resource_without_locations_raises():
    with httpx.Client() as client, pytest.raises(FileNotFoundError):
        neurobank.fetch_resource(client, {"name": "abcd1234", "locations": []})


# describe_resources


def describe(*ids, **kwargs):
    return dict(neurobank.describe_resources(*ids, registry_url=REGISTRY, **kwargs))


def test_describe_from_alt_base(tmp_path, registry):
    base = tmp_path / "base"
    base.mkdir()
    (base / "abcd1234.json").write_text(json.dumps({"name": "abcd1234", "k": 1}))
    assert describe("abcd1234", alt_base=base) == {
        "abcd1234": {"name": "abcd1234", "k": 1}
    }
    assert registry.requests == []


def test_describe_from_alt_base_str(tmp_path, registry):
    (tmp_path / "abcd1234.json").write_text('{"name": "abcd1234"}')
    assert "abcd1234" in describe("abcd1234", alt_base=str(tmp_path))


def test_describe_uses_registry_for_ids_not_in_alt_base(tmp_path, registry):
    (tmp_path / "local001.json").write_text('{"name": "local001"}')
    registry.resources["remote01"] = {"name": "remote01", "type": "x"}
    result = describe("local001", "remote01", alt_base=tmp_path)
    assert result["remote01"] == {"name": "remote01", "type": "x"}
    assert result["local001"] == {"name": "local001"}
    (post,) = registry.registry_posts
    assert json.loads(post.content)["names"] == ["remote01"]


def test_describe_registry_unreachable_reports_each_id(registry):
    registry.registry_down = True
    result = describe("abcd1234", "efgh5678")
    assert set(result) == {"abcd1234", "efgh5678"}
    assert all(isinstance(v, FileNotFoundError) for v in result.values())


def test_describe_partial_registry_failure_does_not_repeat_ids(registry, monkeypatch):
    def flaky(registry_url, *ids):
        yield {"name": ids[0]}
        raise httpx.ReadError("connection dropped")

    monkeypatch.setattr(neurobank, "describe_many", flaky)
    results = list(neurobank.describe_resources("a", "b", registry_url=REGISTRY))
    names = [name for name, _ in results]
    assert sorted(names) == ["a", "b"]
    assert not isinstance(dict(results)["a"], FileNotFoundError)
    assert isinstance(dict(results)["b"], FileNotFoundError)


def test_describe_registry_unreachable_with_alt_base_hit(tmp_path, registry):
    (tmp_path / "local001.json").write_text('{"name": "local001"}')
    registry.registry_down = True
    result = describe("local001", "remote01", alt_base=tmp_path)
    assert result["local001"] == {"name": "local001"}
    assert isinstance(result["remote01"], FileNotFoundError)


def test_describe_unregistered_id_is_reported(registry):
    result = describe("nonexistent")
    assert isinstance(result.get("nonexistent"), FileNotFoundError)


# script


def test_add_registry_argument_default_and_override():
    import argparse

    p = argparse.ArgumentParser()
    neurobank.add_registry_argument(p)
    assert p.parse_args([]).registry_url == neurobank.default_registry
    assert p.parse_args(["-r", "http://x/"]).registry_url == "http://x/"


def test_add_registry_argument_custom_dest():
    import argparse

    p = argparse.ArgumentParser()
    neurobank.add_registry_argument(p, dest="reg")
    assert p.parse_args(["--registry", "http://x/"]).reg == "http://x/"


def test_main_locates_resources(tmp_path, registry, caplog):
    base = make_archive(tmp_path / "base", "abcd1234")
    with caplog.at_level(logging.INFO):
        neurobank.main(["-r", REGISTRY, "-b", str(base), "abcd1234"])
    assert "abcd1234" in caplog.text


def test_main_clear_cache_removes_downloaded_resources(registry, remote_cache):
    # resources are cached under the archive host, which isn't the registry host
    registry.add_http("abcd1234")
    path = find("abcd1234")["abcd1234"]
    assert path.exists()
    neurobank.main(["-r", REGISTRY, "--clear-cache"])
    assert not path.exists()
    assert not remote_cache.exists()
