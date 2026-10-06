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
        self.auth_required = False  # downloads need an Authorization header

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
        if self.auth_required and "authorization" not in request.headers:
            return httpx.Response(403)
        name = request.url.path.rstrip("/").rsplit("/", 1)[-1]
        files = self.mirror_files if request.url.host == "mirror.test" else self.files
        if name in files:
            return httpx.Response(200, content=files[name])
        return httpx.Response(404)


@pytest.fixture
def registry(monkeypatch):
    """Route all HTTP traffic to a FakeRegistry via httpx.MockTransport.

    Patches the Client used by dlab.neurobank and the one nbank.core uses
    for describe_many, so the real nbank query and fetch code runs but nothing
    touches the network. Returns the fake so tests can add resources and inspect
    the requests that were made.
    """
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
    """find_resources collected into {id: Path | FileNotFoundError}."""
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
    """A resource present in alt_base is resolved locally, with no registry request."""
    base = make_archive(tmp_path / "base", "abcd1234")
    result = find("abcd1234", alt_base=base)
    assert result == {"abcd1234": base / "abcd1234"}
    assert registry.requests == [], "alt_base hit should not contact the registry"


def test_alt_base_resolves_missing_extension(tmp_path, registry):
    """alt_base is searched by id; a file stored as <id>.wav is still found.

    Also checks that alt_base may be a str as well as a Path.
    """
    base = make_archive(tmp_path / "base", "abcd1234", suffix=".wav")
    result = find("abcd1234", alt_base=str(base))  # str is accepted
    assert result == {"abcd1234": base / "abcd1234.wav"}
    assert registry.requests == [], "alt_base hit should not contact the registry"


def test_alt_base_found_items_are_yielded_before_registry_is_queried(
    tmp_path, registry
):
    """find_resources is a generator: ids found in alt_base are yielded first, before
    any request is made to the registry for the remaining ids.
    """
    base = make_archive(tmp_path / "base", "abcd1234")
    registry.add_http("efgh5678")
    gen = neurobank.find_resources(
        "abcd1234", "efgh5678", registry_url=REGISTRY, alt_base=base
    )
    assert next(gen) == ("abcd1234", base / "abcd1234")
    assert registry.requests == [], "registry queried before alt_base hits were yielded"
    name, _ = next(gen)
    assert name == "efgh5678"
    assert len(registry.registry_posts) == 1, (
        "expected one bulk query for the remaining id"
    )


def test_only_ids_missing_from_alt_base_are_sent_to_registry(tmp_path, registry):
    """The bulk registry query contains only the ids alt_base could not satisfy."""
    base = make_archive(tmp_path / "base", "abcd1234")
    registry.add_http("efgh5678")
    find("abcd1234", "efgh5678", alt_base=base)
    (post,) = registry.registry_posts
    assert json.loads(post.content)["names"] == ["efgh5678"], (
        "ids found in alt_base should not be sent to the registry"
    )


# find_resources: locations


def test_local_archive_location(tmp_path, registry):
    """A registry location with scheme 'neurobank' names a local archive. The file is
    resolved to <root>/resources/<id[:2]>/<id> and nothing is downloaded.
    """
    root = make_local_archive(tmp_path, "abcd1234")
    registry.resources["abcd1234"] = {
        "name": "abcd1234",
        "locations": [
            {"scheme": "neurobank", "root": str(root), "resource_name": "abcd1234"}
        ],
    }
    result = find("abcd1234")
    assert result["abcd1234"] == root / "resources" / "ab" / "abcd1234"
    assert registry.downloads == [], "a local archive copy should not be downloaded"


def test_local_archive_location_remapped_by_alt_base(tmp_path, registry):
    """alt_base replaces the directory containing the archive, for archives mounted
    at a different path on this host than the one the registry recorded. The
    recorded root does not exist here; only its final component is used.
    """
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
    assert result["abcd1234"] == copy / "resources" / "ab" / "abcd1234", (
        "alt_base should replace the archive's parent directory"
    )


def test_local_archive_preferred_over_download(tmp_path, registry):
    """When a location list has a local archive ahead of an http location, the local
    file is returned and no download is attempted.
    """
    root = make_local_archive(tmp_path, "abcd1234")
    registry.add_http("abcd1234")
    registry.resources["abcd1234"]["locations"].insert(
        0, {"scheme": "neurobank", "root": str(root), "resource_name": "abcd1234"}
    )
    result = find("abcd1234")
    assert result["abcd1234"] == root / "resources" / "ab" / "abcd1234", (
        "local location should win over http"
    )
    assert registry.downloads == [], "a local archive copy should not be downloaded"


def test_unavailable_local_location_falls_through_to_remote(tmp_path, registry):
    """If the archive named by a local location is not reachable from this host,
    fetch_resource moves on to the next location instead of failing.
    """
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
    assert result["abcd1234"].read_bytes() == b"remote", (
        "unreachable local location should fall through to http"
    )


def test_unknown_scheme_is_skipped(registry):
    """Locations whose scheme nbank does not recognise are ignored, not errors."""
    registry.add_http("abcd1234", b"remote")
    registry.resources["abcd1234"]["locations"].insert(
        0, {"scheme": "carrier-pigeon", "root": "x", "resource_name": "abcd1234"}
    )
    assert find("abcd1234")["abcd1234"].read_bytes() == b"remote", (
        "unknown scheme should be skipped, not fatal"
    )


def test_unregistered_id_is_reported_not_raised(registry):
    """An id the registry does not know is yielded as a FileNotFoundError value (not
    raised), so one bad id does not abort a batch. The error names the id.
    """
    result = find("nonexistent")
    assert isinstance(result["nonexistent"], FileNotFoundError)
    assert "nonexistent" in str(result["nonexistent"]), (
        "error should name the missing id"
    )


def test_resource_with_no_locations(registry):
    """A registered resource with an empty location list is a FileNotFoundError."""
    registry.resources["abcd1234"] = {"name": "abcd1234", "locations": []}
    assert isinstance(find("abcd1234")["abcd1234"], FileNotFoundError)


def test_mixed_results_keyed_by_requested_id(tmp_path, registry):
    """A batch mixing an alt_base hit, a remote download and a missing id returns one
    result per requested id, keyed by the id the caller asked for.
    """
    base = make_archive(tmp_path / "base", "local001")
    registry.add_http("remote01", b"r")
    result = find("local001", "remote01", "missing1", alt_base=base)
    assert set(result) == {"local001", "remote01", "missing1"}
    assert result["local001"] == base / "local001"
    assert result["remote01"].read_bytes() == b"r"
    assert isinstance(result["missing1"], FileNotFoundError)


def test_each_id_is_reported_once(tmp_path, registry):
    """Across the alt_base, registry and never-answered paths, every requested id is
    yielded exactly once: none is lost and none is duplicated.
    """
    base = make_archive(tmp_path / "base", "local001")
    registry.add_http("remote01")
    results = list(
        neurobank.find_resources(
            "local001", "remote01", "missing1", registry_url=REGISTRY, alt_base=base
        )
    )
    assert sorted(name for name, _ in results) == [
        "local001",
        "missing1",
        "remote01",
    ], "each id should be reported exactly once"


# find_resources: downloading and the cache


def test_download_is_stored_in_cache(registry, remote_cache):
    """A remote resource is downloaded to <cache>/<archive host>/resources/<name>,
    and that path is returned.
    """
    registry.add_http("abcd1234", b"payload")
    path = find("abcd1234")["abcd1234"]
    assert path == remote_cache / "abcd1234"
    assert path.read_bytes() == b"payload"


def test_cache_dir_is_keyed_by_host_of_the_resource_url(registry, cache_dir):
    """The cache subdirectory is named for the host of the archive URL, not the
    registry's host. This is why --clear-cache has to clear the whole cache (see
    test_main_clear_cache_removes_downloaded_resources).
    """
    registry.add_http("abcd1234")
    find("abcd1234")
    assert (cache_dir / "archive.test" / "resources" / "abcd1234").exists(), (
        "cache should be keyed by the archive host"
    )
    assert not (cache_dir / "registry.test").exists(), (
        "cache should not be keyed by the registry host"
    )


def test_cached_resource_is_not_downloaded_again(registry):
    """A second lookup is served from the cache. The server changing the file in the
    meantime is not noticed: cached entries are never revalidated.
    """
    registry.add_http("abcd1234", b"v1")
    first = find("abcd1234")["abcd1234"]
    assert len(registry.downloads) == 1, "setup: first lookup should download"
    registry.files["abcd1234"] = b"v2"
    second = find("abcd1234")["abcd1234"]
    assert second == first, "second lookup should return the cached path"
    assert second.read_bytes() == b"v1", "cache is not expected to be revalidated"
    assert len(registry.downloads) == 1, "second lookup should not download"


def test_filename_field_determines_cached_name(registry, remote_cache):
    """If the registry record has a 'filename', the cached file uses it (typically to
    keep the extension), and later lookups find the cache entry under that name.
    """
    registry.add_http("abcd1234", b"x", filename="abcd1234.wav")
    path = find("abcd1234")["abcd1234"]
    assert path == remote_cache / "abcd1234.wav"
    assert path.exists()
    # and the cached copy is found again under that name
    find("abcd1234")
    assert len(registry.downloads) == 1, (
        "cached file under 'filename' should be found again"
    )


def test_no_download_returns_cached_file(registry, remote_cache):
    """no_download only prevents network fetches; files already in the cache are
    still returned.
    """
    registry.add_http("abcd1234", b"cached")
    find("abcd1234")
    registry.requests.clear()
    path = find("abcd1234", no_download=True)["abcd1234"]
    assert path.read_bytes() == b"cached"
    assert registry.downloads == [], "no_download must not fetch"


def test_no_download_does_not_fetch_uncached_resource(registry, remote_cache):
    """With no_download, an uncached remote resource is a FileNotFoundError. No
    request is made to the archive and no cache file is created.
    """
    registry.add_http("abcd1234")
    result = find("abcd1234", no_download=True)
    assert isinstance(result["abcd1234"], FileNotFoundError)
    assert registry.downloads == [], "no_download must not fetch"
    assert not (remote_cache / "abcd1234").exists(), "no cache file should be created"


def test_no_download_still_uses_local_archive(tmp_path, registry):
    """no_download restricts only http fetching; a local archive location still works."""
    root = make_local_archive(tmp_path, "abcd1234")
    registry.add_http("abcd1234")
    registry.resources["abcd1234"]["locations"].insert(
        0, {"scheme": "neurobank", "root": str(root), "resource_name": "abcd1234"}
    )
    path = find("abcd1234", no_download=True)["abcd1234"]
    assert path.parent == root / "resources" / "ab", (
        "local archive should be used despite no_download"
    )


def test_http_error_is_reported_and_logged(registry, caplog):
    """A 404 from the archive becomes a FileNotFoundError result, and a warning
    naming the resource and the http status is logged.
    """
    registry.add_http("abcd1234", served=False)
    with caplog.at_level(logging.WARNING, logger="dlab.neurobank"):
        result = find("abcd1234")
    assert isinstance(result["abcd1234"], FileNotFoundError)
    assert "abcd1234" in caplog.text, "warning should name the resource"
    assert "404" in caplog.text, "warning should include the http status"


def test_failed_download_leaves_no_cache_entry(registry, remote_cache):
    """A failed download must not leave a file at the cache path, because the
    existence of that file is what counts as a cache hit on the next lookup.
    """
    registry.add_http("abcd1234", served=False)
    find("abcd1234")
    # a leftover file would be mistaken for a cache hit on the next call
    assert not (remote_cache / "abcd1234").exists(), (
        "a leftover file would count as a cache hit"
    )


def test_failed_location_falls_through_to_next(registry):
    """If the first location fails (404), the next one (a mirror on another host) is
    tried, and the result is cached under the mirror's host.
    """
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
    assert path.parent.parent.name == "mirror.test", (
        "mirror download should be cached under the mirror's host"
    )


def test_registry_failure_propagates(registry):
    """find_resources does not swallow connection errors from the registry itself.
    Contrast with describe_resources, which reports them per id.
    """
    registry.registry_down = True
    with pytest.raises(httpx.ConnectError):
        find("abcd1234")


# find_resource


def test_find_resource_returns_path(tmp_path, registry):
    """find_resource returns the Path for a single id."""
    base = make_archive(tmp_path / "base", "abcd1234")
    assert (
        neurobank.find_resource("abcd1234", registry_url=REGISTRY, alt_base=base)
        == base / "abcd1234"
    )


def test_find_resource_raises_when_missing(registry):
    """find_resource raises the FileNotFoundError that find_resources would have
    yielded as a value.
    """
    with pytest.raises(FileNotFoundError):
        neurobank.find_resource("nonexistent", registry_url=REGISTRY)


def test_find_resource_passes_no_download(registry):
    """no_download is forwarded by the find_resource convenience wrapper."""
    registry.add_http("abcd1234")
    with pytest.raises(FileNotFoundError):
        neurobank.find_resource("abcd1234", registry_url=REGISTRY, no_download=True)
    assert registry.downloads == [], "no_download should be forwarded"


# fetch_resource


def test_fetch_resource_without_locations_raises():
    """fetch_resource raises FileNotFoundError when given an empty location list."""
    with httpx.Client() as client, pytest.raises(FileNotFoundError):
        neurobank.fetch_resource(client, {"name": "abcd1234", "locations": []})


# describe_resources


def describe(*ids, **kwargs):
    """describe_resources collected into {id: dict | FileNotFoundError}."""
    return dict(neurobank.describe_resources(*ids, registry_url=REGISTRY, **kwargs))


def test_describe_from_alt_base(tmp_path, registry):
    """describe_resources reads <alt_base>/<id>.json and returns its contents without
    contacting the registry.
    """
    base = tmp_path / "base"
    base.mkdir()
    (base / "abcd1234.json").write_text(json.dumps({"name": "abcd1234", "k": 1}))
    assert describe("abcd1234", alt_base=base) == {
        "abcd1234": {"name": "abcd1234", "k": 1}
    }
    assert registry.requests == [], "alt_base hit should not contact the registry"


def test_describe_from_alt_base_str(tmp_path, registry):
    """alt_base may be given as a str."""
    (tmp_path / "abcd1234.json").write_text('{"name": "abcd1234"}')
    assert "abcd1234" in describe("abcd1234", alt_base=str(tmp_path))


def test_describe_uses_registry_for_ids_not_in_alt_base(tmp_path, registry):
    """Ids with no local json are described by the registry, and only those ids are
    sent to it; local and remote results come back together.
    """
    (tmp_path / "local001.json").write_text('{"name": "local001"}')
    registry.resources["remote01"] = {"name": "remote01", "type": "x"}
    result = describe("local001", "remote01", alt_base=tmp_path)
    assert result["remote01"] == {"name": "remote01", "type": "x"}
    assert result["local001"] == {"name": "local001"}
    (post,) = registry.registry_posts
    assert json.loads(post.content)["names"] == ["remote01"], (
        "only ids missing locally should be sent to the registry"
    )


def test_describe_registry_unreachable_reports_each_id(registry):
    """If the registry cannot be reached, each id is reported as a FileNotFoundError
    rather than the exception propagating.
    """
    registry.registry_down = True
    result = describe("abcd1234", "efgh5678")
    assert set(result) == {"abcd1234", "efgh5678"}
    assert all(isinstance(v, FileNotFoundError) for v in result.values()), (
        "every id should be an error when the registry is down"
    )


def test_describe_partial_registry_failure_does_not_repeat_ids(registry, monkeypatch):
    """If the connection drops partway through the response, ids already delivered
    are not repeated as errors; only the undelivered ones are.

    The ids are unpacked from a set, so which one arrives first is arbitrary and
    the test works out which from what the fake registry was asked for.
    """
    delivered = []

    def flaky(registry_url, *ids):
        # ids come from a set, so which one arrives first is arbitrary
        delivered.append(ids[0])
        yield {"name": ids[0]}
        raise httpx.ReadError("connection dropped")

    monkeypatch.setattr(neurobank, "describe_many", flaky)
    results = list(neurobank.describe_resources("a", "b", registry_url=REGISTRY))
    assert sorted(name for name, _ in results) == ["a", "b"], (
        "each id should be reported exactly once"
    )
    by_name = dict(results)
    (got,) = delivered
    (missed,) = {"a", "b"} - {got}
    assert by_name[got] == {"name": got}, "delivered id should keep its record"
    assert isinstance(by_name[missed], FileNotFoundError), (
        "undelivered id should be an error"
    )


def test_describe_registry_unreachable_with_alt_base_hit(tmp_path, registry):
    """When the registry is down, ids found in alt_base still succeed; only the rest
    are reported as FileNotFoundError.
    """
    (tmp_path / "local001.json").write_text('{"name": "local001"}')
    registry.registry_down = True
    result = describe("local001", "remote01", alt_base=tmp_path)
    assert result["local001"] == {"name": "local001"}
    assert isinstance(result["remote01"], FileNotFoundError)


def test_describe_unregistered_id_is_reported(registry):
    """An id the registry does not know is reported as a FileNotFoundError, as the
    docstring promises. (Previously it was silently omitted.)
    """
    result = describe("nonexistent")
    assert isinstance(result.get("nonexistent"), FileNotFoundError), (
        "unregistered id should be reported, not omitted"
    )


# script


def test_add_registry_argument_default_and_override():
    """The -r/--registry option defaults to the module's default registry and can be
    overridden.
    """
    import argparse

    p = argparse.ArgumentParser()
    neurobank.add_registry_argument(p)
    assert p.parse_args([]).registry_url == neurobank.default_registry
    assert p.parse_args(["-r", "http://x/"]).registry_url == "http://x/"


def test_add_registry_argument_custom_dest():
    """The destination attribute name is configurable."""
    import argparse

    p = argparse.ArgumentParser()
    neurobank.add_registry_argument(p, dest="reg")
    assert p.parse_args(["--registry", "http://x/"]).reg == "http://x/"


def test_main_locates_resources(tmp_path, registry, caplog):
    """Smoke test for the script: main() resolves an id from --base and logs it."""
    base = make_archive(tmp_path / "base", "abcd1234")
    with caplog.at_level(logging.INFO):
        neurobank.main(["-r", REGISTRY, "-b", str(base), "abcd1234"])
    assert "abcd1234" in caplog.text, "main should log the located resource"


def test_main_clear_cache_removes_downloaded_resources(registry, remote_cache):
    """--clear-cache removes resources downloaded from an archive host, which differs
    from the registry host, so the whole cache must be cleared.
    """
    # resources are cached under the archive host, which isn't the registry host
    registry.add_http("abcd1234")
    path = find("abcd1234")["abcd1234"]
    assert path.exists(), "setup: download should have been cached"
    neurobank.main(["-r", REGISTRY, "--clear-cache"])
    assert not path.exists(), "--clear-cache should remove downloads from archive hosts"
    assert not remote_cache.exists()


def test_download_uses_default_auth(registry, remote_cache, monkeypatch):
    """Downloads use the credentials in default_auth (from ~/.netrc), so
    archives that need a login can be fetched."""
    registry.auth_required = True
    registry.add_http("abcd1234", b"secret")
    monkeypatch.setattr(neurobank, "default_auth", httpx.BasicAuth("user", "pass"))
    assert find("abcd1234")["abcd1234"].read_bytes() == b"secret"
    assert registry.downloads[0].headers["authorization"].startswith("Basic ")


def test_download_without_credentials_fails(registry, remote_cache, monkeypatch):
    """Without credentials, an archive that needs a login refuses the download,
    which is reported as not found."""
    registry.auth_required = True
    registry.add_http("abcd1234", b"secret")
    monkeypatch.setattr(neurobank, "default_auth", None)
    assert isinstance(find("abcd1234")["abcd1234"], FileNotFoundError)
