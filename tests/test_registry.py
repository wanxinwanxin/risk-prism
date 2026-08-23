import io
import tarfile

import pytest

from riskprism import registry


RELEASES = [
    {
        "tag_name": "model-2026-08-22b",
        "name": "Model build 2026-08-22b (PRISM-US-MH-0.9 + SH)",
        "published_at": "2026-08-22T03:24:10Z",
        "prerelease": False,
        "assets": [
            {"name": "riskprism-artifacts.tar.gz",
             "browser_download_url": "https://example.test/mh.tar.gz",
             "size": 100},
            {"name": "riskprism-artifacts-sh.tar.gz",
             "browser_download_url": "https://example.test/sh.tar.gz",
             "size": 90},
        ],
    },
    {
        "tag_name": "v0.9.0",  # package release: not a model build
        "name": "riskprism 0.9.0",
        "published_at": "2026-08-23T00:00:00Z",
        "prerelease": False,
        "assets": [],
    },
    {
        "tag_name": "model-2026-08-20-demo",
        "name": "Provisional demo build (300 names)",
        "published_at": "2026-08-20T15:47:08Z",
        "prerelease": True,
        "assets": [
            {"name": "riskprism-artifacts.tar.gz",
             "browser_download_url": "https://example.test/demo.tar.gz",
             "size": 10},
        ],
    },
    {
        "tag_name": "model-2026-08-21",
        "name": "PRISM-US-MH-0.6: value & quality composites",
        "published_at": "2026-08-21T18:47:25Z",
        "prerelease": False,
        "assets": [
            {"name": "riskprism-artifacts.tar.gz",
             "browser_download_url": "https://example.test/v06.tar.gz",
             "size": 50},
        ],
    },
]


class FakeResponse:
    def __init__(self, payload=None, content=b""):
        self._payload = payload
        self.content = content

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


@pytest.fixture
def fake_api(monkeypatch):
    calls = {"n": 0}

    def fake_get(url, **kwargs):
        calls["n"] += 1
        return FakeResponse(payload=RELEASES)

    monkeypatch.setattr(registry.requests, "get", fake_get)
    registry._cache.update(at=0.0, repo=None, entries=None)
    yield calls
    registry._cache.update(at=0.0, repo=None, entries=None)


def test_list_models_filters_and_sorts(fake_api):
    builds = registry.list_models(refresh=True)
    assert [b["tag"] for b in builds] == [
        "model-2026-08-22b", "model-2026-08-21", "model-2026-08-20-demo"]
    latest = builds[0]
    assert latest["model_version"] == "PRISM-US-MH-0.9"
    assert latest["horizons"] == ["medium", "short"]
    assert builds[1]["model_version"] == "PRISM-US-MH-0.6"
    assert builds[1]["horizons"] == ["medium"]


def test_latest_tag_skips_prereleases(fake_api):
    assert registry.latest_tag() == "model-2026-08-22b"


def test_list_models_is_cached(fake_api):
    registry.list_models(refresh=True)
    registry.list_models()
    registry.list_models()
    assert fake_api["n"] == 1
    registry.list_models(refresh=True)
    assert fake_api["n"] == 2


def test_download_artifacts_unknown_tag(fake_api):
    with pytest.raises(KeyError, match="no model release tagged"):
        registry.download_artifacts("model-1999-01-01")
    with pytest.raises(KeyError, match="no short-horizon"):
        registry.download_artifacts("model-2026-08-21", horizon="short")
    with pytest.raises(ValueError, match="horizon"):
        registry.download_artifacts(horizon="long")


def test_unpack_tarball_flattens_and_refuses_traversal(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in [("nested/dir/meta.json", b"{}"),
                           ("../../evil.txt", b"nope"),
                           (".hidden", b"nope")]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    out = registry.unpack_tarball(buf.getvalue(), tmp_path / "art")
    names = sorted(p.name for p in out.iterdir())
    # nested paths flatten to their basename; traversal collapses inside
    # the destination; dotfiles are skipped
    assert names == ["evil.txt", "meta.json"]
    assert not (tmp_path.parent / "evil.txt").exists()
