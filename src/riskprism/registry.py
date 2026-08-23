"""Versioned model registry — the catalog of published model builds.

Every weekly or manual build is a GitHub release tagged `model-*`
carrying the artifact tarballs (docs/RELEASING.md). This module turns
that release history into a machine-readable registry: list published
builds, resolve "latest", and download any historical build by tag.
The releases page is the source of truth — there is no second database
to drift out of sync.
"""

import io
import os
import re
import tarfile
import time
from pathlib import Path

import requests

DEFAULT_REPO = "wanxinwanxin/risk-prism"
_API = "https://api.github.com/repos/{repo}/releases?per_page=100"
_MODEL_VERSION_RE = re.compile(r"PRISM-US-[A-Z]+-[0-9][0-9.]*")

ASSETS = {"medium": "riskprism-artifacts.tar.gz",
          "short": "riskprism-artifacts-sh.tar.gz"}

# Unauthenticated GitHub API allows 60 requests/hour per IP; the hosted
# server answers /api/v1/registry from this cache between refreshes.
CACHE_TTL = 900.0
_cache: dict = {"at": 0.0, "repo": None, "entries": None}


def _repo() -> str:
    return os.environ.get("RISKPRISM_REPO", DEFAULT_REPO)


def list_models(refresh: bool = False) -> list[dict]:
    """Published model builds, newest first.

    Each entry: ``tag``, ``title``, ``published_at``, ``model_version``
    (parsed from the release title when present), ``prerelease``,
    ``horizons`` (which artifact tarballs the release carries), and
    ``assets`` mapping asset name to ``{url, size}``.
    """
    repo = _repo()
    now = time.monotonic()
    if (not refresh and _cache["entries"] is not None
            and _cache["repo"] == repo and now - _cache["at"] < CACHE_TTL):
        return _cache["entries"]
    resp = requests.get(_API.format(repo=repo),
                        headers={"Accept": "application/vnd.github+json"},
                        timeout=30)
    resp.raise_for_status()
    entries = []
    for rel in resp.json():
        tag = rel.get("tag_name", "")
        if not tag.startswith("model-"):
            continue
        assets = {a["name"]: {"url": a["browser_download_url"],
                              "size": int(a["size"])}
                  for a in rel.get("assets", [])}
        m = _MODEL_VERSION_RE.search(rel.get("name") or "")
        entries.append({
            "tag": tag,
            "title": rel.get("name"),
            "published_at": rel.get("published_at"),
            "model_version": m.group(0) if m else None,
            "prerelease": bool(rel.get("prerelease")),
            "horizons": [h for h, n in ASSETS.items() if n in assets],
            "assets": assets,
        })
    entries.sort(key=lambda e: e["published_at"] or "", reverse=True)
    _cache.update(at=now, repo=repo, entries=entries)
    return entries


def latest_tag() -> str | None:
    """Newest non-prerelease build that carries medium-horizon artifacts."""
    for e in list_models():
        if not e["prerelease"] and "medium" in e["horizons"]:
            return e["tag"]
    return None


def unpack_tarball(content: bytes, dest: str | Path) -> Path:
    """Unpack an artifact tarball flat into dest, refusing path traversal."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as tar:
        for member in tar.getmembers():
            name = Path(member.name).name  # flatten; refuse traversal
            if not member.isfile() or name.startswith("."):
                continue
            src = tar.extractfile(member)
            if src is not None:
                (dest / name).write_bytes(src.read())
    return dest


def download_artifacts(tag: str = "latest", dest: str | Path = "artifacts",
                       horizon: str = "medium") -> Path:
    """Download and unpack one published build's artifacts by release tag."""
    if horizon not in ASSETS:
        raise ValueError(f"horizon must be one of {sorted(ASSETS)}")
    if tag == "latest":
        resolved = latest_tag()
        if resolved is None:
            raise RuntimeError("no published model releases found")
        tag = resolved
    entry = next((e for e in list_models() if e["tag"] == tag), None)
    if entry is None:
        raise KeyError(f"no model release tagged {tag!r}")
    asset = entry["assets"].get(ASSETS[horizon])
    if asset is None:
        raise KeyError(f"release {tag} has no {horizon}-horizon artifacts")
    resp = requests.get(asset["url"], timeout=300)
    resp.raise_for_status()
    return unpack_tarball(resp.content, dest)
