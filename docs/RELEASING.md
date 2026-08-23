# Releasing

Two release tracks share the GitHub releases page and are told apart by
tag prefix. They never trigger each other's automation.

## Model releases — `model-YYYY-MM-DD[suffix]`

The weekly artifact builds. Automated: `build-model.yml` runs Mondays
06:00 UTC, builds at 8,000 candidate names, publishes
`riskprism-artifacts.tar.gz` (medium horizon) and
`riskprism-artifacts-sh.tar.gz` (short horizon) under a `model-$(date)`
tag, re-renders the site, and redeploys.

Manual model releases (mid-week methodology ships) follow the same
shape. Order matters: **create/upload the release before `git push`** —
the deploy workflow boots the server against the *latest* release, so
the artifacts must be up before the code that expects them. Same-day
re-releases append a suffix (`model-2026-08-22b`) or `--clobber` the
assets.

```bash
gh release create model-$(date +%F) riskprism-artifacts.tar.gz \
  --title "Model build ..." --notes "..."
gh release upload model-$(date +%F) riskprism-artifacts-sh.tar.gz --clobber
git push
```

## Package releases — `vX.Y.Z`

The Python package on PyPI. Versioning convention: the package tracks
the model version line — model `PRISM-US-MH-0.9` ↔ package `0.9.y`
(patch releases for code-only fixes); the package becomes `1.0.0` when
the v1.0 milestone is declared.

To publish:

1. Bump `version` in `pyproject.toml`; commit.
2. Tag and release — the `publish.yml` workflow builds and uploads to
   PyPI via trusted publishing:

```bash
git tag v0.9.0 && git push origin v0.9.0
gh release create v0.9.0 --title "riskprism 0.9.0" \
  --notes "Package release. Model artifacts live under model-* releases."
```

The workflow refuses to publish if the tag doesn't match the pyproject
version.

### One-time PyPI setup (before the first publish)

Trusted publishing needs the publisher registered on PyPI once — no API
token is ever stored in the repo:

1. Log in to pypi.org → *Your projects* → *Publishing* → **Add a new
   pending publisher** (the project name `riskprism` is unclaimed as of
   2026-08-23).
2. Fill in: PyPI project name `riskprism`, owner `wanxinwanxin`,
   repository `risk-prism`, workflow `publish.yml`, environment `pypi`.
3. In the GitHub repo settings, create an environment named `pypi`
   (Settings → Environments → New environment; no secrets needed).
4. Publish with the steps above. After the first successful publish,
   switch the README quickstart from the git install to
   `pip install riskprism`.
