import json
import zipfile

import pandas as pd

from riskprism.data.edgar import EdgarClient


def _client(tmp_path):
    return EdgarClient(user_agent="test test@example.com", cache_dir=tmp_path)


def test_extract_bulk_populates_cache(tmp_path):
    c = _client(tmp_path)
    zpath = tmp_path / "bulk_facts.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("CIK0000320193.json", json.dumps({"entityName": "Apple"}))
        z.writestr("CIK0000789019.json", json.dumps({"entityName": "Microsoft"}))
    n = c._extract_bulk(zpath, [320193, 789019, 999999], "facts")
    assert n == 2  # 999999 not in archive
    assert json.loads((tmp_path / "facts_0000320193.json").read_text())["entityName"] == "Apple"
    # and the normal API path now serves it straight from cache
    assert c.company_facts(320193)["entityName"] == "Apple"


def test_blocked_client_still_serves_cache(tmp_path):
    c = _client(tmp_path)
    (tmp_path / "facts_0000320193.json").write_text(json.dumps({"entityName": "Apple"}))
    c._consecutive_blocks = 99  # WAF-blocked
    assert c.is_blocked
    assert c.company_facts(320193)["entityName"] == "Apple"  # cache still works
    assert c.company_facts(111111) is None  # uncached name degrades to None


def test_bulk_prefetch_noop_when_cache_warm(tmp_path):
    c = _client(tmp_path)
    for cik in range(50):
        (tmp_path / f"facts_{cik:010d}.json").write_text("{}")
        (tmp_path / f"subs_{cik:010d}.json").write_text("{}")
    # under min_missing threshold -> no network attempted (would raise if it were)
    c.session = None
    c.bulk_prefetch(list(range(50)), min_missing=200, verbose=False)
