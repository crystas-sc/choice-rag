from pathlib import Path
import pytest
from app.corpus import build_fts_query, chunk_to_dict, parse_markdown
from app.db import Database

CORPUS = Path("corpus/weekly-market-view-16-10-2026-qa.md")


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "test.db")
    if CORPUS.exists():
        d.ingest("test.md", "sum1", [chunk_to_dict(c) for c in parse_markdown(CORPUS)])
    return d


def test_fts_query():
    assert build_fts_query("US CPI") == '"us" OR "cpi"'
    assert build_fts_query("") == ""


def test_search_returns_results(db):
    if not CORPUS.exists():
        pytest.skip("corpus missing")
    rows = db.search(build_fts_query("US CPI surprise"), k=3)
    assert rows and all("evidence" in r for r in rows)


def test_idempotent(db):
    if not CORPUS.exists():
        pytest.skip("corpus missing")
    chunks = parse_markdown(CORPUS)
    r = db.ingest("test.md", "sum1", [chunk_to_dict(c) for c in chunks])
    assert r["skipped"] is True
