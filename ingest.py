#!/usr/bin/env python3
"""
ingest.py — parse the Q&A markdown and store chunks in SQLite FTS5.

Self-contained: parsing logic lives in this file (no dependency on app.corpus).
Uses the same schema as app/db.py so the FastAPI service reads it without changes.

Usage:
    python ingest.py corpus/my-corpus.md
    python ingest.py corpus/my-corpus.md --dry-run --verbose
    python ingest.py corpus/my-corpus.md --db data/corpus.db
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:
    print("✗ pyyaml required: pip install pyyaml", file=sys.stderr)
    sys.exit(1)


# ===========================================================================
# 1. SCHEMA (must match app/db.py exactly)
# ===========================================================================
SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT, filename TEXT NOT NULL,
    version INTEGER NOT NULL, checksum TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1, ingested_at REAL NOT NULL,
    chunk_count INTEGER NOT NULL DEFAULT 0, UNIQUE(filename, version));
CREATE INDEX IF NOT EXISTS idx_documents_active ON documents(filename, is_active);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_id TEXT NOT NULL, question TEXT NOT NULL, variants TEXT,
    answer TEXT NOT NULL, source TEXT, original_text TEXT,
    keywords TEXT, evidence TEXT);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    question, variants, answer, original_text, keywords,
    content='chunks', content_rowid='id', tokenize='porter unicode61');
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, question, variants, answer, original_text, keywords)
    VALUES (new.id, new.question, new.variants, new.answer, new.original_text, new.keywords);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, question, variants, answer, original_text, keywords)
    VALUES ('delete', old.id, old.question, old.variants, old.answer, old.original_text, old.keywords);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, question, variants, answer, original_text, keywords)
    VALUES ('delete', old.id, old.question, old.variants, old.answer, old.original_text, old.keywords);
    INSERT INTO chunks_fts(rowid, question, variants, answer, original_text, keywords)
    VALUES (new.id, new.question, new.variants, new.answer, new.original_text, new.keywords);
END;
"""


# ===========================================================================
# 2. PARSING — file text → list of {id, question, variants, answer, ...}
# ===========================================================================
CHUNK_HEADER = re.compile(r"^##\s+(Q\d+)\s+(?:—|–|-|:)\s*(.+)$", re.MULTILINE)
FIELD_RE     = re.compile(r"^\*\*([A-Za-z ]+):\*\*", re.MULTILINE)


def _split_fields(body: str) -> Dict[str, str]:
    """Find every `**Label:**` marker and slice the text between them."""
    fields: Dict[str, str] = {}
    markers = list(FIELD_RE.finditer(body))
    for i, m in enumerate(markers):
        label = m.group(1).strip()
        start = m.end()
        end = markers[i + 1].start() if i + 1 < len(markers) else len(body)
        fields[label] = body[start:end].strip()
    return fields


def _parse_evidence(block: str) -> Optional[Dict[str, Any]]:
    """Accept either ```yaml …``` fenced blocks or inline `{key: val}` dicts."""
    if not block:
        return None
    m = re.search(r"```ya?ml\s*\n(.*?)```", block, re.DOTALL)
    if m:
        try:
            return yaml.safe_load(m.group(1))
        except yaml.YAMLError as e:
            print(f"  ⚠ evidence YAML error: {e}", file=sys.stderr)
            return None
    m = re.search(r"`(\{.*?\})`", block, re.DOTALL)
    if m:
        try:
            return yaml.safe_load(m.group(1))
        except yaml.YAMLError as e:
            print(f"  ⚠ inline evidence error: {e}", file=sys.stderr)
            return None
    return None


def parse_markdown(path: Path, verbose: bool = False) -> List[Dict[str, Any]]:
    """
    Parse the whole file into a list of dicts (one per `## Qn — …` block).

    Prints a stage-by-stage diagnostic so you can see exactly where it goes
    wrong if the count is 0.
    """
    print(f"  [1/4] reading file: {path}")
    text = path.read_text(encoding="utf-8")
    print(f"        {len(text):,} chars, {text.count(chr(10)) + 1:,} lines")

    headers = list(CHUNK_HEADER.finditer(text))
    print(f"  [2/4] `## Q…` headers found: {len(headers)}")
    if not headers:
        print("        ⚠ no headers matched. Expected lines like `## Q1 — Title`.",
              file=sys.stderr)
        print("        first 5 lines of file:", file=sys.stderr)
        for line in text.splitlines()[:5]:
            print(f"        | {line!r}", file=sys.stderr)
        return []

    print(f"  [3/4] extracting fields per chunk…")
    rows: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for i, m in enumerate(headers):
        chunk_id = m.group(1)
        title = m.group(2).strip()
        body_end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        body = text[m.end():body_end]

        fields = _split_fields(body)
        variants = [v.strip()
                    for v in re.split(r"\s*\|\s*", fields.get("Variants", ""))
                    if v.strip()]
        answer = fields.get("Answer", "").strip()

        if not answer:
            skipped.append(f"{chunk_id} (no **Answer:**)")
            continue

        row = {
            "id":            chunk_id,
            "question":      title,
            "variants":      variants,
            "answer":        answer,
            "source":        fields.get("Source", "").strip(),
            "original_text": fields.get("Original text", "").strip(),
            "keywords":      fields.get("Keywords", "").strip(),
            "evidence":      _parse_evidence(fields.get("Evidence", "")),
        }
        rows.append(row)

        if verbose:
            ev = row["evidence"]
            ev_type = ev.get("type") if isinstance(ev, dict) else None
            print(f"        {chunk_id:>5}  title={title[:45]!r:<48} "
                  f"variants={len(variants):>2}  evidence={ev_type or 'none'}")

    print(f"        parsed: {len(rows)} chunks, skipped: {len(skipped)}")
    if skipped:
        print(f"        skipped: {skipped[:10]}{' …' if len(skipped) > 10 else ''}")

    # Sanity summary — surfaces parsing mistakes immediately
    print(f"  [4/4] field coverage:")
    n = len(rows) or 1
    for label, key in [("question", "question"), ("variants", "variants"),
                       ("answer", "answer"), ("source", "source"),
                       ("original_text", "original_text"),
                       ("keywords", "keywords"), ("evidence", "evidence")]:
        if key in ("variants", "evidence"):
            filled = sum(1 for r in rows if r.get(key))
        else:
            filled = sum(1 for r in rows if r.get(key))
        print(f"        {label:<14} {filled:>4}/{len(rows)} "
              f"({100 * filled / n:.0f}%)")

    if verbose and rows:
        print("\n  sample chunk (first):")
        r = rows[0]
        for k, v in r.items():
            s = json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else str(v)
            if len(s) > 100:
                s = s[:97] + "…"
            print(f"        {k:<14} {s}")
    return rows


# ===========================================================================
# 3. DATABASE write — same schema/behaviour as app/db.py
# ===========================================================================
class DB:
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = self._conn()
        conn.executescript(SCHEMA)
        conn.close()

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.db_path, check_same_thread=False,
                            timeout=10.0, isolation_level=None)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys=ON")
        c.execute("PRAGMA busy_timeout=5000")
        return c

    def latest_version(self, filename: str) -> Optional[int]:
        c = self._conn()
        row = c.execute("SELECT MAX(version) FROM documents WHERE filename=?",
                        (filename,)).fetchone()
        c.close()
        return row[0] if row and row[0] else None

    def is_already_active(self, filename: str, checksum: str) -> bool:
        c = self._conn()
        row = c.execute(
            "SELECT 1 FROM documents WHERE filename=? AND checksum=? AND is_active=1",
            (filename, checksum)).fetchone()
        c.close()
        return row is not None

    def insert(self, filename: str, checksum: str, rows: List[Dict[str, Any]],
               purge_old: bool = False) -> Dict[str, Any]:
        c = self._conn()
        c.execute("BEGIN IMMEDIATE")
        try:
            next_v = (c.execute(
                "SELECT COALESCE(MAX(version),0)+1 FROM documents WHERE filename=?",
                (filename,)).fetchone()[0])
            c.execute("UPDATE documents SET is_active=0 WHERE filename=? AND is_active=1",
                      (filename,))
            if purge_old:
                c.execute("DELETE FROM documents WHERE filename=? AND is_active=0",
                          (filename,))
            doc_id = c.execute(
                "INSERT INTO documents (filename,version,checksum,is_active,"
                "ingested_at,chunk_count) VALUES (?,?,?,1,?,?)",
                (filename, next_v, checksum, time.time(), len(rows))).lastrowid
            c.executemany(
                "INSERT INTO chunks (document_id,chunk_id,question,variants,answer,"
                "source,original_text,keywords,evidence) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                [(doc_id, r["id"], r["question"], json.dumps(r["variants"]),
                  r["answer"], r["source"], r["original_text"], r["keywords"],
                  json.dumps(r["evidence"]) if r["evidence"] else None)
                 for r in rows])
            c.execute("COMMIT")
            return {"document_id": doc_id, "version": next_v, "chunks": len(rows)}
        except Exception:
            c.execute("ROLLBACK")
            raise
        finally:
            c.close()

    def show_db(self) -> None:
        c = self._conn()
        try:
            docs = c.execute(
                "SELECT id, filename, version, is_active, chunk_count "
                "FROM documents ORDER BY filename, version").fetchall()
            chunks = c.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            fts = c.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
        finally:
            c.close()
        print("\n  DB state:")
        print(f"    chunks table : {chunks} rows")
        print(f"    chunks_fts   : {fts} rows")
        print(f"    documents:")
        if not docs:
            print("      (none)")
        for d in docs:
            print(f"      id={d['id']:<3} {d['filename']:<45} "
                  f"v{d['version']} active={d['is_active']} "
                  f"chunks={d['chunk_count']}")


# ===========================================================================
# 4. CLI
# ===========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Parse the Q&A markdown and ingest chunks into SQLite FTS5.")
    ap.add_argument("file", type=Path, help="path to the .md corpus")
    ap.add_argument("--db", default="data/corpus.db", help="SQLite path")
    ap.add_argument("--filename", default=None,
                    help="logical name (default: file name)")
    ap.add_argument("--purge-old", action="store_true",
                    help="delete older versions after ingest")
    ap.add_argument("--dry-run", action="store_true",
                    help="parse + print summary but do not write to DB")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="print every chunk as it is parsed")
    args = ap.parse_args()

    if not args.file.exists():
        sys.exit(f"✗ file not found: {args.file}")

    logical_name = args.filename or args.file.name
    print(f"→ Ingesting {args.file} as {logical_name!r}")

    # --- parse ---
    rows = parse_markdown(args.file, verbose=args.verbose)
    if not rows:
        sys.exit("✗ no chunks parsed — see diagnostic above")

    checksum = hashlib.sha256(args.file.read_bytes()).hexdigest()
    print(f"  checksum: {checksum[:16]}…")

    if args.dry_run:
        print("\n  --dry-run: not writing to DB. Parsed rows are shown above.")
        return

    # --- connect ---
    db = DB(args.db)
    print(f"  DB connected: {args.db}")

    if db.is_already_active(logical_name, checksum):
        print(f"  SKIP: {logical_name} with this checksum is already active")
        db.show_db()
        return

    # --- insert ---
    result = db.insert(logical_name, checksum, rows, purge_old=args.purge_old)
    print(f"\n✓ OK: {result['chunks']} chunks → {logical_name} "
          f"v{result['version']} (doc_id={result['document_id']})")

    db.show_db()


if __name__ == "__main__":
    main()