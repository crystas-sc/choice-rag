"""db.py — SQLite + FTS5 with document versioning."""
from __future__ import annotations
import json, logging, sqlite3, threading, time
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

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
    keywords TEXT, evidence TEXT, answer_source TEXT DEFAULT 'corpus');
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

BM25_WEIGHTS = (3.0, 2.0, 1.0, 1.0, 2.0)


class Database:
    def __init__(self, db_path):
        self.db_path = str(db_path)
        self._write_lock = threading.Lock()
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            try:
                conn.execute("ALTER TABLE chunks ADD COLUMN answer_source TEXT DEFAULT 'corpus'")
            except Exception:
                pass
        logger.info("DB ready at %s", self.db_path)

    def _new_conn(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False,
                               timeout=10.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    @contextmanager
    def connect(self):
        c = self._new_conn()
        try:
            yield c
        finally:
            c.close()

    def ingest(self, filename, checksum, chunks, purge_old=False):
        with self._write_lock, self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT id, version FROM documents WHERE filename=? AND is_active=1 "
                    "AND checksum=? ORDER BY version DESC LIMIT 1",
                    (filename, checksum)).fetchone()
                if row:
                    conn.execute("COMMIT")
                    return {"document_id": row["id"], "version": row["version"],
                            "chunks": len(chunks), "skipped": True}
                nv = conn.execute("SELECT COALESCE(MAX(version),0)+1 FROM documents "
                                  "WHERE filename=?", (filename,)).fetchone()[0]
                conn.execute("UPDATE documents SET is_active=0 WHERE filename=? AND is_active=1",
                             (filename,))
                if purge_old:
                    conn.execute("DELETE FROM documents WHERE filename=? AND is_active=0",
                                 (filename,))
                cur = conn.execute(
                    "INSERT INTO documents (filename,version,checksum,is_active,ingested_at,chunk_count) "
                    "VALUES (?,?,?,1,?,?)",
                    (filename, nv, checksum, time.time(), len(chunks)))
                doc_id = cur.lastrowid
                conn.executemany(
                    "INSERT INTO chunks (document_id,chunk_id,question,variants,answer,"
                    "source,original_text,keywords,evidence) VALUES (?,?,?,?,?,?,?,?,?)",
                    [(doc_id, c["id"], c["question"], json.dumps(c.get("variants") or []),
                      c["answer"], c.get("source") or "", c.get("original_text") or "",
                      c.get("keywords") or "",
                      json.dumps(c.get("evidence")) if c.get("evidence") else None)
                     for c in chunks])
                conn.execute("COMMIT")
                return {"document_id": doc_id, "version": nv,
                        "chunks": len(chunks), "skipped": False}
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def search(self, fts_q, k=5, min_score=0.0):
        if not fts_q:
            return []
        w = BM25_WEIGHTS
        sql = f"""
            SELECT c.chunk_id, c.question, c.answer, c.source, c.keywords,
                   c.variants, c.original_text, c.evidence, c.answer_source, d.filename AS document,
                   -bm25(chunks_fts,{w[0]},{w[1]},{w[2]},{w[3]},{w[4]}) AS score
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            JOIN documents d ON d.id = c.document_id
            WHERE chunks_fts MATCH ? AND d.is_active=1
            ORDER BY bm25(chunks_fts,{w[0]},{w[1]},{w[2]},{w[3]},{w[4]}) ASC
            LIMIT ?"""
        with self.connect() as conn:
            rows = conn.execute(sql, (fts_q, max(k * 3, 20))).fetchall()
        out = []
        for row in rows:
            s = float(row["score"])
            if s < min_score:
                continue
            ev = json.loads(row["evidence"]) if row["evidence"] else None
            quote_text = ""
            if isinstance(ev, dict) and ev.get("quote"):
                quote_text = str(ev["quote"]).strip()
            elif row["original_text"]:
                quote_text = str(row["original_text"]).strip(' "\'')
            out.append({"rank": len(out) + 1, "score": round(s, 4),
                        "id": row["chunk_id"], "question": row["question"],
                        "answer": row["answer"], "source": row["source"],
                        "keywords": row["keywords"], "document": row["document"],
                        "variants": json.loads(row["variants"]) if row["variants"] else [],
                        "original_text": row["original_text"] or "",
                        "quote": quote_text,
                        "evidence": ev,
                        "answer_source": row["answer_source"] or "corpus"})
            if len(out) >= k:
                break
        return out

    def insert_generated_chunk(self, question: str, answer: str, evidence: dict,
                               keywords: str = "", source: str = "generated_on_the_fly",
                               answer_source: str = "generated_on_the_fly") -> str:
        """Insert a newly synthesized Q&A chunk into the database and FTS index."""
        with self._write_lock, self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                doc = conn.execute(
                    "SELECT id FROM documents WHERE filename='generated_on_the_fly' AND is_active=1"
                ).fetchone()
                if not doc:
                    cur = conn.execute(
                        "INSERT INTO documents (filename, version, checksum, is_active, ingested_at, chunk_count) "
                        "VALUES ('generated_on_the_fly', 1, 'generated', 1, ?, 0)",
                        (time.time(),)
                    )
                    doc_id = cur.lastrowid
                else:
                    doc_id = doc["id"]

                chunk_id = f"GEN_{int(time.time() * 1000)}"
                conn.execute(
                    "INSERT INTO chunks (document_id, chunk_id, question, variants, answer, "
                    "source, original_text, keywords, evidence, answer_source) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        doc_id,
                        chunk_id,
                        question,
                        json.dumps([]),
                        answer,
                        source,
                        "",
                        keywords,
                        json.dumps(evidence) if evidence else None,
                        answer_source,
                    )
                )
                conn.execute(
                    "UPDATE documents SET chunk_count = chunk_count + 1 WHERE id = ?",
                    (doc_id,)
                )
                conn.execute("COMMIT")
                logger.info("Saved generated chunk %s (answer_source=%s) for question: %s",
                            chunk_id, answer_source, question)
                return chunk_id
            except Exception:
                conn.execute("ROLLBACK")
                logger.exception("Failed to insert generated chunk")
                raise

    def stats(self):
        with self.connect() as conn:
            docs = conn.execute(
                "SELECT filename, MAX(version) AS latest_version, COUNT(*) AS versions, "
                "SUM(is_active) AS active_versions FROM documents GROUP BY filename"
            ).fetchall()
            ac = conn.execute("SELECT COUNT(*) FROM chunks c JOIN documents d "
                              "ON d.id=c.document_id WHERE d.is_active=1").fetchone()[0]
            fr = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
        return {"documents": [dict(r) for r in docs],
                "active_chunks": ac, "fts_rows": fr}

    def activate_version(self, filename, version):
        with self._write_lock, self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute(
                    "SELECT id FROM documents WHERE filename=? AND version=?",
                    (filename, version)).fetchone()
                if not row:
                    conn.execute("ROLLBACK")
                    return False
                conn.execute("UPDATE documents SET is_active=0 WHERE filename=?", (filename,))
                conn.execute("UPDATE documents SET is_active=1 WHERE filename=? AND version=?",
                             (filename, version))
                conn.execute("COMMIT")
                return True
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def purge_inactive(self, filename=None):
        with self._write_lock, self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                cur = (conn.execute("DELETE FROM documents WHERE filename=? AND is_active=0",
                                    (filename,))
                       if filename
                       else conn.execute("DELETE FROM documents WHERE is_active=0"))
                conn.execute("COMMIT")
                return cur.rowcount
            except Exception:
                conn.execute("ROLLBACK")
                raise

    def get_chunk(self, chunk_id):
        with self.connect() as conn:
            row = conn.execute(
                "SELECT c.*, d.filename AS document FROM chunks c JOIN documents d ON d.id=c.document_id "
                "WHERE c.chunk_id=? AND d.is_active=1 LIMIT 1", (chunk_id,)).fetchone()
        if not row:
            return None
        return {"id": row["chunk_id"], "question": row["question"],
                "variants": json.loads(row["variants"]) if row["variants"] else [],
                "answer": row["answer"], "source": row["source"],
                "original_text": row["original_text"], "keywords": row["keywords"],
                "document": row["document"],
                "evidence": json.loads(row["evidence"]) if row["evidence"] else None}
