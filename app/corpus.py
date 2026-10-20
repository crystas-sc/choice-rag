"""corpus.py — markdown parser + FTS5 query builder."""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

CHUNK_HEADER = re.compile(r"^##\s+(Q\d+)\s+(?:—|–|-|:)\s*(.+)$", re.MULTILINE)
FIELD_RE = re.compile(r"^\*\*([A-Za-z ]+):\*\*", re.MULTILINE)
FTS_ALNUM_RE = re.compile(r"[A-Za-z0-9]+")


@dataclass
class Chunk:
    id: str
    question: str
    variants: List[str] = field(default_factory=list)
    answer: str = ""
    source: str = ""
    original_text: str = ""
    evidence: Optional[Dict[str, Any]] = None
    keywords: str = ""


def _parse_fields(body):
    fields, ms = {}, list(FIELD_RE.finditer(body))
    for i, m in enumerate(ms):
        start, end = m.end(), (ms[i + 1].start() if i + 1 < len(ms) else len(body))
        fields[m.group(1).strip()] = body[start:end].strip()
    return fields


def _parse_evidence(block):
    if not block:
        return None
    m = re.search(r"```ya?ml\s*\n(.*?)```", block, re.DOTALL)
    if m:
        try:
            return yaml.safe_load(m.group(1))
        except yaml.YAMLError:
            return None
    m = re.search(r"`(\{.*?\})`", block, re.DOTALL)
    if m:
        try:
            return yaml.safe_load(m.group(1))
        except yaml.YAMLError:
            return None
    return None


def parse_markdown(path):
    md = Path(path).read_text(encoding="utf-8")
    headers = list(CHUNK_HEADER.finditer(md))
    chunks = []
    for i, m in enumerate(headers):
        end = headers[i + 1].start() if i + 1 < len(headers) else len(md)
        f = _parse_fields(md[m.end():end])
        variants = [v.strip() for v in re.split(r"\s*\|\s*", f.get("Variants", "")) if v.strip()]
        chunks.append(Chunk(
            id=m.group(1), question=m.group(2).strip(), variants=variants,
            answer=f.get("Answer", "").strip(), source=f.get("Source", "").strip(),
            original_text=f.get("Original text", "").strip(),
            evidence=_parse_evidence(f.get("Evidence", "")),
            keywords=f.get("Keywords", "").strip(),
        ))
    return chunks


def chunk_to_dict(c):
    return {
        "id": c.id, "question": c.question, "variants": c.variants,
        "answer": c.answer, "source": c.source,
        "original_text": c.original_text, "evidence": c.evidence,
        "keywords": c.keywords,
    }


def build_fts_query(query):
    tokens = FTS_ALNUM_RE.findall(query.lower())
    return " OR ".join(f'"{t}"' for t in tokens) if tokens else ""
