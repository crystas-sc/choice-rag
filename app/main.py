"""main.py — FastAPI BM25 service over SQLite FTS5."""
from __future__ import annotations
import hashlib, json, logging, os, re, tempfile, time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional



from fastapi import FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.corpus import build_fts_query, chunk_to_dict, parse_markdown
from app.db import Database
from app.models import (ActivateRequest, ChunkOut, DecisionOut, DecisionRequest, DecisionResponse,
                        GenerateRequest, GenerateResponse, HealthOut, IngestResult, SearchRequest,
                        SearchResponse, SearchResult)

# ADD these next to your existing imports
import httpx
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s — %(message)s")
log = logging.getLogger("choicerag")

DB_PATH = os.getenv("DB_PATH", "data/corpus.db")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "")
CORS_ORIGINS = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()]
PDF_DIR = Path(os.getenv("PDF_DIR", "pdfs")).resolve()
DEFAULT_PDF = os.getenv("DEFAULT_PDF", "")
IS_DOCKER = Path("/.dockerenv").exists()
DEFAULT_OLLAMA_HOST = "host.docker.internal" if IS_DOCKER else "localhost"
OLLAMA_URL = os.getenv("OLLAMA_URL", f"http://{DEFAULT_OLLAMA_HOST}:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "tev1:0.8b")
OLLAMA_GEN_MODEL = os.getenv("OLLAMA_GEN_MODEL", "qwen2.5:0.5b")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "120.0"))

DECISION_LABEL_NONE = "No appropriate answer found, Ask an agent"

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["db"] = Database(DB_PATH)
    log.info("ChoiceRAG ready — docs at /docs")
    yield


app = FastAPI(title="ChoiceRAG — BM25 on SQLite FTS5", version="1.0.0",
              lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS,
                   allow_methods=["*"], allow_headers=["*"])


def db() -> Database:
    d = state.get("db")
    if d is None:
        raise HTTPException(503, "DB not initialised")
    return d


def _require_admin(token: Optional[str]) -> None:
    if ADMIN_TOKEN and token != ADMIN_TOKEN:
        raise HTTPException(401, "invalid admin token")


STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index():
    f = STATIC_DIR / "index.html"
    if f.exists():
        return FileResponse(f)
    return {
        "service": "ChoiceRAG",
        "docs": "/docs",
        "note": f"UI not found — place index.html at {f}",
    }


@app.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    return HealthOut(status="ok", **db().stats())


def _normalize_decision(text: str, num_candidates: int) -> tuple[str, str]:
    """Parse output from the 0.8B decision model into a standard state and user-facing label."""
    t = text.strip().lower()
    if "both" in t or ("top 1" in t and "top 2" in t) or ("top1" in t and "top2" in t):
        if num_candidates >= 2:
            return "both top1 and top2", "both top1 and top2"
        return "Just top 1", "Just top 1"

    if "just top 2" in t or "top 2" in t or "top2" in t or "candidate 2" in t:
        if num_candidates >= 2:
            return "Just top 2", "Just top 2"
        return "none", DECISION_LABEL_NONE

    if "just top 1" in t or "top 1" in t or "top1" in t or "candidate 1" in t:
        if num_candidates >= 1:
            return "Just top 1", "Just top 1"
        return "none", DECISION_LABEL_NONE

    if "none" in t or "neither" in t or "not found" in t or "agent" in t or t == "no":
        return "none", DECISION_LABEL_NONE

    # Default fallback
    if num_candidates >= 1:
        return "Just top 1", "Just top 1"
    return "none", DECISION_LABEL_NONE


NOUL_THRESHOLD = float(os.getenv("NOUL_THRESHOLD", "0.5"))


def _parse_noul_val(v: Any, threshold: float = NOUL_THRESHOLD) -> tuple[bool, Optional[float]]:
    """Extract (is_true, score) from a SystemOne 'noul' response.

    e.g. {"type": "noul", "noul": 0.76449} -> (True, 0.76449)
    """
    if isinstance(v, bool):
        return v, 1.0 if v else 0.0
    if isinstance(v, dict):
        if "noul" in v:
            return _parse_noul_val(v["noul"], threshold)
        for k in ("score", "probability", "value", "choice", "answer", "result", "bool"):
            if k in v:
                return _parse_noul_val(v[k], threshold)
        for item in v.values():
            if isinstance(item, (int, float, bool)):
                return _parse_noul_val(item, threshold)
    if isinstance(v, (int, float)):
        score = float(v)
        return (score >= threshold), score
    if isinstance(v, str):
        v_low = v.strip().lower()
        if v_low in ("true", "yes", "t", "y"):
            return True, 1.0
        if v_low in ("false", "no", "f", "n"):
            return False, 0.0
        try:
            score = float(v_low)
            return (score >= threshold), score
        except ValueError:
            pass
    return False, None


def _parse_systemone_response(resp_data: Any, raw_text: str, num_candidates: int) -> tuple[str, str, str]:
    """Parse the /v1/systemone response into state, label, and raw value."""
    # 1. Handle top1_adds_value / top2_adds_value boolean / noul probability questions format
    if isinstance(resp_data, dict):
        answers = resp_data.get("answers") if isinstance(resp_data.get("answers"), dict) else resp_data
        has_t1 = "top1_adds_value" in answers
        has_t2 = "top2_adds_value" in answers

        if has_t1 or has_t2:
            b1, s1 = _parse_noul_val(answers.get("top1_adds_value"), NOUL_THRESHOLD) if has_t1 else (False, None)
            b2, s2 = _parse_noul_val(answers.get("top2_adds_value"), NOUL_THRESHOLD) if has_t2 else (False, None)
            s1_str = f"{s1:.4f}" if s1 is not None else "n/a"
            s2_str = f"{s2:.4f}" if s2 is not None else "n/a"
            raw_repr = f"top1={b1} ({s1_str}), top2={b2} ({s2_str})"

            # Truth Table:
            # true   true  -> BOTH
            # true   false -> TOP_1
            # false  true  -> TOP_2
            # false  false -> NONE
            if b1 and b2:
                if num_candidates >= 2:
                    return "both top1 and top2", "both top1 and top2", raw_repr
                return "Just top 1", "Just top 1", raw_repr
            elif b1 and not b2:
                if num_candidates >= 1:
                    return "Just top 1", "Just top 1", raw_repr
                return "none", DECISION_LABEL_NONE, raw_repr
            elif not b1 and b2:
                if num_candidates >= 2:
                    return "Just top 2", "Just top 2", raw_repr
                return "none", DECISION_LABEL_NONE, raw_repr
            else:
                return "none", DECISION_LABEL_NONE, raw_repr

    val = ""
    if isinstance(resp_data, dict):
        answers = resp_data.get("answers")
        if isinstance(answers, dict) and "answer_selection" in answers:
            sel = answers["answer_selection"]
            val = sel.get("choice") if isinstance(sel, dict) else str(sel)
        elif "answer_selection" in resp_data:
            sel = resp_data["answer_selection"]
            val = sel.get("choice") if isinstance(sel, dict) else str(sel)
        elif "choice" in resp_data:
            val = resp_data["choice"]
        elif "decision" in resp_data:
            val = resp_data["decision"]
        elif "response" in resp_data:
            val = resp_data["response"]
        elif "result" in resp_data:
            val = resp_data["result"]
        else:
            for k, v in resp_data.items():
                if isinstance(v, str) and v.upper().strip() in ("TOP_1", "TOP_2", "BOTH", "NONE"):
                    val = v
                    break

    if not val:
        val = raw_text

    val_upper = str(val).upper().strip()

    if "BOTH" in val_upper:
        if num_candidates >= 2:
            return "both top1 and top2", "both top1 and top2", str(val)
        return "Just top 1", "Just top 1", str(val)

    if "TOP_2" in val_upper or "TOP2" in val_upper:
        if num_candidates >= 2:
            return "Just top 2", "Just top 2", str(val)
        return "none", DECISION_LABEL_NONE, str(val)

    if "TOP_1" in val_upper or "TOP1" in val_upper:
        if num_candidates >= 1:
            return "Just top 1", "Just top 1", str(val)
        return "none", DECISION_LABEL_NONE, str(val)

    if "NONE" in val_upper or "NEITHER" in val_upper:
        return "none", DECISION_LABEL_NONE, str(val)

    state, label = _normalize_decision(str(val), num_candidates)
    return state, label, str(val)


def _get_ollama_base(ollama_url: str) -> str:
    base = ollama_url.rstrip("/")
    for suffix in ("/v1/systemone", "/v1", "/api/generate", "/api"):
        if base.endswith(suffix):
            base = base[:-len(suffix)]
            break
    return base.rstrip("/")


def build_generate_prompt(query: str, results: list[dict]) -> str:
    db_records = []
    for idx, r in enumerate(results[:5], start=1):
        q = r.get("question", "") if isinstance(r, dict) else getattr(r, "question", "")
        a = r.get("answer", "") if isinstance(r, dict) else getattr(r, "answer", "")
        db_records.append(f"[{idx}] Question: {q}\nAnswer: {a}")

    records_str = "\n\n".join(db_records)

    prompt = (
        "You are a question-answering assistant. Answer the user's question using only the supplied database records. "
        "Combine relevant information from multiple records when needed. Do not invent facts or add information not supported by the records. "
        "If the records do not contain enough information, respond exactly: [No appropriate answer found, Ask an agent]. "
        "Preserve numerical values, units, dates, and qualifications accurately.\n\n"
        f"USER QUESTION:\n{query}\n\n"
        f"DATABASE RECORDS:\n{records_str}\n\n"
        "Write a concise answer that addresses every part of the user's question."
    )
    return prompt


async def generate_answer_from_records(
    query: str,
    results: list[dict],
    ollama_url: str,
    ollama_gen_model: str,
) -> tuple[Optional[str], float, Optional[str]]:
    """Query Ollama /api/generate using top 5 DB records."""
    t0 = time.perf_counter()
    if not results:
        return None, 0.0, "No records to generate from"

    prompt = build_generate_prompt(query, results)
    base_url = _get_ollama_base(ollama_url)
    gen_endpoint = f"{base_url}/api/generate"

    payload = {
        "model": ollama_gen_model,
        "prompt": prompt,
        "stream": False,
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "*/*",
        "User-Agent": "ChoiceRAG/1.0",
    }
    req_json_str = json.dumps(payload, ensure_ascii=False)
    log.info("Request sent to Ollama generate (%s):\n%s", gen_endpoint, req_json_str)

    async def _post_gen(url: str):
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            return await client.post(url, headers=headers, content=req_json_str.encode("utf-8"))

    try:
        try:
            resp = await _post_gen(gen_endpoint)
        except (httpx.ConnectError, httpx.ConnectTimeout) as conn_err:
            if "localhost" in gen_endpoint or "127.0.0.1" in gen_endpoint:
                docker_url = gen_endpoint.replace("localhost", "host.docker.internal").replace("127.0.0.1", "host.docker.internal")
                log.info("Connection to %s failed (%s). Retrying generate via Docker host (%s)...", gen_endpoint, conn_err, docker_url)
                resp = await _post_gen(docker_url)
                gen_endpoint = docker_url
            else:
                raise conn_err

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        log.info("Response from Ollama generate (%s) [%d] in %.2f ms:\n%s", gen_endpoint, resp.status_code, elapsed_ms, resp.text)

        if resp.status_code == 200:
            try:
                data = resp.json()
            except Exception:
                data = {}
            ans_text = str(data.get("response", "")).strip()
            return ans_text, elapsed_ms, None
        else:
            err = f"HTTP {resp.status_code}: {resp.text[:200]}"
            log.warning("Ollama generate error: %s", err)
            return None, elapsed_ms, err
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        log.warning("Ollama generate call failed (%s): %s", gen_endpoint, exc)
        return None, elapsed_ms, str(exc)


async def evaluate_decision(
    query: str,
    results: list[dict],
    ollama_url: str,
    ollama_model: str,
    ollama_gen_model: Optional[str] = None,
    synthesize_if_none: bool = False,
) -> tuple[DecisionOut, float]:
    """Ask /v1/systemone decision model whether Top 1, Top 2, both, or none answer the user's question.
    If synthesize_if_none is True and decision is none, synthesize an answer on the fly using /api/generate with top 5 DB records."""
    gen_model = ollama_gen_model or OLLAMA_GEN_MODEL
    if not results:
        return DecisionOut(
            decision="none",
            label=DECISION_LABEL_NONE,
            model=ollama_model,
            url=ollama_url,
            raw_output="",
            error="No matching chunks retrieved from database",
        ), 0.0

    candidate_answers = []
    for idx, r in enumerate(results[:2]):
        cand_id = "TOP_1" if idx == 0 else "TOP_2"
        q_text = r.get("question", "") if isinstance(r, dict) else getattr(r, "question", "")
        a_text = r.get("answer", "") if isinstance(r, dict) else getattr(r, "answer", "")

        # Extract question variants for additional semantic context
        raw_variants = r.get("variants", []) if isinstance(r, dict) else getattr(r, "variants", [])
        if isinstance(raw_variants, str):
            try:
                raw_variants = json.loads(raw_variants)
            except Exception:
                raw_variants = [v.strip() for v in raw_variants.split("|") if v.strip()]
        variants = [str(v).strip() for v in raw_variants if str(v).strip()] if isinstance(raw_variants, list) else []

        # Extract quote text from quote, evidence.quote, or original_text for context
        quote_text = ""
        if isinstance(r, dict):
            quote_text = r.get("quote") or ""
            if not quote_text and isinstance(r.get("evidence"), dict):
                quote_text = r["evidence"].get("quote") or r["evidence"].get("caption_anchor") or ""
            if not quote_text:
                quote_text = r.get("original_text") or ""
        else:
            quote_text = getattr(r, "quote", "") or ""
            if not quote_text:
                ev = getattr(r, "evidence", None)
                if isinstance(ev, dict):
                    quote_text = ev.get("quote") or ev.get("caption_anchor") or ""
            if not quote_text:
                quote_text = getattr(r, "original_text", "") or ""

        quote_text = str(quote_text).strip(' "\'')

        candidate_answers.append({
            "id": cand_id,
            "question": q_text,
            "question_variants": variants,
            "answer": a_text,
            "quote": quote_text,
        })

    q1 = candidate_answers[0]["question"] if len(candidate_answers) >= 1 else ""
    q2 = candidate_answers[1]["question"] if len(candidate_answers) >= 2 else ""

    top1_ctx = f" Here, TOP_1 explains '{q1}'." if q1 else ""
    top2_ctx = f" Here, TOP_2 explains '{q2}'." if q2 else ""

    if len(candidate_answers) >= 2:
        questions = {
            "top1_adds_value": {
                "type": "noul",
                "instructions": (
                    "Evaluate TOP_1 against the entire user question and TOP_2. "
                    "Answer true only if TOP_1 provides relevant, factually supported information that answers at least one part of the user's question and adds distinct information not already provided by TOP_2."
                    + top1_ctx +
                    " Do not reject it merely because TOP_2 answers a different part of the question. "
                    "Answer false if TOP_1 is irrelevant, unsupported, or redundant."
                ),
            },
            "top2_adds_value": {
                "type": "noul",
                "instructions": (
                    "Evaluate TOP_2 against the entire user question and TOP_1. "
                    "Answer true only if TOP_2 provides relevant, factually supported information that answers at least one part of the user's question and adds distinct information not already provided by TOP_1."
                    + top2_ctx +
                    " Do not reject it merely because TOP_1 answers a different part of the question. "
                    "Answer false if TOP_2 is irrelevant, unsupported, or redundant."
                ),
            },
        }
    else:
        questions = {
            "top1_adds_value": {
                "type": "noul",
                "instructions": (
                    "Evaluate TOP_1 against the entire user question. "
                    "Answer true only if TOP_1 provides relevant, factually supported information that answers at least one part of the user's question."
                    + top1_ctx +
                    " Answer false if TOP_1 is irrelevant, unsupported, or redundant."
                ),
            }
        }

    payload = {
        "model": ollama_model,
        "state": {
            "user_question": query,
            "candidate_answers": candidate_answers
        },
        "questions": questions
    }

    base = ollama_url.rstrip("/")
    if base.endswith("/v1/systemone"):
        endpoint_url = base
    elif "/v1" in base:
        endpoint_url = f"{base}/systemone"
    else:
        endpoint_url = f"{base}/v1/systemone"

    headers = {
        "Content-Type": "application/json",
        "Accept": "*/*",
        "User-Agent": "ChoiceRAG/1.0",
    }

    req_json_str = json.dumps(payload, indent=2)
    log.info("Request sent to decision model (%s):\n%s", endpoint_url, req_json_str)

    t0 = time.perf_counter()

    async def _post(target_url: str):
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            return await client.post(target_url, json=payload, headers=headers)

    try:
        try:
            resp = await _post(endpoint_url)
        except (httpx.ConnectError, httpx.ConnectTimeout) as conn_err:
            if "localhost" in endpoint_url or "127.0.0.1" in endpoint_url:
                docker_url = endpoint_url.replace("localhost", "host.docker.internal").replace("127.0.0.1", "host.docker.internal")
                log.info("Connection to %s failed (%s). Retrying via Docker host bridge (%s)...", endpoint_url, conn_err, docker_url)
                resp = await _post(docker_url)
                endpoint_url = docker_url
            else:
                raise conn_err

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        log.info("Response from decision model (%s) [%d] in %.2f ms:\n%s", endpoint_url, resp.status_code, elapsed_ms, resp.text)

        if resp.status_code == 200:
            try:
                data = resp.json()
            except Exception:
                data = {}
            state, label, raw = _parse_systemone_response(data, resp.text, len(results))

            # When decision is "none" and synthesize_if_none is True, synthesize an answer on the fly using /api/generate
            if state == "none" and results and synthesize_if_none:
                log.info("System 1 decision was 'none' and synthesize_if_none=True. Querying Ollama /api/generate (%s)...", gen_model)
                gen_text, gen_ms, gen_err = await generate_answer_from_records(
                    query=query,
                    results=results,
                    ollama_url=endpoint_url,
                    ollama_gen_model=gen_model,
                )
                if gen_text and "[No appropriate answer found, Ask an agent]" not in gen_text and "No appropriate answer found" not in gen_text:
                    evidence = {
                        "type": "generated",
                        "answer_source": "generated_on_the_fly",
                        "generator_model": gen_model,
                        "decision_model": ollama_model,
                        "source_chunks": [r.get("id") if isinstance(r, dict) else getattr(r, "id", "") for r in results[:5]],
                        "timestamp": time.time(),
                    }
                    chunk_id = db().insert_generated_chunk(
                        question=query,
                        answer=gen_text,
                        evidence=evidence,
                        keywords="generated on the fly",
                        source="generated_on_the_fly",
                        answer_source="generated_on_the_fly",
                    )
                    return DecisionOut(
                        decision="generated",
                        label="Generated on the fly",
                        model=gen_model,
                        url=endpoint_url,
                        raw_output=f"decision={raw} -> generated with {gen_model} ({gen_ms:.1f}ms)",
                        generated_answer=gen_text,
                        answer_source="generated_on_the_fly",
                        generation_ms=gen_ms,
                        chunk_id=chunk_id,
                    ), elapsed_ms + gen_ms

            return DecisionOut(
                decision=state,
                label=label,
                model=ollama_model,
                url=endpoint_url,
                raw_output=raw,
            ), elapsed_ms
        else:
            err_text = resp.text[:200]
            log.warning("Systemone error %d at %s: %s", resp.status_code, endpoint_url, err_text)
            return DecisionOut(
                decision="none",
                label=DECISION_LABEL_NONE,
                model=ollama_model,
                url=endpoint_url,
                error=f"HTTP {resp.status_code}: {err_text}",
            ), elapsed_ms
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        log.warning("Systemone call failed (%s): %s", endpoint_url, exc)
        hint = " (If backend is running inside Docker, use http://host.docker.internal:11434)" if "localhost" in endpoint_url else ""
        return DecisionOut(
            decision="none",
            label=DECISION_LABEL_NONE,
            model=ollama_model,
            url=endpoint_url,
            error=f"Unreachable ({endpoint_url}): {exc}{hint}",
        ), elapsed_ms


@app.post("/search", response_model=SearchResponse)
async def search(req: SearchRequest) -> SearchResponse:
    t_start = time.perf_counter()
    fts_q = build_fts_query(req.query)
    if not fts_q:
        raise HTTPException(400, "query has no usable tokens")

    t_db0 = time.perf_counter()
    rows = db().search(fts_q, k=req.top_k, min_score=req.min_score)
    retrieval_ms = round((time.perf_counter() - t_db0) * 1000, 2)

    ollama_url = req.ollama_url or OLLAMA_URL
    ollama_model = req.ollama_model or OLLAMA_MODEL
    ollama_gen_model = req.ollama_gen_model or OLLAMA_GEN_MODEL

    if req.skip_decision:
        decision_out = None
        decision_ms = None
    else:
        decision_out, decision_ms = await evaluate_decision(
            query=req.query,
            results=rows,
            ollama_url=ollama_url,
            ollama_model=ollama_model,
            ollama_gen_model=ollama_gen_model,
        )

    retrieval_sec = round(retrieval_ms / 1000, 2)
    decision_sec = round(decision_ms / 1000, 2) if decision_ms is not None else None
    total_elapsed = round((time.perf_counter() - t_start) * 1000, 2)
    elapsed_sec = round(total_elapsed / 1000, 2)
    return SearchResponse(
        query=req.query,
        returned=len(rows),
        elapsed_ms=total_elapsed,
        elapsed_sec=elapsed_sec,
        retrieval_ms=retrieval_ms,
        retrieval_sec=retrieval_sec,
        decision_ms=decision_ms,
        decision_sec=decision_sec,
        decision=decision_out,
        results=[SearchResult(**r) for r in rows],
    )


@app.get("/search", response_model=SearchResponse, include_in_schema=False)
async def search_get(
    q: str = Query(..., min_length=1),
    k: int = Query(5, ge=1, le=50),
    ollama_url: Optional[str] = Query(None),
    ollama_model: Optional[str] = Query(None),
    ollama_gen_model: Optional[str] = Query(None),
    skip_decision: bool = Query(False),
):
    return await search(
        SearchRequest(
            query=q,
            top_k=k,
            ollama_url=ollama_url,
            ollama_model=ollama_model,
            ollama_gen_model=ollama_gen_model,
            skip_decision=skip_decision,
        )
    )


@app.post("/search/decision", response_model=DecisionResponse)
async def evaluate_candidates_decision(req: DecisionRequest) -> DecisionResponse:
    """Step 2: Evaluate candidate answers using the System 1 decision model asynchronously."""
    ollama_url = req.ollama_url or OLLAMA_URL
    ollama_model = req.ollama_model or OLLAMA_MODEL
    ollama_gen_model = req.ollama_gen_model or OLLAMA_GEN_MODEL
    decision_out, decision_ms = await evaluate_decision(
        query=req.query,
        results=req.results,
        ollama_url=ollama_url,
        ollama_model=ollama_model,
        ollama_gen_model=ollama_gen_model,
        synthesize_if_none=req.synthesize_if_none,
    )
    decision_sec = round(decision_ms / 1000, 2)
    return DecisionResponse(decision=decision_out, decision_ms=decision_ms, decision_sec=decision_sec)


@app.post("/search/generate", response_model=GenerateResponse)
async def generate_candidate_answer(req: GenerateRequest) -> GenerateResponse:
    """Step 3: Synthesize an answer on the fly using Ollama /api/generate with top DB records."""
    ollama_url = req.ollama_url or OLLAMA_URL
    gen_model = req.ollama_gen_model or OLLAMA_GEN_MODEL

    log.info("Step 3 generate triggered for query: '%s' with %d DB records (model: %s)...", req.query, len(req.results), gen_model)
    gen_text, gen_ms, gen_err = await generate_answer_from_records(
        query=req.query,
        results=req.results,
        ollama_url=ollama_url,
        ollama_gen_model=gen_model,
    )
    gen_sec = round(gen_ms / 1000, 2)

    if gen_err:
        return GenerateResponse(
            query=req.query,
            generated_answer=None,
            chunk_id=None,
            generator_model=gen_model,
            generation_ms=gen_ms,
            generation_sec=gen_sec,
            error=gen_err,
            fallback_to_agent=True,
        )

    if gen_text and "[No appropriate answer found, Ask an agent]" not in gen_text and "No appropriate answer found" not in gen_text:
        evidence = {
            "type": "generated",
            "answer_source": "generated_on_the_fly",
            "generator_model": gen_model,
            "source_chunks": [r.get("id") if isinstance(r, dict) else getattr(r, "id", "") for r in req.results[:5]],
            "timestamp": time.time(),
        }
        chunk_id = db().insert_generated_chunk(
            question=req.query,
            answer=gen_text,
            evidence=evidence,
            keywords="generated on the fly",
            source="generated_on_the_fly",
            answer_source="generated_on_the_fly",
        )
        return GenerateResponse(
            query=req.query,
            generated_answer=gen_text,
            chunk_id=chunk_id,
            answer_source="generated_on_the_fly",
            generator_model=gen_model,
            generation_ms=gen_ms,
            generation_sec=gen_sec,
            fallback_to_agent=False,
        )

    return GenerateResponse(
        query=req.query,
        generated_answer=gen_text,
        chunk_id=None,
        answer_source="corpus",
        generator_model=gen_model,
        generation_ms=gen_ms,
        generation_sec=gen_sec,
        fallback_to_agent=True,
    )


@app.get("/ollama/status")
async def ollama_status(url: Optional[str] = Query(None)):
    target_url = (url or OLLAMA_URL).rstrip("/")
    base_url = target_url
    if base_url.endswith("/v1/systemone"):
        base_url = base_url[:-len("/v1/systemone")]
    elif base_url.endswith("/v1"):
        base_url = base_url[:-len("/v1")]

    async def _try_tags(u: str):
        async with httpx.AsyncClient(timeout=3.0) as client:
            return await client.get(f"{u}/api/tags")

    try:
        try:
            resp = await _try_tags(base_url)
        except (httpx.ConnectError, httpx.ConnectTimeout) as conn_err:
            if "localhost" in base_url or "127.0.0.1" in base_url:
                docker_base = base_url.replace("localhost", "host.docker.internal").replace("127.0.0.1", "host.docker.internal")
                resp = await _try_tags(docker_base)
                target_url = target_url.replace("localhost", "host.docker.internal").replace("127.0.0.1", "host.docker.internal")
            else:
                raise conn_err

        if resp.status_code == 200:
            tags = resp.json()
            models = [m.get("name") for m in tags.get("models", [])]
            return {"status": "ok", "url": target_url, "models": models}
        return {"status": "error", "url": target_url, "detail": f"HTTP {resp.status_code}"}
    except Exception as e:
        hint = " — If backend runs in Docker, use http://host.docker.internal:11434" if "localhost" in target_url else ""
        return {"status": "unreachable", "url": target_url, "detail": f"{e}{hint}"}


def _resolve_pdf(doc: Optional[str]) -> Optional[Path]:
    """Map a corpus filename (e.g. 'weekly-...-qa.md') to its source PDF in PDF_DIR."""
    if not PDF_DIR.is_dir():
        return None
    candidates = []
    if doc:
        stem = Path(Path(doc).name).stem  # basename only — no path traversal
        candidates += [stem, re.sub(r"[-_ ]?qa$", "", stem, flags=re.I)]
    if DEFAULT_PDF:
        candidates.append(Path(Path(DEFAULT_PDF).name).stem)
    for c in candidates:
        p = PDF_DIR / f"{c}.pdf"
        if c and p.is_file():
            return p
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    return pdfs[0] if len(pdfs) == 1 else None


@app.get("/pdf", include_in_schema=True)
def get_pdf(doc: Optional[str] = Query(None, description="corpus filename the chunk came from")):
    p = _resolve_pdf(doc)
    if not p:
        raise HTTPException(404, f"no source PDF found for {doc or 'corpus'} in {PDF_DIR}")
    return FileResponse(p, media_type="application/pdf",
                        headers={"Cache-Control": "no-cache"})  # revalidate via ETag after re-uploads


@app.get("/documents")
def list_documents():
    docs = db().stats()["documents"]
    for d in docs:
        p = _resolve_pdf(d["filename"])
        d["pdf"] = p.name if p else None
    return {"documents": docs}


@app.get("/chunks/{chunk_id}", response_model=ChunkOut)
def get_chunk(chunk_id: str) -> ChunkOut:
    row = db().get_chunk(chunk_id)
    if not row:
        raise HTTPException(404, f"chunk {chunk_id} not found")
    return ChunkOut(**row)


@app.post("/admin/ingest", response_model=IngestResult)
async def ingest(file: UploadFile = File(...),
                 x_admin_token: Optional[str] = Header(None),
                 purge_old: bool = Query(False)) -> IngestResult:
    _require_admin(x_admin_token)
    content = await file.read()
    chunks = _parse_md_bytes(content)
    r = db().ingest(filename=file.filename or "upload.md",
                    checksum=hashlib.sha256(content).hexdigest(),
                    chunks=[chunk_to_dict(c) for c in chunks],
                    purge_old=purge_old)
    return IngestResult(**r)


def _parse_md_bytes(content: bytes):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".md")
    try:
        tmp.write(content)
        tmp.close()
        chunks = parse_markdown(Path(tmp.name))
    except UnicodeDecodeError:
        raise HTTPException(400, "markdown must be UTF-8 text")
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
    if not chunks:
        raise HTTPException(400, "no '## Q…' chunks found in markdown")
    return chunks


MAX_PDF_BYTES = int(os.getenv("MAX_PDF_MB", "100")) * 1024 * 1024


@app.post("/admin/upload")
async def upload(markdown: UploadFile = File(..., description="Q&A corpus (.md)"),
                 pdf: Optional[UploadFile] = File(None, description="source PDF"),
                 x_admin_token: Optional[str] = Header(None),
                 purge_old: bool = Query(False)):
    """Ingest a markdown corpus and store its source PDF as PDF_DIR/<md-stem>.pdf."""
    _require_admin(x_admin_token)
    md_name = Path(markdown.filename or "upload.md").name
    if not md_name.lower().endswith((".md", ".markdown")):
        raise HTTPException(400, "markdown file must have a .md extension")

    pdf_bytes = None
    if pdf is not None and pdf.filename:
        pdf_bytes = await pdf.read()
        if not pdf_bytes.startswith(b"%PDF-"):
            raise HTTPException(400, f"{pdf.filename} is not a valid PDF")
        if len(pdf_bytes) > MAX_PDF_BYTES:
            raise HTTPException(413, f"PDF exceeds {MAX_PDF_BYTES // (1024 * 1024)} MB")

    content = await markdown.read()
    chunks = _parse_md_bytes(content)
    r = db().ingest(filename=md_name, checksum=hashlib.sha256(content).hexdigest(),
                    chunks=[chunk_to_dict(c) for c in chunks], purge_old=purge_old)

    pdf_name = None
    if pdf_bytes is not None:
        PDF_DIR.mkdir(parents=True, exist_ok=True)
        target = PDF_DIR / f"{Path(md_name).stem}.pdf"
        tmp = target.with_suffix(".pdf.part")
        tmp.write_bytes(pdf_bytes)
        os.replace(tmp, target)  # atomic swap so readers never see a half-written file
        pdf_name = target.name
        log.info("stored PDF %s (%d bytes) for %s", target, len(pdf_bytes), md_name)

    return {**r, "filename": md_name, "pdf": pdf_name}


@app.post("/admin/activate")
def activate(payload: ActivateRequest, x_admin_token: Optional[str] = Header(None)):
    _require_admin(x_admin_token)
    if not db().activate_version(payload.filename, payload.version):
        raise HTTPException(404, f"{payload.filename} v{payload.version} not found")
    return {"status": "ok", "filename": payload.filename, "version": payload.version}


@app.post("/admin/purge")
def purge(filename: Optional[str] = Query(None),
          x_admin_token: Optional[str] = Header(None)):
    _require_admin(x_admin_token)
    return {"status": "ok", "deleted_documents": db().purge_inactive(filename)}
