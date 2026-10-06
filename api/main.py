"""Phase 1 API: raw retrieval results. (Phase 2 adds /api/ask, Phase 3 /api/verify.)"""
import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from core import pipeline
from core import reports
from core import retrieve as R
from core import verifier_mode
from core.db import connect

ROOT = Path(__file__).resolve().parent.parent


class _ScrubQuery(logging.Filter):
    """Privacy: never write users' questions to logs (GET /api/retrieve?q=... would otherwise be logged)."""

    def filter(self, record):
        if isinstance(record.args, tuple):
            record.args = tuple(a.split("?", 1)[0] if isinstance(a, str) and "?" in a else a for a in record.args)
        return True


logging.getLogger("uvicorn.access").addFilter(_ScrubQuery())
import collections
import time

_hits: dict[str, collections.deque] = collections.defaultdict(collections.deque)
_llm_gate = threading.BoundedSemaphore(int(os.environ.get("MAX_CONCURRENT_ANSWERS", 2)))   # one GPU: extra askers wait in line instead of piling up


def rate_limit(request: Request, bucket: str, per_minute: int):
    """In-memory per-client limiter (no IP is stored anywhere persistent: the counters live in RAM and expire after a minute)."""
    limit = int(os.environ.get("RATE_LIMIT_PER_MINUTE", per_minute))
    if limit <= 0:
        return
    key = f"{bucket}:{request.client.host if request.client else '?'}"
    q, now = _hits[key], time.time()
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= limit:
        raise HTTPException(429, "too many requests, please wait a moment")
    q.append(now)


_VECTORS = {"ready": False}


def _warm():
    try:
        from core.embedding import embed_texts
        embed_texts(["warm-up"])
        _VECTORS["ready"] = True
    except Exception as e:  # offline first run: retrieval falls back to keyword search; /api/health says so
        logging.getLogger("uvicorn.error").warning("embedding model unavailable (%s): keyword search only", str(e)[:100])


@asynccontextmanager
async def lifespan(_app):
    try:
        reports.purge()  # retention: reports older than REPORT_RETENTION_DAYS are deleted at every start
    except Exception:
        pass
    try:  # an empty database would make every question abstain silently: say so loudly (see docs/QUICKSTART.md)
        n = connect().execute("SELECT count(*) FROM passages").fetchone()[0]
        if n == 0:
            logging.getLogger("uvicorn.error").error("Baseera database is EMPTY: run `python ingest/build_all.py` or unzip the ready-made working copy (docs/QUICKSTART.md)")
    except Exception:
        pass
    if os.environ.get("WARM_EMBEDDINGS", "1") == "1":  # load the retrieval model in the background so the first question is not slow
        threading.Thread(target=_warm, daemon=True).start()
    yield


app = FastAPI(title="Baseera", lifespan=lifespan)
NO_CACHE = {"Cache-Control": "no-cache"}  # browsers must re-check the UI on every load (a stale cached page hid new features)


@app.get("/api/health")
def health():
    con = connect()
    counts = {r[0]: r[1] for r in con.execute("SELECT type, count(*) FROM passages GROUP BY type")}
    from core import llm as L
    d = L.describe()
    reachable = None
    if d.get("provider") == "local":  # is the local model server up? (cheap check, 2 s)
        try:
            import httpx
            reachable = httpx.get(os.environ.get("LOCAL_BASE_URL", "http://localhost:11434/v1").rstrip("/") + "/models", timeout=2).status_code == 200
        except Exception:
            reachable = False
    llm = {"provider": d.get("provider"), "configured": L.llm_available(), "reachable": reachable}
    return {"ok": sum(counts.values()) > 0, "llm": llm, "passages": counts, "database_empty": sum(counts.values()) == 0,
            "vectors_ready": _VECTORS["ready"], "verses": con.execute("SELECT count(*) FROM quran").fetchone()[0]}


@app.get("/api/retrieve")
def retrieve(q: str = Query(..., min_length=1, max_length=500), dorar: bool = False, vectors: bool = True):
    dorar = dorar and os.environ.get("BASEERA_DEBUG") == "1"   # live Dorar calls are not available to anonymous callers
    return {"query": q, "results": R.retrieve(q, vectors=vectors, dorar=dorar)}


class AskBody(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    language: str | None = None


class ReportBody(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    reason: str
    comment: str | None = Field(None, max_length=500)
    consent: bool = False
    outcome: dict | None = None


class VerifyBody(BaseModel):
    text: str = Field(..., min_length=1, max_length=6000)


@app.post("/api/ask")
def ask(body: AskBody, request: Request):
    rate_limit(request, "ask", 30)
    with _llm_gate:
        resp = pipeline.ask(body.question, body.language)  # never debug=True here
    for internal in ("verification_errors", "debug"):  # may contain model-written text / raw outputs: dev and eval use only
        resp.pop(internal, None)
    return resp


@app.post("/api/report")
def report(body: ReportBody, request: Request):
    """Opt-in problem report: stored ONLY with explicit consent (the UI shows the policy next to the checkbox)."""
    rate_limit(request, "report", 10)
    if len(json.dumps(body.outcome or {}, ensure_ascii=False)) > 20000:
        raise HTTPException(413, "outcome too large")
    try:
        reports.add_report(body.question, body.reason, body.comment, body.outcome or {}, body.consent)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True}


@app.get("/api/privacy")
def privacy():
    return {"policy": reports.POLICY, "retention_days": reports.RETENTION_DAYS, "stores": ["question", "answer summary", "cited source ids", "reason", "comment"],
            "never_stores": ["IP address", "user agent", "session or account id", "anything without consent"]}


@app.post("/api/verify")
def verify(body: VerifyBody, request: Request):
    rate_limit(request, "verify", 30)
    return {"mode": "verify", **verifier_mode.verify_text(body.text), "ai_disclosure": pipeline.DISCLOSURE["ar"]}


@app.post("/api/transcribe")
async def transcribe(request: Request, language: str | None = Query(None, pattern="^(ar|en)$")):
    """Ask by voice: raw 16 kHz mono PCM16 in the body -> text for the question box. Audio is never stored."""
    from core import speech as S
    max_bytes = S.SAMPLE_RATE * 2 * (S.MAX_SECONDS + 1)
    if int(request.headers.get("content-length") or 0) > max_bytes:  # refuse before buffering the upload
        raise HTTPException(413, "audio too long")
    data = b""
    async for chunk in request.stream():
        data += chunk
        if len(data) > max_bytes:
            raise HTTPException(413, "audio too long")
    try:
        text = await run_in_threadpool(S.transcribe, data, language)
    except S.AudioError as e:
        raise HTTPException(400, str(e))
    except Exception:
        raise HTTPException(503, "speech recognition unavailable")
    return {"text": text}


@app.get("/api/audio/{surah}")
def audio(surah: int):
    """Recitation audio for a surah (mp3quran, cached). Optional demo feature."""
    from core import audio as A
    if not 1 <= surah <= 114:
        raise HTTPException(404, "surah must be 1-114")
    return A.surah_audio(surah)


@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "index.html", headers=NO_CACHE)


@app.get("/retrieval")
def retrieval_page():
    return FileResponse(ROOT / "web" / "retrieval.html", headers=NO_CACHE)


app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")
