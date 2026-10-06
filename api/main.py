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
import itertools
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

_hits: dict[str, collections.deque] = collections.defaultdict(collections.deque)


def client_ip(request: Request) -> str:
    """Behind the Cloudflare tunnel every request comes from 127.0.0.1: TRUST_CF_IP=1 (uvicorn bound to 127.0.0.1 only) uses the real address."""
    if os.environ.get("TRUST_CF_IP") == "1":
        ip = request.headers.get("cf-connecting-ip")
        if ip:
            return ip[:64]
    return request.client.host if request.client else "?"


def rate_limit(request: Request, bucket: str, per_minute: int):
    """In-memory per-client limiter (no IP is stored anywhere persistent: the counters live in RAM and expire after a minute)."""
    limit = int(os.environ.get("RATE_LIMIT_PER_MINUTE", per_minute))
    if limit <= 0:
        return
    key = f"{bucket}:{client_ip(request)}"
    q, now = _hits[key], time.time()
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= limit:
        raise HTTPException(429, "too many requests, please wait a moment")
    q.append(now)


_VECTORS = {"ready": False, "llm": False}


def _warm():
    try:
        from core.embedding import embed_texts
        embed_texts(["warm-up"])
        _VECTORS["ready"] = True
    except Exception as e:  # offline first run: retrieval falls back to keyword search; /api/health says so
        logging.getLogger("uvicorn.error").warning("embedding model unavailable (%s): keyword search only", str(e)[:100])


def _warm_llm():
    from core import llm as L
    try:
        _VECTORS["llm"] = L.ping()
    except Exception:
        _VECTORS["llm"] = False


def _keepalive():
    """Local model server: a small call every few minutes while idle keeps the model loaded in VRAM (no cold start for the next judge)."""
    from core import llm as L
    from core.embedding import embed_texts
    every = int(os.environ.get("KEEPALIVE_S", 240))
    while True:
        time.sleep(every)
        try:
            if L.queue_stats()["active"] == 0:
                embed_texts(["ping"])
                _VECTORS["llm"] = L.ping()
        except Exception:
            _VECTORS["llm"] = False


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
    from core import llm as L
    if L.provider_name() == "local" and os.environ.get("WARM_LLM", "1") == "1":
        threading.Thread(target=_warm_llm, daemon=True).start()
        threading.Thread(target=_keepalive, daemon=True).start()
    if os.environ.get("SPEECH_PRELOAD") == "1":
        from core import speech
        speech.preload()  # load the voice model in the background so the first recording does not wait for it
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
    return {"ok": sum(counts.values()) > 0, "llm": llm, "speech": _speech_status(), "passages": counts, "database_empty": sum(counts.values()) == 0,
            "vectors_ready": _VECTORS["ready"], "llm_warm": _VECTORS["llm"], "queue": L.queue_stats(), "verses": con.execute("SELECT count(*) FROM quran").fetchone()[0]}


def _speech_status() -> dict:
    from core import speech
    return speech.describe()


@app.get("/api/retrieve")
def retrieve(request: Request, q: str = Query(..., min_length=1, max_length=500), dorar: bool = False, vectors: bool = True):
    rate_limit(request, "retrieve", 20)
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


def _run_ask(question, language):
    resp = pipeline.ask(question, language)  # never debug=True here
    for internal in ("verification_errors", "debug"):  # may contain model-written text / raw outputs: dev and eval use only
        resp.pop(internal, None)
    return resp


def _run_verify(text):
    return {"mode": "verify", **verifier_mode.verify_text(text), "ai_disclosure": pipeline.DISCLOSURE["ar"]}


# ---- jobs: a question is a short job the browser polls (Cloudflare drops any single request after ~100 s) ----
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_job_counter = itertools.count(1)
_durations: collections.deque = collections.deque(maxlen=10)
_executor = ThreadPoolExecutor(max_workers=int(os.environ.get("JOB_WORKERS", 8)))
JOB_TTL_S = 180


def _capacity() -> int:
    from core import llm as L
    return L.GATE.capacity if L.GATE else int(os.environ.get("MAX_CONCURRENT_ANSWERS", 2))


def _purge_jobs(now: float):
    for k in [k for k, j in _jobs.items() if j["state"] in ("done", "error") and now - j["finished"] > JOB_TTL_S]:
        del _jobs[k]


def _job_main(jid: str, kind: str, payload: str, language):
    from core import llm as L
    t0 = time.time()
    with _jobs_lock:
        _jobs[jid]["state"] = "running"
        seq = _jobs[jid]["seq"]
    try:
        with L.request_seq(seq):
            res = _run_ask(payload, language) if kind == "ask" else _run_verify(payload)
        out = {"state": "done", "result": res}
    except Exception as e:  # never log the text; the class name is enough
        logging.getLogger("uvicorn.error").warning("job failed: %s", type(e).__name__)
        out = {"state": "error", "error": "busy" if type(e).__name__ == "LLMBusy" else "failed"}
    with _jobs_lock:
        _durations.append(time.time() - t0)
        _jobs[jid].update(out, finished=time.time())


def _position(jid: str) -> int:
    """0 = being served, n = n-th in line beyond the free slots (jobs are served in start order). Call with the lock held."""
    ahead = sorted(j["seq"] for j in _jobs.values() if j["state"] in ("queued", "running"))
    r = ahead.index(_jobs[jid]["seq"])
    return max(0, r - _capacity() + 1)


class JobBody(BaseModel):
    kind: str = Field(..., pattern="^(ask|verify)$")
    text: str = Field(..., min_length=1, max_length=6000)
    language: str | None = None


@app.post("/api/jobs")
def create_job(body: JobBody, request: Request):
    rate_limit(request, "ask" if body.kind == "ask" else "verify", 30)
    if body.kind == "ask" and len(body.text) > 2000:
        raise HTTPException(422, "question too long")
    now = time.time()
    with _jobs_lock:
        _purge_jobs(now)
        pending = sum(1 for j in _jobs.values() if j["state"] in ("queued", "running"))
        if pending >= _capacity() + int(os.environ.get("MAX_QUEUE", 10)):
            raise HTTPException(503, "busy")
        jid = uuid.uuid4().hex[:16]
        _jobs[jid] = {"state": "queued", "seq": next(_job_counter), "created": now}
        pos = _position(jid)
    _executor.submit(_job_main, jid, body.kind, body.text, body.language)
    return {"job": jid, "position": pos}


@app.get("/api/jobs/{jid}")
async def job_status(jid: str):
    with _jobs_lock:
        j = _jobs.get(jid)
        if j is None:
            raise HTTPException(404, "unknown job")
        if j["state"] == "done":
            return {"state": "done", "result": j["result"]}
        if j["state"] == "error":
            return {"state": "error", "error": j["error"]}
        pos = _position(jid)
        avg = (sum(_durations) / len(_durations)) if _durations else 45.0
        return {"state": j["state"], "position": pos, "est_wait_s": int(avg * (pos / _capacity() + 0.5)) if pos else 0}


@app.get("/api/queue")
async def queue():
    from core import llm as L
    return L.queue_stats()


@app.post("/api/ask")
def ask(body: AskBody, request: Request):
    rate_limit(request, "ask", 30)
    return _run_ask(body.question, body.language)


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
    return _run_verify(body.text)


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
def audio(surah: int, request: Request):
    rate_limit(request, "audio", 20)
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
