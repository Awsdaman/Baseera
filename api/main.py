"""Phase 1 API: raw retrieval results. (Phase 2 adds /api/ask, Phase 3 /api/verify.)"""
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from core import pipeline
from core import retrieve as R
from core import verifier_mode
from core.db import connect

ROOT = Path(__file__).resolve().parent.parent
app = FastAPI(title="Baseera")


@app.get("/api/health")
def health():
    con = connect()
    counts = {r[0]: r[1] for r in con.execute("SELECT type, count(*) FROM passages GROUP BY type")}
    return {"ok": True, "passages": counts, "verses": con.execute("SELECT count(*) FROM quran").fetchone()[0]}


@app.get("/api/retrieve")
def retrieve(q: str = Query(..., min_length=1, max_length=500), dorar: bool = False, vectors: bool = True):
    return {"query": q, "results": R.retrieve(q, vectors=vectors, dorar=dorar)}


class AskBody(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    language: str | None = None


class VerifyBody(BaseModel):
    text: str = Field(..., min_length=1, max_length=6000)


@app.post("/api/ask")
def ask(body: AskBody):
    return pipeline.ask(body.question, body.language)


@app.post("/api/verify")
def verify(body: VerifyBody):
    return {"mode": "verify", **verifier_mode.verify_text(body.text), "ai_disclosure": pipeline.DISCLOSURE["ar"]}


@app.get("/api/audio/{surah}")
def audio(surah: int):
    """Recitation audio for a surah (mp3quran, cached). Optional demo feature."""
    from core import audio as A
    if not 1 <= surah <= 114:
        raise HTTPException(404, "surah must be 1-114")
    return A.surah_audio(surah)


@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "index.html")


app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")
