"""Semantic support check: does the cited passage actually back the sentence that cites it?

core/verify.py checks FORM (ids exist, nothing typed is scripture, every stretch has a citation). This module checks that a
citation is not decorative: each cited stretch is scored against the passages it cites, and a stretch whose best score is below
a threshold theta is reported as unsupported (the pipeline can feed that back to the generator exactly like a verifier error).

Scorers (SUPPORT_MODEL):
    bge-m3            cosine similarity of bge-m3 embeddings (already loaded for retrieval; fast; topical, not entailment)
    bge-reranker      BAAI/bge-reranker-v2-m3 cross-encoder (multilingual; sigmoid of the relevance logit; slower, sharper)
Mode (SUPPORT_MODE): off | log (default: scores stored in the debug trace, no behaviour change) | enforce (unsupported stretch = verifier error)
"""
import math
import os
import re

from core.normalize import normalize_ar
from core.verify import CITE, PLACEHOLDER, cite_ids

MIN_WORDS = 6
CHUNK_CHARS = 320


def mode() -> str:
    m = os.environ.get("SUPPORT_MODE", "log").lower()
    return m if m in ("off", "log", "enforce") else "log"


def theta() -> float:
    return float(os.environ.get("SUPPORT_THETA", DEFAULT_THETA[scorer_name()]))


# calibrated on evals/support_dev.jsonl (see evals/support_dev.py); recalibrate when the scorer or the generator changes
DEFAULT_THETA = {"bge-m3": 0.49, "bge-reranker": 0.50}  # bge-m3: keeps 99% of cited stretches in the dev set


def scorer_name() -> str:
    return os.environ.get("SUPPORT_MODEL", "bge-m3")


# ---------------------------------------------------------------- text preparation
def stretches(text: str) -> list[dict]:
    """Explanation stretches (between blank lines and placeholders) that carry citations: [{"text", "cites"}]."""
    out = []
    for para in re.split(r"\n\s*\n", text):
        for seg in PLACEHOLDER.split(para)[0::3]:
            ids = cite_ids(seg)
            clean = re.sub(r"[*_#>`]", "", CITE.sub(" ", seg))
            clean = re.sub(r"\s+", " ", clean).strip()
            if ids and len(clean.split()) >= MIN_WORDS:
                out.append({"text": clean, "cites": ids})
    return out


def passage_chunks(p: dict) -> list[str]:
    """Pieces of a passage a sentence can be matched against: its Arabic text in sentence-sized chunks, plus the English."""
    body = (p.get("text") or "").replace("\n", " ")
    parts = [s.strip() for s in re.split(r"(?<=[.!?؟؛])\s+", body) if s.strip()]
    chunks, cur = [], ""
    for s in parts:
        if cur and len(cur) + len(s) > CHUNK_CHARS:
            chunks.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        chunks.append(cur)
    chunks = [c[:CHUNK_CHARS * 2] for c in chunks] or [body[:CHUNK_CHARS * 2]]
    if p.get("text_en"):
        chunks.append(p["text_en"][:CHUNK_CHARS * 2])
    if p.get("title"):
        chunks.append(p["title"])
    return chunks


# ---------------------------------------------------------------- scorers
class BiEncoderScorer:
    name = "bge-m3"

    def scores(self, pairs: list[tuple[str, str]]) -> list[float]:
        from core.embedding import embed_texts
        claims = embed_texts([normalize_ar(c) if re.search("[؀-ۿ]", c) else c for c, _ in pairs])
        chunks = embed_texts([normalize_ar(c) if re.search("[؀-ۿ]", c) else c for _, c in pairs])
        return [float(sum(a * b for a, b in zip(x, y))) for x, y in zip(claims, chunks)]


class CrossEncoderScorer:
    name = "bge-reranker"
    _model = None

    def scores(self, pairs: list[tuple[str, str]]) -> list[float]:
        if CrossEncoderScorer._model is None:
            import truststore
            truststore.inject_into_ssl()
            from sentence_transformers import CrossEncoder
            CrossEncoderScorer._model = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cpu", max_length=512)
        raw = CrossEncoderScorer._model.predict(pairs, batch_size=16, show_progress_bar=False)
        return [1 / (1 + math.exp(-float(x))) for x in raw]


def get_scorer(name: str | None = None):
    return CrossEncoderScorer() if (name or scorer_name()) == "bge-reranker" else BiEncoderScorer()


# ---------------------------------------------------------------- the check
def support_report(text: str, passages_by_id: dict, scorer=None) -> list[dict]:
    """For every cited stretch: the best (cited passage, score) over the passages it cites."""
    scorer = scorer or get_scorer()
    items = []  # (stretch index, cite id, claim, chunk)
    sts = stretches(text)
    for i, st in enumerate(sts):
        for cid in st["cites"]:
            p = passages_by_id.get(cid)
            if p:
                items += [(i, cid, st["text"], c) for c in passage_chunks(p)]
    scores = scorer.scores([(it[2], it[3]) for it in items]) if items else []
    best: dict[int, tuple[float, str]] = {}
    for (i, cid, _, _), s in zip(items, scores):
        if i not in best or s > best[i][0]:
            best[i] = (s, cid)
    return [{"text": st["text"], "cites": st["cites"], "best_id": best.get(i, (0.0, None))[1], "score": round(best.get(i, (0.0, None))[0], 4)}
            for i, st in enumerate(sts)]


def unsupported(text: str, passages: list[dict], th: float | None = None, scorer=None) -> tuple[list[str], list[dict]]:
    """(error messages for stretches scoring below theta, full report)."""
    th = theta() if th is None else th
    report = support_report(text, {p["id"]: p for p in passages}, scorer)
    errs = [f"Cited stretch is not supported by the passage(s) it cites (best match [[{r['best_id']}]], support {r['score']:.2f} < {th:.2f}). "
            "Fix: cite the passage that really says this, rewrite the sentence to say only what the cited passage says, or delete it. "
            f"Stretch: {r['text'][:200]}" for r in report if r["score"] < th]
    return errs, report
