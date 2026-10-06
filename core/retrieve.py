"""Hybrid retrieval: FTS5 BM25 over normalized Arabic/English + Chroma vectors, merged by reciprocal rank fusion.

Every result: {source, id, text, text_en, title, reference_url, grade, type, meta, score}
"""
import json
import os
import re
import sqlite3
import threading

from core.db import connect
from core.normalize import light_stem, normalize_ar, search_form

TYPES = ("quran", "hadith", "qa", "tafsir", "term")
DEFAULT_PER_TYPE = {"quran": 5, "hadith": 5, "qa": 4, "tafsir": 3, "term": 2}
RRF_K = 60
_STOP = set("""من في على الى عن ما هل هو هي هذا هذه ذلك التي الذي ان انه انها كان كيف لماذا ماذا متي اين و او ثم لا لم لن قد
the a an of to in is are was be and or for on with what why how does do did can i you it that this as at by from
who which when where about""".split())
_REF = re.compile(r"(?<!\d)(\d{1,3})\s*[:：]\s*(\d{1,3})(?:\s*-\s*(\d{1,3}))?(?!\d)")

_tls = threading.local()


def con():
    """One SQLite connection per thread (a connection cannot be shared across threads; evals run cases in parallel)."""
    c = getattr(_tls, "con", None)
    if c is None:
        c = connect()
        c.row_factory = sqlite3.Row
        _tls.con = c
    return c


def _fts_query(query: str) -> str:
    toks = [t for t in search_form(query).split() if t not in _STOP and len(t) > 1]
    toks = list(dict.fromkeys(toks))[:24]
    return " OR ".join('"%s"' % t.replace('"', "") for t in toks)


def keyword_search(query: str, ptype: str, k: int = 30, sources: tuple | None = None) -> list[str]:
    q = _fts_query(query)
    if not q:
        return []
    sql = ("SELECT f.id FROM passages_fts f JOIN passages p ON p.id=f.id WHERE passages_fts MATCH ? AND p.type=?")
    args: list = [q, ptype]
    if sources:
        sql += " AND p.source IN (%s)" % ",".join("?" * len(sources))
        args += list(sources)
    rows = con().execute(sql + " ORDER BY bm25(passages_fts) LIMIT ?", (*args, k)).fetchall()
    return [r["id"] for r in rows]


def vector_search(query: str, ptype: str, k: int = 30) -> list[str]:
    from core.embedding import COLLECTION, chroma_client, embed_texts
    col = chroma_client().get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    if col.count() == 0:
        return []
    res = col.query(query_embeddings=embed_texts([query], query=True), n_results=k, where={"type": ptype})
    return res["ids"][0]


def rrf(rankings: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    score: dict[str, float] = {}
    for ranking in rankings:
        for rank, pid in enumerate(ranking):
            score[pid] = score.get(pid, 0.0) + 1.0 / (k + rank + 1)
    return score


def _row_to_result(r, score: float) -> dict:
    meta = json.loads(r["meta"] or "{}")
    meta.pop("full_text", None)
    return {"source": r["source"], "id": r["id"], "type": r["type"], "title": r["title"], "text": r["text_ar"],
            "text_en": r["text_en"], "reference_url": r["reference_url"], "grade": r["grade"], "meta": meta,
            "score": round(score, 5)}


def get_passage(pid: str) -> dict | None:
    r = con().execute("SELECT * FROM passages WHERE id=?", (pid,)).fetchone()
    return _row_to_result(r, 0.0) if r else None


def verse_ref_lookup(query: str) -> list[dict]:
    """Explicit references like '2:255' or '2:255-257' resolve directly (rank 1)."""
    out = []
    for m in _REF.finditer(query):
        s, a1 = int(m.group(1)), int(m.group(2))
        a2 = int(m.group(3) or a1)
        for a in range(a1, min(a2, a1 + 9) + 1):
            p = get_passage(f"quran:{s}:{a}")
            if p:
                p["score"] = 1.0
                out.append(p)
    return out


# Q&A slots are filled group by group in this order (n = how many the caller wants): curated Q&A first, then the fiqh encyclopedia,
# then the books (keyword only, numerous, so a small share), then reference cards (people, places, sects, Names of Allah).
QA_GROUPS = [("curated", ("icadb", "bayyinat"), 4), ("fiqh", ("dorar-feqhia",), 3), ("books", ("icadb-books",), 2),
             ("reference", ("icadb-aalam", "icadb-places", "icadb-firaq", "icadb-asma"), 1)]
_src_cache: dict[str, str] = {}


def _source_of(pid: str) -> str | None:
    if pid not in _src_cache:
        r = con().execute("SELECT source FROM passages WHERE id=?", (pid,)).fetchone()
        _src_cache[pid] = r["source"] if r else None
    return _src_cache[pid]


def retrieve(query: str, per_type: dict | None = None, vectors: bool = True, dorar: bool = False) -> list[dict]:
    per_type = per_type or DEFAULT_PER_TYPE
    results, seen = [], set()
    from core.aliases import named_refs
    for p in verse_ref_lookup(query):
        results.append(p)
        seen.add(p["id"])
    for pid in named_refs(query):  # "آية الكرسي", "سورة الإخلاص"
        p = get_passage(pid)
        if p and pid not in seen:
            p["score"] = 1.0
            results.append(p)
            seen.add(pid)
    for ptype in TYPES:
        n = per_type.get(ptype, 0)
        if not n:
            continue
        if ptype == "qa":  # the Q&A type now mixes curated Q&A, a fiqh encyclopedia, reference cards and 40k book chunks: rank each group on its own
            vec = []
            if vectors:
                try:
                    vec = vector_search(query, "qa")
                except Exception as e:
                    print("vector search unavailable:", e)
            picked = []
            off = set(filter(None, os.environ.get("DISABLE_SOURCES", "").split(",")))  # rollback switch: DISABLE_SOURCES=icadb-books,dorar-feqhia
            for _name, srcs, share in QA_GROUPS:
                srcs = tuple(x for x in srcs if x not in off)
                if not srcs:
                    continue
                kw = keyword_search(query, "qa", sources=srcs)
                vv = [i for i in vec if _source_of(i) in srcs]
                picked += [(pid, sc) for pid, sc in sorted(rrf([kw, vv] if vv else [kw]).items(), key=lambda x: -x[1])[:share]]
            for pid, sc in picked[:n]:
                if pid in seen:
                    continue
                row = con().execute("SELECT * FROM passages WHERE id=?", (pid,)).fetchone()
                if row:
                    results.append(_row_to_result(row, sc))
                    seen.add(pid)
            continue
        rankings = [keyword_search(query, ptype)]
        if vectors:
            try:
                rankings.append(vector_search(query, ptype))
            except Exception as e:  # vector index may not be built yet; keyword still works
                print("vector search unavailable:", e)
        for pid, sc in sorted(rrf(rankings).items(), key=lambda x: -x[1])[:n]:
            if pid in seen:
                continue
            row = con().execute("SELECT * FROM passages WHERE id=?", (pid,)).fetchone()
            if row:
                results.append(_row_to_result(row, sc))
                seen.add(pid)
    # A retrieved tafsir passage always brings its verse along, so the Quran text is available to cite.
    for r in list(results):
        if r["type"] == "tafsir":
            qid = "quran:{surah}:{ayah}".format(**r["meta"])
            if qid not in seen:
                v = get_passage(qid)
                if v:
                    v["score"] = r["score"]
                    results.append(v)
                    seen.add(qid)
    if dorar:
        from core import dorar as dorar_client
        try:
            for h in dorar_client.search(query)[:5]:
                results.append({"source": "dorar", "id": h["id"], "type": "hadith", "title": h["book"], "text": h["text"],
                                "text_en": None, "reference_url": h["reference_url"], "grade": h["grade"],
                                "meta": {k: h[k] for k in ("grader", "narrator", "book", "page_or_number")}, "score": 0.0})
        except Exception as e:
            print("dorar unavailable:", e)
    return results
