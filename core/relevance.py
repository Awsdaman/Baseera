"""On-topic check: do the passages an answer cites actually contain what the QUESTION asks?

core/verify.py checks form (real ids, nothing typed is scripture) and core/support.py checks that each sentence is backed by the passage
it cites. Neither notices an answer built from real, correctly cited passages that are only loosely related to the question
("is zakat due on jewellery gold?" answered with a general definition of zakat). This check asks the model one narrow question about the
retrieved text, never about its own knowledge, and CODE applies the outcome:

    answers    the passages state the answer                      -> keep the answer
    partial    they discuss the topic but not the specific point  -> keep it, and add the code-owned {{note:no_ruling}} notice if missing
    unrelated  they are about something else                      -> abstain (warm message + referrals), never show the stitched answer

The model can only make an answer MORE cautious here; it can never add content. A failed or unparsable check changes nothing.
Policy (pipeline): "partial" or "unrelated" adds the code-owned {{note:not_direct}} notice; the answer is withheld only when the model says
"unrelated" AND the question's bge-m3 similarity to the best cited passage is below REL_MIN_SIM (two independent weak signals; calibrated on
48 real fiqh questions, where either signal alone was too trigger-happy).
REL_MODE: off | log (verdict stored in the trace only) | enforce (default).
"""
import os

from core import llm as L

SYSTEM = """You judge whether source passages contain the answer to a question. You use ONLY the passages, never your own knowledge.
Reply with ONE JSON object: {"verdict": "answers" | "partial" | "unrelated"}
- answers: the passages state the ruling, meaning or explanation the question asks for, or give the facts from which it follows directly. A Quran verse together with its tafsir counts when the question asks about that verse's meaning.
- partial: the passages are about the same subject but do not state the specific point asked (for example they define the subject, or give a general rule without covering the particular case in the question).
- unrelated: the passages have no real connection to the subject of the question.
Choose "partial" rather than "unrelated" whenever the subject matches even loosely."""

VERDICTS = ("answers", "partial", "unrelated")
REL_MIN_SIM = float(os.environ.get("REL_MIN_SIM", 0.50))


def mode() -> str:
    m = os.environ.get("REL_MODE", "enforce").lower()
    return m if m in ("off", "log", "enforce") else "enforce"


def check(question: str, passages: list[dict], max_chars: int = 1300) -> dict:
    """{"verdict": ..., "error": optional}. Never raises: any problem returns the neutral verdict 'answers' (no behaviour change)."""
    if not passages or not L.llm_available():
        return {"verdict": "answers", "skipped": True}
    body = "\n\n".join(f"[{p['id']}] ({p['type']})\n{(p.get('text') or '').replace(chr(10), ' ')[:max_chars]}" for p in passages)
    user = f"QUESTION:\n{question}\n\nPASSAGES:\n{body}"
    try:
        with L.no_thinking():
            raw = L.get_llm().complete(L.ROUTER_MODEL, SYSTEM, user, max_tokens=int(os.environ.get("LLM_ROUTER_MAX_TOKENS", 300)))
        verdict = L.extract_json(raw).get("verdict")
        if verdict in VERDICTS:
            return {"verdict": verdict}
        return {"verdict": "answers", "error": f"unexpected verdict {verdict!r}"}
    except Exception as e:  # the check is a safety net: never break answering
        return {"verdict": "answers", "error": str(e)[:120]}


def question_similarity(question: str, passages: list[dict]) -> float:
    """Best bge-m3 cosine between the question and any chunk of any cited passage (topical, cheap, no LLM)."""
    import re

    from core.embedding import embed_texts
    from core.normalize import normalize_ar
    from core.support import passage_chunks

    def prep(t):
        return normalize_ar(t) if re.search("[؀-ۿ]", t) else t
    chunks = [c for p in passages for c in passage_chunks(p)]
    if not chunks:
        return 1.0
    q = embed_texts([prep(question)])[0]
    return max(sum(a * b for a, b in zip(q, c)) for c in embed_texts([prep(c) for c in chunks]))


def decide(question: str, passages: list[dict]) -> dict:
    """The full on-topic decision for an answer that already passed the verifier:
    {"verdict", "sim" (when computed), "action": "keep" | "note" | "abstain"}."""
    r = check(question, passages)
    if r["verdict"] == "answers":
        return r | {"action": "keep"}
    if r["verdict"] == "unrelated":
        try:
            r["sim"] = round(question_similarity(question, passages), 3)
        except Exception as e:
            r["error"] = str(e)[:120]
            return r | {"action": "note"}
        if r["sim"] < REL_MIN_SIM:
            return r | {"action": "abstain"}
    return r | {"action": "note"}
