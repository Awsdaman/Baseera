"""Answer verification and placeholder substitution. PURE CODE: no LLM anywhere in this module.

The model may only emit:
    {{quran:S:A}} {{quran:S:A1-A2}} {{hadith:SOURCE:ID}} {{tafsir:S:A}}   (sacred / source text, filled by code)
    [[passage-id]]                                                          (citation of a retrieved passage)
We reject: unresolved or unretrieved ids, malformed placeholders, ornate-bracket verses, Arabic quotation marks
around text that is not an attributed quote of a retrieved scholarly passage, text that looks like a Quran verse
or retrieved hadith written by the model, and explanation paragraphs without any citation.
"""
import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz, process

from core.db import connect
from core.normalize import normalize_ar

PLACEHOLDER = re.compile(r"\{\{\s*(quran|hadith|tafsir)\s*:\s*([^{}]+?)\s*\}\}")
ANY_BRACES = re.compile(r"\{\{.*?\}\}", re.S)
CITE = re.compile(r"\[\[\s*([^\[\]\s]+)\s*\]\]")
_QUOTES = [re.compile(p, re.S) for p in ("«([^»]{1,2000})»", '"([^"]{1,2000})"', "“([^”]{1,2000})”", "‘([^’]{1,2000})’")]
_AR = re.compile("[ء-ي]")
_SPLIT = re.compile(r"[.!?؟؛:\n،,;()\[\]]+")
NO_EVIDENCE = "INSUFFICIENT_EVIDENCE"
MIN_QUOTE_WORDS = 3
LEAK_WORDS = 6
LEAK_CUTOFF = 93

_quran_cache = None


def _long_verses() -> list[str]:
    """Normalized verses of >=5 words. Tiny verses ("الرحمن", "طه") fuzzy-match 'inside' any long sentence, so they
    are excluded from the fuzzy leak checks (exact-substring checks still use the whole Mushaf)."""
    global _long_cache
    if _long_cache is None:
        _long_cache = [q[2] for q in _quran() if len(q[2].split()) >= 5]
    return _long_cache


_long_cache = None


def _quran():
    """[(surah, ayah, normalized emlaey, normalized uthmani)] for the whole Mushaf."""
    global _quran_cache
    if _quran_cache is None:
        con = connect()
        _quran_cache = [(r["surah"], r["ayah"], r["text_norm"], normalize_ar(r["text_uthmani"]))
                        for r in con.execute("SELECT surah, ayah, text_norm, text_uthmani FROM quran")]
    return _quran_cache


@dataclass
class VerifyResult:
    ok: bool
    abstain: bool = False
    errors: list[str] = field(default_factory=list)
    blocks: list[dict] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    answer_text: str = ""


def _ids_for_placeholder(kind: str, body: str) -> list[str] | None:
    body = body.strip()
    if kind == "quran":
        m = re.fullmatch(r"(\d{1,3}):(\d{1,3})(?:\s*-\s*(\d{1,3}))?", body)
        if not m:
            return None
        s, a1 = int(m.group(1)), int(m.group(2))
        a2 = int(m.group(3) or a1)
        if a2 < a1 or a2 - a1 > 20:
            return None
        return [f"quran:{s}:{a}" for a in range(a1, a2 + 1)]
    if kind == "tafsir":
        m = re.fullmatch(r"(?:muyassar:)?(\d{1,3}):(\d{1,3})", body)
        return [f"tafsir:muyassar:{m.group(1)}:{m.group(2)}"] if m else None
    if kind == "hadith":
        m = re.fullmatch(r"([a-z]+):([A-Za-z0-9_\-]+)", body)
        return [f"hadith:{m.group(1)}:{m.group(2)}"] if m else None
    return None


def _arabic_words(s: str) -> int:
    return len([w for w in normalize_ar(s).split() if _AR.search(w)])


def _check_quotes(text: str, retrieved: dict, errors: list[str]):
    for rx in _QUOTES:
        for m in rx.finditer(text):
            seg = m.group(1)
            if not _AR.search(seg) or _arabic_words(seg) < MIN_QUOTE_WORDS:
                continue
            n = normalize_ar(seg)
            if any(n in q[2] or n in q[3] for q in _quran()) or \
                    process.extractOne(n, _long_verses(), scorer=fuzz.partial_ratio, score_cutoff=LEAK_CUTOFF):
                errors.append(f"Quran text written by the model inside quotation marks (use a {{{{quran:S:A}}}} placeholder): {seg[:60]}")
                continue
            hadith = [p for p in retrieved.values() if p["type"] == "hadith"]
            if any(n in normalize_ar(p["text"] or "") for p in hadith):
                errors.append(f"Hadith text written by the model (use a {{{{hadith:...}}}} placeholder): {seg[:60]}")
                continue
            tail = text[m.end(): m.end() + 80]
            window = tail + " " + text[max(0, m.start() - 80): m.start()]
            cited = {c for c in CITE.findall(window)}
            src = [p for p in retrieved.values() if p["type"] in ("qa", "tafsir", "term")
                   and n in normalize_ar(p["text"] or "")]
            if not src or not (cited & {p["id"] for p in src}):
                errors.append(f"Unattributed Arabic quotation (must be a verbatim quote of a retrieved passage with its [[id]] right next to it): {seg[:60]}")


def _check_leaks(text: str, retrieved: dict, errors: list[str]):
    clean = CITE.sub(" ", PLACEHOLDER.sub(" ", text))
    qnorms = _long_verses()
    hadith_norms = [(p["id"], normalize_ar(p["text"] or "")) for p in retrieved.values() if p["type"] == "hadith"]
    for chunk in _SPLIT.split(clean):
        n = normalize_ar(chunk)
        if _arabic_words(chunk) < LEAK_WORDS:
            continue
        hit = process.extractOne(n, qnorms, scorer=fuzz.partial_ratio, score_cutoff=LEAK_CUTOFF)
        if hit:
            errors.append(f"Text resembling a Quran verse written by the model (use a placeholder): {chunk.strip()[:60]}")
            continue
        for pid, hn in hadith_norms:
            if fuzz.partial_ratio(n, hn) >= LEAK_CUTOFF:
                errors.append(f"Text resembling retrieved hadith {pid} written by the model: {chunk.strip()[:60]}")
                break


def verify_answer(text: str, retrieved: list[dict], lang: str = "ar") -> VerifyResult:
    text = (text or "").strip()
    if text == NO_EVIDENCE or text.startswith(NO_EVIDENCE) and len(text) < len(NO_EVIDENCE) + 4:
        return VerifyResult(ok=True, abstain=True)
    by_id = {p["id"]: p for p in retrieved}
    errors: list[str] = []
    if not text:
        return VerifyResult(ok=False, errors=["empty answer"])
    if "﴿" in text or "﴾" in text:
        errors.append("Ornate verse brackets are reserved for code-inserted Quran text; use a {{quran:S:A}} placeholder")

    valid_spans = {m.span() for m in PLACEHOLDER.finditer(text)}
    for m in ANY_BRACES.finditer(text):
        if m.span() not in valid_spans:
            errors.append(f"Malformed placeholder: {m.group(0)[:60]}")

    for m in PLACEHOLDER.finditer(text):
        ids = _ids_for_placeholder(m.group(1), m.group(2))
        if ids is None:
            errors.append(f"Malformed placeholder: {m.group(0)}")
            continue
        for pid in ids:
            if pid not in by_id:
                errors.append(f"Placeholder {m.group(0)} refers to {pid}, which was NOT retrieved for this question")
    for cid in CITE.findall(text):
        if cid not in by_id:
            errors.append(f"Citation [[{cid}]] was NOT retrieved for this question")

    _check_quotes(text, by_id, errors)
    _check_leaks(text, by_id, errors)

    if not PLACEHOLDER.search(text) and not CITE.search(text):
        errors.append("Answer has no citation at all")
    for para in [p for p in re.split(r"\n\s*\n", text) if p.strip()]:
        if len(para.split()) >= 8 and not CITE.search(para) and not PLACEHOLDER.search(para):
            errors.append(f"Paragraph without a citation: {para.strip()[:60]}")

    if errors:
        return VerifyResult(ok=False, errors=errors)
    blocks, sources, answer_text = build_blocks(text, by_id, lang)
    return VerifyResult(ok=True, blocks=blocks, sources=sources, answer_text=answer_text)


def _source_card(p: dict, n: int) -> dict:
    return {"n": n, "id": p["id"], "type": p["type"], "source": p["source"], "title": p.get("title"),
            "reference_url": p.get("reference_url"), "grade": p.get("grade")}


def build_blocks(text: str, by_id: dict, lang: str):
    """Substitute exact stored text for placeholders; number the citations. Pure string work."""
    order: dict[str, int] = {}

    def num(pid):
        if pid not in order:
            order[pid] = len(order) + 1
        return order[pid]

    def cite_sub(s):
        return CITE.sub(lambda m: f"[{num(m.group(1))}]", s)

    blocks, plain, pos = [], [], 0

    def push_text(seg):
        seg = cite_sub(seg).strip()
        if seg:
            blocks.append({"kind": "explanation", "text": seg})
            plain.append(seg)

    for m in PLACEHOLDER.finditer(text):
        push_text(text[pos:m.start()])
        pos = m.end()
        kind, ids = m.group(1), _ids_for_placeholder(m.group(1), m.group(2))
        ps = [by_id[i] for i in ids]
        for p in ps:
            num(p["id"])
        if kind == "quran":
            s = ps[0]["meta"]["surah"]
            ref = f"{s}:{ps[0]['meta']['ayah']}" + (f"-{ps[-1]['meta']['ayah']}" if len(ps) > 1 else "")
            block = {"kind": "quran", "ref": ref, "surah_name_ar": ps[0]["meta"]["surah_name_ar"],
                     "text_ar": " ".join(p["text"] for p in ps), "text_en": " ".join(p["text_en"] or "" for p in ps).strip(),
                     "translation_source": ps[0]["meta"].get("translation_source"), "ids": ids,
                     "reference_url": ps[0]["reference_url"]}
            plain.append(f"﴿{block['text_ar']}﴾ [{block['surah_name_ar']}: {ref.split(':')[1]}]")
        elif kind == "tafsir":
            p = ps[0]
            block = {"kind": "tafsir", "ref": f"{p['meta']['surah']}:{p['meta']['ayah']}", "text_ar": p["text"],
                     "work": p["meta"].get("work"), "ids": ids, "reference_url": p["reference_url"]}
            plain.append(f"({block['work']}) {p['text']}")
        else:
            p = ps[0]
            block = {"kind": "hadith", "id": p["id"], "text_ar": p["text"], "text_en": p.get("text_en"),
                     "grade": p["grade"], "grade_source": "HadeethEnc.com" if p["source"] == "hadeethenc" else p["source"],
                     "attribution": p["meta"].get("attribution_ar") or p["meta"].get("book"),
                     "reference": p["meta"].get("reference"), "ids": ids, "reference_url": p["reference_url"]}
            plain.append(f"«{p['text']}» ({block['attribution'] or ''} — الدرجة: {p['grade']})")
        blocks.append(block)
    push_text(text[pos:])
    sources = [_source_card(by_id[pid], n) for pid, n in sorted(order.items(), key=lambda x: x[1])]
    return blocks, sources, "\n\n".join(plain)
