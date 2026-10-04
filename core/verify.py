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

PLACEHOLDER = re.compile(r"\{\{\s*(quran|hadith|tafsir|note)\s*:\s*([^{}]+?)\s*\}\}")
ANY_BRACES = re.compile(r"\{\{.*?\}\}", re.S)
CITE = re.compile(r"\[\[\s*([^\[\]]+?)\s*\]\]")  # one bracket may list several ids: [[a, b]]
_CITE_ID = re.compile(r"[a-z\-]+:[A-Za-z0-9_:\-]+")

# Code-owned scope/disclaimer notes. The model may only ask for one by key; the text is ours, so it needs no citation
# (and is never counted as one). This lets an answer say "not exhaustive" or "ask a scholar" without the model
# writing an uncitable sentence.
NOTES = {
    "partial": {"ar": "ملاحظة: هذه الإجابة تقتصر على ما ورد في المصادر المعتمدة المتاحة لبصيرة، وليست حصرًا شاملًا للمسألة.",
                "en": "Note: this answer covers only what Baseera's approved sources provide; it is not exhaustive."},
    "refer": {"ar": "للتفصيل في أحكام هذه المسألة يُرجى الرجوع إلى عالم أو جهة إفتاء مؤهلة.",
              "en": "For the detailed rulings on this matter, please consult a qualified scholar or fatwa authority."},
    "no_ruling": {"ar": "ملاحظة: لم أجد في المصادر المعتمدة المتاحة لي نصًّا يبيّن الحكم الدقيق في هذه النقطة (كالوجوب أو الاستحباب)، وما سبق هو ما ورد فيها؛ فيُرجى سؤال عالم أو جهة إفتاء مؤهلة.",
                  "en": "Note: the approved sources available to me do not state the specific ruling on this point (for example whether it is obligatory or recommended); what is above is what they contain. Please ask a qualified scholar."},
    "disputed": {"ar": "هذه مسألة اختلف فيها أهل العلم؛ وما سبق عرضٌ لما ورد في المصادر المتاحة دون ترجيح بينها.",
                 "en": "Scholars differ on this matter; the above presents what the available sources say, without choosing between the views."},
}


def cite_ids(text: str) -> list[str]:
    """Every passage id cited in `text`, expanding [[a, b]] / [[a; b]] into separate ids."""
    out = []
    for group in CITE.findall(text):
        out += [i for i in re.split(r"[,;،\s]+", group.strip()) if i]
    return out
_ID = r"[a-z\-]+:[A-Za-z0-9_:\-]+"
_SPLIT_LIST = re.compile(rf"\[\[\s*{_ID}\s*\](?:\s*[,،;]?\s*\[\s*{_ID}\s*\])+\s*\]")      # [[a], [b], [c]]
_PH_THEN_CITE = re.compile(r"[ \t]*[:：]?[ \t]*(\{\{\s*(quran|hadith|tafsir)\s*:\s*([^{}]+?)\s*\}\})[ \t]*(\[\[[^\[\]]+\]\])[ \t]*([.،]?)")


def _canon_id(i: str, by_id: dict) -> str:
    """Spell a citation id the way the PASSAGES list does. Only documented short forms are mapped, and only to ids that were retrieved."""
    if i in by_id:
        return i
    m = re.fullmatch(r"tafsir:(?:muyassar:)?(\d{1,3}):(\d{1,3})", i)
    if m and f"tafsir:muyassar:{m[1]}:{m[2]}" in by_id:
        return f"tafsir:muyassar:{m[1]}:{m[2]}"
    m = re.fullmatch(r"quran:(\d{1,3}):(\d{1,3})-(\d{1,3})", i)
    if m and 0 <= int(m[3]) - int(m[2]) <= 20:
        ids = [f"quran:{m[1]}:{a}" for a in range(int(m[2]), int(m[3]) + 1)]
        if all(x in by_id for x in ids):
            return ", ".join(ids)
    return i  # unknown: left as written so the normal "NOT retrieved" error names it


def normalize_citations(text: str, by_id: dict) -> str:
    """Repair citation SYNTAX only (small models write [[a], [b]], short ids, or a placeholder followed by its own citation).
    Every resulting id is still checked against the retrieved set by verify_answer: this never adds a source the model did not name."""
    text = text or ""
    text = _SPLIT_LIST.sub(lambda m: "[[" + ", ".join(re.findall(_ID, m.group(0))) + "]]", text)

    def fix(m):
        ids = [_canon_id(i, by_id) for i in re.split(r"[,;،\s]+", m.group(1).strip()) if i]
        return "[[" + ", ".join(ids) + "]]"
    text = CITE.sub(fix, text)

    def move(m):  # '... {{hadith:X}} [[hadith:X]].' -> the citation backs the sentence before it; the placeholder stands alone
        ph, kind, body, cite, dot = m.groups()
        want = _ids_for_placeholder(kind, body)
        if want and set(cite_ids(cite)) == set(want):
            return " " + cite + dot + chr(10) * 2 + ph + chr(10) * 2
        return m.group(0)
    return _PH_THEN_CITE.sub(move, text)


_QUOTES = [re.compile(p, re.S) for p in ("«([^»]{1,2000})»", '"([^"]{1,2000})"', "“([^”]{1,2000})”", "‘([^’]{1,2000})’")]
_AR = re.compile("[ء-ي]")
_SPLIT = re.compile(r"[.!?؟؛:\n،,;()\[\]]+")
NO_EVIDENCE = "INSUFFICIENT_EVIDENCE"
MIN_QUOTE_WORDS = 3
LEAK_WORDS = 6
LEAK_CUTOFF = 93
LONG_RUN_WORDS = 9      # an unquoted near-copy of at least this many words is treated as typed sacred text
COVER_RATIO = 0.6       # ...or a shorter one that reproduces >= 60% of the verse / hadith it matches

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
            words = len(n.split())
            # Typed scripture = a long run, or a quote that covers most of a verse. A short formula inside a longer verse
            # (the shahada «لا إله إلا الله», «بسم الله») is a stock phrase, not a recitation.
            leak = any(_is_leak(n, words, q[2] if n in q[2] else q[3]) for q in _quran() if n in q[2] or n in q[3])
            if not leak:
                hit = process.extractOne(n, _long_verses(), scorer=fuzz.partial_ratio, score_cutoff=LEAK_CUTOFF)
                leak = bool(hit and _is_leak(n, words, hit[0]))
            if leak:
                errors.append(f"Quran text written by the model inside quotation marks (use a {{{{quran:S:A}}}} placeholder): {seg[:60]}")
                continue
            hadith = [p for p in retrieved.values() if p["type"] == "hadith"]
            if any(n in normalize_ar(p["text"] or "") and _is_leak(n, words, normalize_ar(p["text"] or "")) for p in hadith):
                errors.append(f"Hadith text written by the model (use a {{{{hadith:...}}}} placeholder): {seg[:60]}")
                continue
            tail = text[m.end(): m.end() + 80]
            window = tail + " " + text[max(0, m.start() - 80): m.start()]
            cited = set(cite_ids(window))
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
        nwords = len(n.split())
        hit = process.extractOne(n, qnorms, scorer=fuzz.partial_ratio, score_cutoff=LEAK_CUTOFF)
        if hit and _is_leak(n, nwords, hit[0]):
            errors.append(f"Text resembling a Quran verse written by the model (use a placeholder): {chunk.strip()[:60]}")
            continue
        for pid, hn in hadith_norms:
            if fuzz.partial_ratio(n, hn) >= LEAK_CUTOFF and _is_leak(n, nwords, hn):
                errors.append(f"Text resembling retrieved hadith {pid} written by the model: {chunk.strip()[:60]}")
                break


def _is_leak(chunk_norm: str, nwords: int, source_norm: str) -> bool:
    """A near-copy counts as typed sacred text when it is a long run (>= LONG_RUN_WORDS) or covers most of the source.
    Short stock phrases ("حج البيت لمن استطاع إليه سبيلا") that explanations legitimately reuse are allowed;
    anything inside quotation marks is still checked separately by _check_quotes."""
    return nwords >= LONG_RUN_WORDS or len(chunk_norm) >= COVER_RATIO * len(source_norm)


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

    if lang == "ar":  # one language per answer: Latin words are allowed only inside parentheses (approved glossary terms)
        plain = re.sub(r"\([^)]*\)", " ", CITE.sub(" ", PLACEHOLDER.sub(" ", text)))
        loose = re.findall(_ID, ANY_BRACES.sub(" ", plain))   # passage ids left outside a valid [[ ]] (e.g. [[a], [b]]) are a syntax slip, not English
        if loose:
            errors.append(f"Malformed citation: {', '.join(dict.fromkeys(loose))[:100]} must be inside ONE pair of double brackets, "
                          f"e.g. [[{loose[0]}, {loose[-1]}]] (never [[a], [b]])")
            plain = re.sub(_ID, " ", plain)
        plain = ANY_BRACES.sub(" ", plain)   # an invalid {{...}} is reported as a malformed placeholder, not as English words
        stray = re.findall(r"[A-Za-z]{3,}", plain)
        if stray:
            errors.append(f"English word(s) inside an Arabic answer: {', '.join(dict.fromkeys(stray))[:80]}. Write the whole answer in Arabic "
                          "(a glossary term may appear in parentheses).")

    valid_spans = {m.span() for m in PLACEHOLDER.finditer(text)}
    for m in ANY_BRACES.finditer(text):
        if m.span() not in valid_spans:
            errors.append(f"Malformed placeholder: {m.group(0)[:60]}")

    for m in PLACEHOLDER.finditer(text):
        if m.group(1) == "note":
            if m.group(2).strip() not in NOTES:
                errors.append(f"Unknown note {m.group(0)}: the only notes are {{{{note:partial}}}}, {{{{note:refer}}}}, {{{{note:disputed}}}}, {{{{note:no_ruling}}}}")
            continue
        ids = _ids_for_placeholder(m.group(1), m.group(2))
        if ids is None:
            errors.append(f"Malformed placeholder: {m.group(0)}")
            continue
        for pid in ids:
            if pid not in by_id:
                errors.append(f"Placeholder {m.group(0)} refers to {pid}, which was NOT retrieved for this question")
    for group in CITE.findall(text):
        parts = [i for i in re.split(r"[,;،\s]+", group.strip()) if i]
        if not parts or not all(_CITE_ID.fullmatch(i) for i in parts):
            errors.append(f"Malformed citation [[{group[:60]}]]: put exactly one passage id, e.g. [[qa:bayyinat:17]] (several ids may be separated by commas)")
            continue
        for cid in parts:
            if cid not in by_id:
                errors.append(f"Citation [[{cid}]] was NOT retrieved for this question")

    _check_quotes(text, by_id, errors)
    _check_leaks(text, by_id, errors)

    # At least one real source must be shown or cited; a {{note:...}} is code-owned text and never counts as a source.
    if not CITE.search(text) and not any(m.group(1) != "note" for m in PLACEHOLDER.finditer(text)):
        errors.append("Answer has no citation at all")
    # Every stretch of explanation (between blank lines and between placeholders) of 8+ words needs its own explicit
    # [[passage-id]]: a placeholder inserts source text but does not back up the sentences around it.
    for para in re.split(r"\n\s*\n", text):
        for seg in PLACEHOLDER.split(para)[0::3]:  # split() with 2 groups -> [text, kind, body, text, kind, body, text]
            if len(seg.split()) >= 8 and not CITE.search(seg):
                errors.append("Explanation without an explicit [[passage-id]] citation. Fix: put the [[id]] of the passage that supports it "
                              "right after the sentence (any listed id may be cited, including quran:/hadith: ids), or delete the sentence. "
                              "For a scope or referral remark use {{note:partial}}, {{note:refer}} or {{note:disputed}} instead of writing it. "
                              f"Rejected text: {seg.strip()[:200]}")

    if errors:
        return VerifyResult(ok=False, errors=list(dict.fromkeys(errors)))
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
        return CITE.sub(lambda m: "".join(f"[{num(i)}]" for i in re.split(r"[,;،\s]+", m.group(1).strip()) if i), s)

    blocks, plain, pos = [], [], 0

    def push_text(seg):
        seg = cite_sub(seg).strip()
        if seg:
            blocks.append({"kind": "explanation", "text": seg})
            plain.append(seg)

    for m in PLACEHOLDER.finditer(text):
        push_text(text[pos:m.start()])
        pos = m.end()
        kind = m.group(1)
        if kind == "note":
            note = NOTES[m.group(2).strip()]["en" if lang == "en" else "ar"]
            blocks.append({"kind": "notice", "text": note, "note": m.group(2).strip()})
            plain.append(note)
            continue
        ids = _ids_for_placeholder(kind, m.group(2))
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
