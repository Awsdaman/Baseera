"""VERIFY mode: extract verse / hadith claims from pasted text and check them against approved data.

Verses  -> rapidfuzz + word-level alignment against the full local Mushaf (exact / misquoted / fabricated).
Hadiths -> Dorar (live, scholars' grades) + HadeethEnc (local); grades are copied from those sources, never the model.
The only LLM use is claim EXTRACTION (Haiku); a regex extractor is the fallback.
"""
import difflib
import re

from rapidfuzz import fuzz, process

from core import dorar
from core import llm as L
from core.db import connect
from core.normalize import normalize_ar

MAX_INPUT = 6000
VERSE_MISQUOTE_MIN = 0.72      # word-level similarity to call a claim a "misquoted" verse
HADITH_MATCH_MIN = 85          # rapidfuzz partial_ratio to accept a Dorar / HadeethEnc hit as the same hadith
_PBUH = re.compile(r"(ﷺ|صلى الله عليه وسلم|صلّى الله عليه وسلّم|صلي الله عليه وسلم|عليه الصلاة والسلام|\(ص\))")

EXTRACT_SYSTEM = """Extract every checkable religious claim from the user's pasted message.
Return ONLY a JSON array. Each item: {"type": "verse"|"hadith"|"quote", "text": "<the claimed wording copied EXACTLY from the message, without the introducing phrase>", "claimed_ref": "<reference given in the message or null>", "attributed_to": "<who the message says said it or null>"}
- verse: text presented as Quran (inside ﴿﴾, or after "قال تعالى", "قال الله", "in the Quran").
- hadith: text presented as the Prophet's saying (after "قال رسول الله ﷺ", "عن النبي", "hadith").
- quote: any other quotation attributed to a person (a companion, scholar, imam).
Copy the text verbatim (including any mistakes); never correct, translate or complete it. If there are no claims return []."""

_quran_rows = None


def _qrows():
    global _quran_rows
    if _quran_rows is None:
        rows = connect().execute(
            "SELECT surah, ayah, text_uthmani, text_emlaey, text_norm, surah_name_ar FROM quran ORDER BY surah, ayah").fetchall()
        _quran_rows = [dict(r) | {"norm_u": normalize_ar(r["text_uthmani"])} for r in rows]
    return _quran_rows


# ---------------------------------------------------------------- extraction
def _strip_marks(s: str) -> str:
    return re.sub(r"\s+", " ", _PBUH.sub("", s)).strip(" \t\n\"'«»“”﴿﴾()[]:：-–—.،")


def heuristic_extract(text: str) -> list[dict]:
    text = text[:MAX_INPUT]
    claims, used = [], []

    def add(kind, t, start, end, ref=None):
        t = _strip_marks(t)
        if len(t.split()) >= 2 and not any(s <= start < e for s, e in used):
            claims.append({"type": kind, "text": t, "claimed_ref": ref, "attributed_to": None})
            used.append((start, end))

    for m in re.finditer(r"﴿([^﴾]+)﴾(?:\s*[\[(]\s*([^\])]{2,40})\s*[\])])?", text):
        add("verse", m.group(1), m.start(), m.end(), m.group(2))
    for m in re.finditer(r"(?:قال الله تعالى|قال تعالى|قال الله|يقول الله تعالى|يقول الله)\s*[:：]?\s*[«\"“]?([^\n«»\"”﴿﴾]{8,400})", text):
        add("verse", m.group(1), m.start(1), m.end(1))
    for m in re.finditer(r"(?:قال|يقول|عن)\s+(?:رسول الله|النبي|نبينا|الرسول)[^:：\n«\"“]{0,30}[:：]\s*[«\"“]?([^\n«»\"”]{8,500})", text):
        add("hadith", m.group(1), m.start(1), m.end(1))
    for m in re.finditer(r"(?:وقال|قال|ويقول|يقول)\s*(?:ﷺ|صلى الله عليه وسلم)\s*[:：]\s*[«\"“]?([^\n«»\"”]{8,500})", text):
        add("hadith", m.group(1), m.start(1), m.end(1))
    for m in re.finditer(r"(?:حديث|الحديث)\s*[:：]\s*[«\"“]?([^\n«»\"”]{8,500})", text):
        add("hadith", m.group(1), m.start(1), m.end(1))
    for m in re.finditer(r"[«“\"]([^»”\"\n]{12,500})[»”\"]", text):
        if not any(s <= m.start(1) < e for s, e in used) and re.search("[ء-ي]", m.group(1)):
            add("quote", m.group(1), m.start(1), m.end(1))
    # English
    for m in re.finditer(r"(?:the prophet|prophet muhammad|muhammad)[^:\n]{0,40}(?:said|says)\s*[:,]?\s*[\"“]([^\"”\n]{10,400})[\"”]", text, re.I):
        add("hadith", m.group(1), m.start(1), m.end(1))
    return claims


def extract_claims(text: str) -> tuple[list[dict], str]:
    text = text[:MAX_INPUT]
    if L.llm_available():
        try:
            raw = L.get_llm().complete(L.ROUTER_MODEL, EXTRACT_SYSTEM, text, max_tokens=1500)
            items = L.extract_json(raw)
            claims = []
            for it in items if isinstance(items, list) else []:
                t = _strip_marks(str(it.get("text", "")))
                if it.get("type") in ("verse", "hadith", "quote") and len(t.split()) >= 2:
                    claims.append({"type": it["type"], "text": t, "claimed_ref": it.get("claimed_ref"),
                                   "attributed_to": it.get("attributed_to")})
            return claims, "llm"
        except Exception as e:
            claims = heuristic_extract(text)
            return claims, f"heuristic (llm error: {str(e)[:80]})"
    return heuristic_extract(text), "heuristic"


# ---------------------------------------------------------------- verses
def _word_sim(claim: list[str], correct: list[str]):
    """Best word-level alignment of `claim` against any window of `correct`. -> (similarity, window_start, window_end)"""
    n = len(claim)
    best = (0.0, 0, min(len(correct), n))
    if n == 0 or not correct:
        return best
    for size in {max(1, n - 2), n, n + 2}:
        size = min(size, len(correct))
        for i in range(0, max(1, len(correct) - size + 1)):
            seg = correct[i:i + size]
            r = difflib.SequenceMatcher(None, claim, seg, autojunk=False).ratio()
            if r > best[0]:
                best = (r, i, i + size)
    return best


def _verse_dict(row):
    return {"surah": row["surah"], "ayah": row["ayah"], "ref": f"{row['surah']}:{row['ayah']}",
            "surah_name_ar": row["surah_name_ar"], "text_ar": row["text_uthmani"],
            "reference_url": f"https://quranenc.com/en/browse/english_saheeh/{row['surah']}#{row['ayah']}",
            "id": f"quran:{row['surah']}:{row['ayah']}"}


def _diff(claim_words: list[str], correct_words_norm: list[str], correct_words_show: list[str]):
    sm = difflib.SequenceMatcher(None, claim_words, correct_words_norm, autojunk=False)
    out = []
    for op, a1, a2, b1, b2 in sm.get_opcodes():
        claimed, correct = " ".join(claim_words[a1:a2]), " ".join(correct_words_show[b1:b2])
        if op == "equal":
            out.append({"op": "equal", "text": correct})
        elif op == "replace":
            out.append({"op": "replace", "claimed": claimed, "correct": correct})
        elif op == "delete":
            out.append({"op": "extra", "claimed": claimed})
        else:
            out.append({"op": "missing", "correct": correct})
    return out


def check_verse(claim_text: str, claimed_ref: str | None = None) -> dict:
    rows = _qrows()
    cn = normalize_ar(claim_text)
    words = cn.split()
    base = {"claim_type": "verse", "claim": claim_text, "claimed_ref": claimed_ref}
    if len(words) < 2:
        return base | {"verdict": "unverifiable", "color": "grey", "note": "النص قصير جدًا للتحقق منه."}

    padded = f" {cn} "
    exact = [r for r in rows if padded in f" {r['text_norm']} " or padded in f" {r['norm_u']} "]
    if not exact:  # a claim spanning 2-3 consecutive verses of one surah
        for i in range(len(rows) - 1):
            for k in (2, 3):
                grp = rows[i:i + k]
                if len(grp) == k and len({g["surah"] for g in grp}) == 1:
                    joined = " " + " ".join(g["text_norm"] for g in grp) + " "
                    if padded in joined:
                        exact = grp
                        break
            if exact:
                break
    if exact:
        full = len(exact) == 1 and normalize_ar(exact[0]["text_emlaey"]) == cn or cn in (exact[0]["norm_u"],)
        v = [_verse_dict(r) for r in exact[:3]]
        return base | {"verdict": "verified", "color": "green", "match": v[0], "matches": v, "score": 1.0,
                       "note": "نصٌّ قرآني صحيح." if full else "جزء من آية صحيحة النص.", "diff": []}

    # fuzzy: shortlist verses by partial_ratio on both spellings, then refine word by word
    short = process.extract(cn, [r["text_norm"] for r in rows], scorer=fuzz.partial_ratio, limit=6)
    short += process.extract(cn, [r["norm_u"] for r in rows], scorer=fuzz.partial_ratio, limit=4)
    cand = {i for _, _, i in short}
    best = None
    for i in cand:
        for span in (1, 2, 3):
            grp = rows[i:i + span]
            if len(grp) < span or len({g["surah"] for g in grp}) != 1:
                continue
            norm_words = " ".join(g["text_norm"] for g in grp).split()
            sim, a, b = _word_sim(words, norm_words)
            gap = abs(len(norm_words) - len(words))  # on equal similarity prefer the verse closest in length
            if best is None or (sim, -gap) > (best[0], best[5]):
                best = (sim, grp, a, b, norm_words, -gap)
    if best and best[0] >= VERSE_MISQUOTE_MIN:
        sim, grp, a, b, norm_words, _ = best
        show_words = " ".join(g["text_emlaey"] for g in grp).split()
        if len(show_words) != len(norm_words):
            show_words = norm_words
        v = _verse_dict(grp[0])
        if len(grp) > 1:
            v["text_ar"] = " ".join(g["text_uthmani"] for g in grp)
            v["ref"] = f"{grp[0]['surah']}:{grp[0]['ayah']}-{grp[-1]['ayah']}"
        return base | {"verdict": "misquoted", "color": "amber", "match": v, "score": round(sim, 3),
                       "diff": _diff(words, norm_words[a:b], show_words[a:b]),
                       "note": "قريبة من آية في المصحف لكن الصياغة المنقولة تختلف عن النص الصحيح."}
    return base | {"verdict": "fabricated", "color": "red", "score": round(best[0], 3) if best else 0.0,
                   "note": "لم نجد هذا النص في المصحف (مجمع الملك فهد)؛ فليس آية قرآنية بهذا اللفظ."}


# ---------------------------------------------------------------- hadith grades
_FAB = ("موضوع", "لا اصل له", "لا اصل لها", "مكذوب", "باطل", "كذب", "لا اصل")
_NEG = ("لا يصح", "ليس بصحيح", "لا يثبت", "لم يصح", "لم يثبت", "غير صحيح", "لا تصح", "لا يصحح")
_WEAK = ("ضعيف", "ضعفه", "منكر", "شاذ", "واه", "متروك", "مضطرب", "معلول", "فيه ضعف", "ضعف")
_SOUND = ("صحيح", "صححه", "حسن", "جيد", "قوي", "ثابت", "صحيحه")


def classify_grade(grade: str | None) -> str:
    """'sound' | 'weak' | 'fabricated' | 'unknown' from a scholar's grade text (keyword rules, no model)."""
    g = normalize_ar(grade or "")
    if not g:
        return "unknown"
    if any(normalize_ar(w) in g for w in _FAB):
        return "fabricated"
    if any(normalize_ar(w) in g for w in _NEG):
        return "weak"
    if any(normalize_ar(w) in g for w in _WEAK):
        return "weak"
    if any(normalize_ar(w) in g for w in _SOUND):
        return "sound"
    return "unknown"


def _hadeethenc_matches(cn: str) -> list[dict]:
    from core import retrieve as R
    out = []
    for pid in R.keyword_search(cn, "hadith", k=15):
        p = R.get_passage(pid)
        if not p:
            continue
        score = fuzz.partial_ratio(cn, normalize_ar(p["text"]))
        if score >= HADITH_MATCH_MIN:
            out.append({"source": "hadeethenc", "id": p["id"], "grade": p["grade"], "grade_class": classify_grade(p["grade"]),
                        "grader": "HadeethEnc.com", "book": p["meta"].get("attribution_ar"), "reference": p["meta"].get("reference"),
                        "text": p["text"], "score": score, "reference_url": p["reference_url"]})
    return sorted(out, key=lambda m: -m["score"])[:3]


def _dorar_matches(claim: str, cn: str) -> tuple[list[dict], str | None]:
    words = claim.split()
    queries = [" ".join(words[:10])]
    if len(words) > 10:
        queries += [" ".join(words[:5]), " ".join(words[-6:])]
    seen, out, err = set(), [], None
    for q in queries:
        try:
            hits = dorar.search(q)
        except Exception as e:  # keep the other source; report honestly that Dorar was unreachable
            err = str(e)[:120]
            continue
        for h in hits:
            hn = normalize_ar(h["text"])
            if h["id"] in seen or len(hn.split()) < 3:
                continue
            score = max(fuzz.partial_ratio(cn, hn), fuzz.partial_ratio(hn, cn) if len(hn) >= 0.7 * len(cn) else 0)
            if score >= HADITH_MATCH_MIN:
                seen.add(h["id"])
                out.append({"source": "dorar", "id": h["id"], "grade": h["grade"], "grade_class": classify_grade(h["grade"]),
                            "grader": h["grader"], "book": h["book"], "reference": h["page_or_number"], "narrator": h["narrator"],
                            "text": h["text"], "score": score, "reference_url": h["reference_url"]})
        if out:
            break
    return sorted(out, key=lambda m: -m["score"])[:8], err


def check_hadith(claim_text: str, claimed_ref: str | None = None) -> dict:
    claim = _strip_marks(claim_text)
    cn = normalize_ar(claim)
    base = {"claim_type": "hadith", "claim": claim_text, "claimed_ref": claimed_ref}
    if len(cn.split()) < 3:
        return base | {"verdict": "unverifiable", "color": "grey", "note": "النص قصير جدًا للتحقق منه."}
    matches = _hadeethenc_matches(cn)
    d, err = _dorar_matches(claim, cn)
    matches += d
    if not matches:
        note = "لم يُعثر على هذا الحديث في المصادر المعتمدة المتاحة (الدرر السنية / HadeethEnc). لا نستطيع الحكم عليه، فلا تنسبه إلى النبي ﷺ قبل التثبت من أهل الحديث."
        if err:
            note += " (تعذّر الاتصال بالدرر السنية مؤقتًا.)"
        return base | {"verdict": "not_found", "color": "grey", "matches": [], "note": note, "source_error": err}
    classes = {m["grade_class"] for m in matches}
    curated_sound = [m for m in matches if m["source"] == "hadeethenc" and m["grade_class"] == "sound" and m["score"] >= 90]
    if curated_sound:
        extra = len([m for m in matches if m["source"] == "dorar" and m["grade_class"] != "sound"])
        verdict, color = "sound", "green"
        note = "حديث ثابت: أورده موقع HadeethEnc ضمن الأحاديث الصحيحة المحقَّقة." + (
            f" (وفي الدرر السنية {extra} حكمًا آخر على روايات أو أسانيد أخرى بنفس اللفظ؛ انظرها أدناه.)" if extra else "")
    elif classes == {"sound"}:
        verdict, color, note = "sound", "green", "حديث ثابت بحسب أحكام العلماء في المصادر المعتمدة."
    elif "sound" not in classes and "fabricated" in classes:
        verdict, color, note = "fabricated", "red", "حكم العلماء عليه: موضوع أو لا أصل له — لا تنسبه إلى النبي ﷺ."
    elif "sound" in classes:
        verdict, color, note = "mixed", "amber", "اختلفت أحكام العلماء أو الروايات؛ انظر الدرجات والمصادر أدناه."
    elif "weak" in classes:
        verdict, color, note = "weak", "amber", "حديث ضعيف بحسب أحكام العلماء في المصادر المعتمدة."
    else:
        verdict, color, note = "graded", "grey", "وُجد الحديث لكن تعذّر تصنيف درجته آليًا؛ انظر نصوص أحكام العلماء أدناه."
    return base | {"verdict": verdict, "color": color, "matches": matches, "note": note, "source_error": err}


def check_quote(claim_text: str, claimed_ref: str | None = None, attributed_to: str | None = None) -> dict:
    """Attributed quote: if it is really a verse or hadith we say so; otherwise honestly 'not verifiable'."""
    v = check_verse(claim_text, claimed_ref)
    if v["verdict"] in ("verified",):
        return v | {"claim_type": "quote", "note": "هذا النص قرآني: " + v["note"]}
    h = check_hadith(claim_text, claimed_ref)
    if h["verdict"] not in ("not_found", "unverifiable"):
        return h | {"claim_type": "quote"}
    return {"claim_type": "quote", "claim": claim_text, "claimed_ref": claimed_ref, "attributed_to": attributed_to,
            "verdict": "not_found", "color": "grey",
            "note": "لم نجد هذا القول في المصادر المعتمدة المتاحة؛ لا نستطيع تأكيد نسبته."}


def verify_text(text: str) -> dict:
    claims, how = extract_claims(text)
    results = []
    for c in claims:
        if c["type"] == "verse":
            results.append(check_verse(c["text"], c.get("claimed_ref")))
        elif c["type"] == "hadith":
            results.append(check_hadith(c["text"], c.get("claimed_ref")))
        else:
            results.append(check_quote(c["text"], c.get("claimed_ref"), c.get("attributed_to")))
    tally = {}
    for r in results:
        tally[r["color"]] = tally.get(r["color"], 0) + 1
    return {"extraction": how, "claims": results, "summary": tally,
            "message": None if results else "لم نعثر على آيات أو أحاديث في النص المُدخل."}


def find_embedded_verses(question: str) -> list[dict]:
    """For ASK mode: verses quoted inside a question, checked against the Mushaf (heuristic extraction only)."""
    return [check_verse(c["text"], c.get("claimed_ref")) for c in heuristic_extract(question) if c["type"] == "verse"]
