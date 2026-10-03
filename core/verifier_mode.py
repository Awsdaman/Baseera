"""VERIFY mode: extract verse / hadith claims from pasted text and check them against approved data.

Verses  -> rapidfuzz + word-level alignment against the full local Mushaf (exact / misquoted / fabricated).
Hadiths -> Dorar (live, scholars' grades) + HadeethEnc (local); grades are copied from those sources, never the model.
The only LLM use is claim EXTRACTION (Haiku); a regex extractor is the fallback.
"""
import collections
import difflib
import re

from rapidfuzz import fuzz, process

from core import dorar
from core import llm as L
from core.db import connect
from core.normalize import normalize_ar

MAX_INPUT = 6000
VERSE_MISQUOTE_MIN = 0.72      # word-level similarity to call a claim a "misquoted" verse
SIM_TIE = 0.08                 # a curated (HadeethEnc) wording this close to the best one is preferred
HADITH_WORD_SIM_MIN = 0.80    # word-level similarity that also counts as the same hadith (catches short claims missing a word)
HADITH_MATCH_MIN = 85          # rapidfuzz partial_ratio to accept a Dorar / HadeethEnc hit as the same hadith
_PBUH = re.compile(r"(ﷺ|صلى الله عليه وسلم|صلّى الله عليه وسلّم|صلي الله عليه وسلم|عليه الصلاة والسلام|\(ص\))")

EXTRACT_SYSTEM = """Extract every checkable religious claim from the user's pasted message.
Return ONLY a JSON array. Each item: {"type": "verse"|"hadith"|"quote", "text": "<the claimed wording copied EXACTLY from the message, without the introducing phrase>", "claimed_ref": "<reference given in the message or null>", "attributed_to": "<who the message says said it or null>"}
- verse: text presented as Quran (inside ﴿﴾, or after "قال تعالى", "قال الله", "in the Quran").
- hadith: text presented as the Prophet's saying (after "قال رسول الله ﷺ", "عن النبي", "hadith").
- quote: any other quotation attributed to a person (a companion, scholar, imam).
Copy the text verbatim (including any mistakes); never correct, translate or complete it. If there are no claims return []."""

_INTRO_ONLY = re.compile(r"(?:و?قال|و?يقول|عن)?\s*(?:رسول الله|النبي|نبينا|الرسول|الله تعالي|الله|تعالي)(?:\s+(?:تعالي|عنه|عليه))?")
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
        if _INTRO_ONLY.fullmatch(normalize_ar(t)):  # just "قال رسول الله" etc., not a claim
            return
        if len(t.split()) >= 2 and not any(s <= start < e for s, e in used):
            claims.append({"type": kind, "text": t, "claimed_ref": ref, "attributed_to": None})
            used.append((start, end))

    for m in re.finditer(r"﴿([^﴾]+)﴾(?:\s*[\[(]\s*([^\])]{2,40})\s*[\])])?", text):
        add("verse", m.group(1), m.start(), m.end(), m.group(2))
    for m in re.finditer(r"(?:قال الله تعالى|قال تعالى|قال الله|يقول الله تعالى|يقول الله)\s*[:：]?\s*[«\"“]?([^\n«»\"”﴿﴾]{8,400})", text):
        add("verse", m.group(1), m.start(1), m.end(1))
    for m in re.finditer(r"(?:قال|يقول|عن)\s+(?:رسول الله|النبي|نبينا|الرسول)[^:：\n«\"“]{0,30}[:：]\s*[«\"“]?([^\n«»\"”]{8,500})", text):
        add("hadith", m.group(1), m.start(1), m.end(1))
    for m in re.finditer(r"(?:رسول الله|النبي)[^:：\n«\"“]{0,40}يقول\s*[:：]\s*[«\"“]?([^\n«»\"”]{8,500})", text):
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
            if not claims:  # the model's extraction is not stable (it sometimes returns []): the rule-based extractor gets a turn
                claims = heuristic_extract(text)
                return claims, "heuristic (llm found nothing)" if claims else "llm"
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
    for size in sorted({max(1, n - 2), max(1, n - 1), n, n + 1, n + 2}):
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


def _diff(claim_words: list[str], correct_words_norm: list[str], correct_words_show: list[str], claim_show: list[str] | None = None):
    sm = difflib.SequenceMatcher(None, claim_words, correct_words_norm, autojunk=False)
    out = []
    for op, a1, a2, b1, b2 in sm.get_opcodes():
        claimed, correct = " ".join((claim_show or claim_words)[a1:a2]), " ".join(correct_words_show[b1:b2])
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

    # fuzzy: shortlist candidate verses on both spellings, then refine word by word. partial_ratio alone is a trap found with
    # synthetic data (evals/synth_verify.py): a SHORT verse that fits entirely inside the claim scores 100 and crowds the right
    # verse out of the shortlist, so each shortlist below is restricted to the cases where its score is meaningful.
    clen = len(cn)
    cand: set[int] = set()
    for key in ("text_norm", "norm_u"):
        texts = [r[key] for r in rows]
        # (a) the claim sits (nearly) inside one verse: only verses at least ~as long as the claim can contain it
        idx = [i for i, t in enumerate(texts) if len(t) >= 0.8 * clen]
        cand |= {idx[j] for _, _, j in process.extract(cn, [texts[i] for i in idx], scorer=fuzz.partial_ratio, limit=6)}
        # (b) the claim is about as long as a verse: whole-string similarity
        cand |= {i for _, _, i in process.extract(cn, texts, scorer=fuzz.ratio, limit=6)}
        # (c) the claim spans several verses: the longest verses (>= 5 words) that sit inside it
        inside = [i for i, t in enumerate(texts) if len(t.split()) >= 5 and len(t) < clen]
        hits = process.extract(cn, [texts[i] for i in inside], scorer=fuzz.partial_ratio, score_cutoff=88, limit=None)
        cand |= {i for _, i in sorted(((len(texts[inside[j]]), inside[j]) for _, _, j in hits), reverse=True)[:6]}
    best = None
    for i0 in sorted(cand):
        for i in (i0, i0 - 1, i0 - 2):  # a multi-verse claim may start one or two verses before the shortlisted one
            for span in (1, 2, 3):
                grp = rows[i:i + span] if i >= 0 else []
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


# Formulae that occur in nearly every hadith: they say nothing about WHICH hadith a claim is (normalized spelling).
FORMULA = set("""الله رسول صلي عليه وسلم رضي عنه عنها عنهما عنهم قال قالت النبي يا ان من في علي الي عن ما لا هو هي ثم قد كان كانت اذا ذلك هذا
هذه الذي اللهم سمعت يقول فقال قلت فقلت انه انها او ولا فلا لم لن اي كل بن ابي ابن""".split())
SHORT_CLAIM_WORDS = 8     # below this many words only the word-level alignment may accept a match
MIN_SHARED_IDF = 8.0          # summed rarity (idf) of the distinctive words a SHORT near-match must share with the hadith
MIN_CONTENT_WORDS = 2  # a claim needs at least this many distinctive words to be checkable


_df = None


def _idf_sum(words) -> float:
    """Sum of ln(N/df) over words, with df counted over the HadeethEnc texts (rare words weigh more)."""
    import math
    global _df
    if _df is None:
        _df = collections.Counter()
        rows = connect().execute("SELECT text_ar FROM passages WHERE source='hadeethenc'").fetchall()
        for r in rows:
            _df.update({_stem(w) for w in normalize_ar(r[0]).split()})
        _df["__n__"] = len(rows)
    n = _df["__n__"]
    return sum(math.log(n / max(_df.get(w, 0), 1)) for w in words)


def content_words(words: list[str]) -> list[str]:
    return [w for w in (_stem(x) for x in words) if len(w) > 1 and w not in FORMULA]


def _stem(w: str) -> str:
    """Drop a leading conjunction (و / ف) for ALIGNMENT only: 'فاتقوا' and 'اتقوا' are the same word of the hadith."""
    return w[1:] if len(w) > 3 and w[0] in "وف" else w


def _align(claim_words: list[str], text: str) -> dict:
    """Word-level alignment of a claim against one known hadith text: best window, similarity, exactness, display words."""
    nwords = normalize_ar(text).split()
    nstem = [_stem(w) for w in nwords]
    sim, a, b = _word_sim([_stem(w) for w in claim_words], nstem)
    # exact = the claim's words appear in the text in order, ignoring a leading connecting و / ف (quoting 'اتقوا' for 'فاتقوا' is exact)
    padded = " " + " ".join(_stem(w) for w in claim_words) + " "
    exact = padded in (" " + " ".join(nstem) + " ")
    show = text.split()
    if len(show) != len(nwords):  # punctuation made the token counts differ: fall back to the normalized words
        show = nwords
    return {"sim": sim, "exact": exact, "window_norm": nwords[a:b], "window_stem": nstem[a:b], "window_show": show[a:b]}


def _candidate(source: str, cid: str, text: str, claim_words: list[str], cn: str, **fields) -> dict | None:
    """Accept a known hadith text as a match for the claim: either whole-string similarity or a word-level near-match."""
    score = max(fuzz.partial_ratio(cn, normalize_ar(text)), fuzz.partial_ratio(normalize_ar(text), cn) if len(text) <= 1.4 * len(cn) else 0)
    al = _align(claim_words, text)
    fuzzy_ok = score >= HADITH_MATCH_MIN and len(claim_words) >= SHORT_CLAIM_WORDS  # whole-string fuzzy matching is unreliable for short claims
    if not fuzzy_ok and al["sim"] < HADITH_WORD_SIM_MIN:
        return None
    # the match must rest on distinctive words, not on shared boilerplate ("قال رسول الله صلى الله عليه وسلم")
    cc = set(content_words(claim_words))
    shared = len(cc & set(content_words(al["window_norm"])))
    if shared < min(MIN_CONTENT_WORDS, len(cc)):
        return None
    if not al["exact"] and len(claim_words) < SHORT_CLAIM_WORDS:
        # A short claim that is not word-for-word in the hadith counts as a misquote of it only if the words they share are RARE in
        # the corpus ("تمرة") and not common narrator / narration words ("أنس بن مالك"): rarity-weighted evidence.
        if _idf_sum(cc & set(content_words(al["window_norm"]))) < MIN_SHARED_IDF:
            return None
    return {"source": source, "id": cid, "text": text, "score": score, "sim": round(al["sim"], 3), "exact": al["exact"],
            "window_norm": al["window_norm"], "window_stem": al["window_stem"], "window_show": al["window_show"], **fields}


def _hadeethenc_matches(cn: str) -> list[dict]:
    from core import retrieve as R
    cw = cn.split()
    out = []
    for pid in R.keyword_search(cn, "hadith", k=20):
        p = R.get_passage(pid)
        if not p:
            continue
        m = _candidate("hadeethenc", p["id"], p["text"], cw, cn, grade=p["grade"], grade_class=classify_grade(p["grade"]),
                       grader="HadeethEnc.com", book=p["meta"].get("attribution_ar"), reference=p["meta"].get("reference"),
                       reference_url=p["reference_url"])
        if m:
            out.append(m)
    return sorted(out, key=lambda m: (m["exact"], m["sim"], m["score"]), reverse=True)[:3]


def _dorar_matches(claim: str, cn: str) -> tuple[list[dict], str | None]:
    words = claim.split()
    cw = cn.split()
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
            if h["id"] in seen or len(normalize_ar(h["text"]).split()) < 3:
                continue
            m = _candidate("dorar", h["id"], h["text"], cw, cn, grade=h["grade"], grade_class=classify_grade(h["grade"]),
                           grader=h["grader"], book=h["book"], reference=h["page_or_number"], narrator=h["narrator"],
                           reference_url=h["reference_url"])
            if m:
                seen.add(h["id"])
                out.append(m)
        if out:
            break
    return sorted(out, key=lambda m: (m["exact"], m["sim"], m["score"]), reverse=True)[:8], err


def _grade_verdict(matches: list[dict], curated_min_score: int = 90) -> tuple[str, str, str]:
    """(verdict, color, note) from the scholars' grades of the given matches. HadeethEnc's curated 'sound' decides when present."""
    classes = {m["grade_class"] for m in matches}
    curated_sound = [m for m in matches if m["source"] == "hadeethenc" and m["grade_class"] == "sound" and m["score"] >= curated_min_score]
    if curated_sound:
        extra = len([m for m in matches if m["source"] == "dorar" and m["grade_class"] != "sound"])
        return "sound", "green", "حديث ثابت: أورده موقع HadeethEnc ضمن الأحاديث الصحيحة المحقَّقة." + (
            f" (وفي الدرر السنية {extra} حكمًا آخر على روايات أو أسانيد أخرى بنفس اللفظ؛ انظرها أدناه.)" if extra else "")
    if classes == {"sound"}:
        return "sound", "green", "حديث ثابت بحسب أحكام العلماء في المصادر المعتمدة."
    if "sound" not in classes and "fabricated" in classes:
        return "fabricated", "red", "حكم العلماء عليه: موضوع أو لا أصل له — لا تنسبه إلى النبي ﷺ."
    if "sound" in classes:
        return "mixed", "amber", "اختلفت أحكام العلماء أو الروايات؛ انظر الدرجات والمصادر أدناه."
    if "weak" in classes:
        return "weak", "amber", "حديث ضعيف بحسب أحكام العلماء في المصادر المعتمدة."
    return "graded", "grey", "وُجد الحديث لكن تعذّر تصنيف درجته آليًا؛ انظر نصوص أحكام العلماء أدناه."


def check_hadith(claim_text: str, claimed_ref: str | None = None) -> dict:
    claim = _strip_marks(claim_text)
    cn = normalize_ar(claim)
    base = {"claim_type": "hadith", "claim": claim_text, "claimed_ref": claimed_ref}
    if len(cn.split()) < 3:
        return base | {"verdict": "unverifiable", "color": "grey", "note": "النص قصير جدًا للتحقق منه."}
    if len(content_words(cn.split())) < MIN_CONTENT_WORDS:
        return base | {"verdict": "unverifiable", "color": "grey", "note": "النص عبارة عن صيغ متداولة في كثير من الأحاديث، فلا يمكن تحديد الحديث المقصود والتحقق منه."}
    matches = _hadeethenc_matches(cn)
    d, err = _dorar_matches(claim, cn)
    matches += d
    if not matches:
        note = "لم يُعثر على هذا الحديث في المصادر المعتمدة المتاحة (الدرر السنية / HadeethEnc). لا نستطيع الحكم عليه، فلا تنسبه إلى النبي ﷺ قبل التثبت من أهل الحديث."
        if err:
            note += " (تعذّر الاتصال بالدرر السنية مؤقتًا.)"
        return base | {"verdict": "not_found", "color": "grey", "matches": [], "note": note, "source_error": err}

    # best wording: exact beats near; on equal similarity the curated source (HadeethEnc) wins
    matches.sort(key=lambda m: (m["exact"], m["sim"], m["source"] == "hadeethenc", m["score"]), reverse=True)
    best = matches[0]
    if not best["exact"]:
        # HadeethEnc carries the hadith's OVERALL grade; Dorar entries are remarks on single narrations. So when the curated
        # wording is about as close as the best one, it is the one we show and grade from.
        close = [m for m in matches if m["source"] == "hadeethenc" and m["sim"] >= best["sim"] - SIM_TIE]
        if close:
            best = close[0]
    if not best["exact"]:
        # The claim does not reproduce any known wording exactly: show the closest authentic wording and what differs.
        same = [m for m in matches if m["window_norm"] == best["window_norm"]] or [best]
        g_verdict, g_color, g_note = _grade_verdict(same, curated_min_score=0)
        if best["source"] != "hadeethenc" and g_verdict in ("weak", "mixed", "graded"):
            # only Dorar remarks on single narrations exist for this wording: say that, do not call the hadith weak
            g_verdict, g_note = "graded", "انظر أدناه أحكام العلماء على روايات هذا اللفظ (وهي ملاحظات على أسانيد بعينها وليست حكمًا عامًا على الحديث)."
        correct = {"text_ar": " ".join(best["window_show"]).strip(" «»\"“”'.،,;:"), "source": best["source"], "id": best["id"], "grade": best["grade"],
                   "grade_class": best["grade_class"], "grader": best["grader"], "book": best["book"], "reference": best["reference"],
                   "reference_url": best["reference_url"], "similarity": best["sim"],
                   "diff": _diff([_stem(w) for w in cn.split()], best["window_stem"], best["window_show"], claim_show=cn.split())}
        note = (f"اللفظ المنقول يختلف عن اللفظ الوارد في المصادر؛ الصواب: «{correct['text_ar']}». " + g_note)
        color = "red" if g_verdict == "fabricated" else "amber"
        return base | {"verdict": "misquoted", "color": color, "grade_verdict": g_verdict, "correct": correct, "diff": correct["diff"],
                       "matches": same + [m for m in matches if m not in same][:4], "note": note, "source_error": err}

    exact = [m for m in matches if m["exact"]]
    verdict, color, note = _grade_verdict(exact if any(m["source"] == "hadeethenc" for m in exact) else matches)
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
