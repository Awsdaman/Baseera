"""ASK pipeline: route -> (term | referral | verify | retrieve -> generate -> verify -> retry once -> abstain)."""
from core import llm as L
from core import retrieve as R
from core import support as S
from core.generate import generate
from core.glossary import GLOSSARY, lookup
from core.router import detect_language, find_term, route
from core.verify import VerifyResult, verify_answer

DISCLOSURE = {
    "ar": "بصيرة أداة مدعومة بالذكاء الاصطناعي وليست عالمًا ولا مفتيًا؛ تجيب من مصادر معتمدة وتعرض مراجعها. لا نخزّن أي بيانات شخصية.",
    "en": "Baseera is an AI-powered tool, not a scholar or mufti. It answers only from approved sources and shows its references. No personal data is stored.",
}
REFERRALS = [
    {"name": "IslamQA — الإسلام سؤال وجواب", "url": "https://islamqa.info"},
    {"name": "موقع الشيخ ابن باز", "url": "https://binbaz.org.sa"},
    {"name": "موقع الشيخ ابن عثيمين", "url": "https://binothaimeen.net"},
]
MIN_GENERAL_SCORE = 0.026  # general-info cards must be found by BOTH keyword and vector search (RRF of two top-10 ranks)
PER_TYPE_ASK = {"quran": 4, "hadith": 4, "qa": 4, "tafsir": 2, "term": 2}

TEXT = {
    "personal": {
        "ar": ("يبدو أن سؤالك يتعلق بحالة شخصية خاصة، والحكم على الوقائع الفردية يحتاج إلى معرفة التفاصيل وتقديرٍ شرعي متخصص، "
               "ولا تستطيع بصيرة إصدار فتوى أو حكم في حالتك. ما نقدّمه هنا معلومات عامة فقط، ونوصيك بسؤال عالم أو جهة إفتاء مؤهلة "
               "في بلدك، وبذكر تفاصيل حالتك كاملة."),
        "en": ("Your question looks like a personal situation. Judging an individual case needs the full facts and qualified scholarly "
               "assessment, so Baseera cannot give you a ruling (fatwa) for your case. Below is general information only; please ask a "
               "qualified scholar or a recognized fatwa authority in your country and give them the full details."),
        "general": {"ar": "معلومات عامة من المصادر المعتمدة (وليست حكمًا في حالتك):", "en": "General information from approved sources (not a ruling on your case):"},
        "refer": {"ar": "للاستفتاء الشخصي يمكنك الرجوع إلى:", "en": "For a personal ruling you can consult:"},
    },
    "abstain": {
        "ar": ("لم أجد في المصادر المعتمدة المتاحة لي ما يكفي للإجابة عن سؤالك بثقة، وأفضّل ألا أُخمّن أو أنسب إلى المصادر ما ليس فيها. "
               "يمكنك إعادة صياغة السؤال بشكل أوضح، أو الرجوع إلى المواقع الموثوقة التالية، أو سؤال أهل العلم."),
        "en": ("I could not find enough in the approved sources available to me to answer this confidently, and I prefer not to guess or "
               "attribute to the sources what is not in them. You can rephrase the question more specifically, check the trusted sites below, "
               "or ask a qualified scholar."),
    },
    "unavailable": {
        "ar": "خدمة توليد الإجابة غير متاحة حاليًا (لم يتم ضبط مفتاح النموذج)، وهذه أقرب المصادر المعتمدة لسؤالك:",
        "en": "The answer-generation service is not configured (no model key), so here are the closest approved sources for your question:",
    },
    "term": {
        "ar": "الترجمة المعتمدة لمصطلح «{ar}» هي: {en}.",
        "en": "The approved translation of the term “{ar}” is: {en}.",
    },
}


def _resp(status, route_info, lang, **kw):
    base = {"status": status, "level": route_info.get("level"), "intent": route_info.get("intent"), "language": lang,
            "blocks": [], "sources": [], "answer_text": "", "referrals": [], "ai_disclosure": DISCLOSURE.get(lang, DISCLOSURE["ar"]),
            "route": route_info}
    base.update(kw)
    return base


def _card(p, n):
    return {"n": n, "id": p["id"], "type": p["type"], "source": p["source"], "title": p.get("title"),
            "reference_url": p.get("reference_url"), "grade": p.get("grade"),
            "text": (p.get("text") or "")[:400], "text_en": (p.get("text_en") or "")[:300] or None}


def term_response(info, lang):
    g = lookup(info.get("term") or "")
    if not g:
        return None
    msg = TEXT["term"][lang if lang == "en" else "ar"].format(ar=g["ar"], en=g["en"])
    note = g["note"]
    idx = next(i for i, x in enumerate(GLOSSARY, 1) if x["ar"] == g["ar"])
    src = {"n": 1, "id": f"term:glossary:{idx}", "type": "term", "source": "glossary", "title": g["ar"],
           "reference_url": "docs/data.pdf#page=7", "grade": None}
    blocks = [{"kind": "notice", "text": msg + " [1]"},
              {"kind": "glossary", "term_ar": g["ar"], "term_en": g["en"], "usage_note_ar": note}]
    return _resp("answered", info, lang, blocks=blocks, sources=[src], answer_text=msg + "\n" + note)


def personal_response(question, info, lang):
    t = TEXT["personal"]
    general = [p for p in R.retrieve(question, per_type={"qa": 3, "term": 1})  # no hadith cards: weak matches read as a ruling
               if p["type"] in ("qa", "term") and p["score"] >= MIN_GENERAL_SCORE][:2]
    cards = [_card(p, i + 1) for i, p in enumerate(general)]
    blocks = [{"kind": "notice", "text": t["en"] if lang == "en" else t["ar"]}]
    if cards:
        blocks.append({"kind": "general_info", "label": t["general"]["en" if lang == "en" else "ar"], "cards": cards})
    return _resp("referral", info, lang, blocks=blocks, sources=cards, referrals=REFERRALS,
                 answer_text=blocks[0]["text"], refer_label=t["refer"]["en" if lang == "en" else "ar"])


def abstain_response(info, lang, reason="insufficient_evidence", sources=None, errors=None):
    msg = TEXT["abstain"]["en" if lang == "en" else "ar"]
    return _resp("abstained", info, lang, blocks=[{"kind": "notice", "text": msg}], referrals=REFERRALS,
                 answer_text=msg, abstain_reason=reason, verification_errors=errors or [], sources=sources or [])


CORRECTION = {
    "misquoted": {"ar": "تنبيه لطيف: الآية التي ذكرتَها وردت بصياغة تختلف قليلًا عن نص المصحف. هذا هو النص الصحيح، وسأبني الإجابة عليه:",
                  "en": "A gentle note: the verse you quoted differs slightly from the Mushaf text. Here is the correct text, and I will base the answer on it:"},
    "fabricated": {"ar": "تنبيه لطيف: لم أجد النص الذي ذكرتَه في المصحف بهذا اللفظ، فلن أبني الإجابة عليه.",
                   "en": "A gentle note: I could not find the text you quoted in the Mushaf in that wording, so I will not build the answer on it."},
}


def _corrections(question: str, lang: str):
    """Verses quoted inside the question are checked against the Mushaf; wrong ones are corrected, never relied on."""
    from core.verifier_mode import find_embedded_verses
    blocks, ids = [], []
    for v in find_embedded_verses(question):
        if v["verdict"] not in ("misquoted", "fabricated"):
            continue
        b = {"kind": "verse_correction", "verdict": v["verdict"], "claimed": v["claim"],
             "message": CORRECTION[v["verdict"]]["en" if lang == "en" else "ar"]}
        if v.get("match"):
            b |= {"ref": v["match"]["ref"], "surah_name_ar": v["match"]["surah_name_ar"], "text_ar": v["match"]["text_ar"],
                  "diff": v.get("diff"), "reference_url": v["match"]["reference_url"]}
            ids.append(v["match"]["id"])
        blocks.append(b)
    return blocks, ids


def ask(question: str, lang: str | None = None, debug: bool = False) -> dict:
    """`debug=True` (evals/dev only, never exposed by the API) adds out["debug"]: route, retrieved ids, every generation
    attempt with its raw model output and verifier errors, and the abstention reason."""
    trace = {"retrieved_ids": [], "attempts": []}
    out = _ask(question, lang, trace)
    if debug:
        out["debug"] = trace | {"route": out.get("route"), "abstain_reason": out.get("abstain_reason")}
    return out


def _ask(question: str, lang: str | None = None, trace: dict | None = None) -> dict:
    trace = trace if trace is not None else {"retrieved_ids": [], "attempts": []}
    question = (question or "").strip()
    lang = lang if lang in ("ar", "en") else detect_language(question)
    info = route(question)
    info["language"] = lang if lang in ("ar", "en") else info.get("language", "ar")
    lang = info["language"] if info["language"] in ("ar", "en") else "ar"

    if info["intent"] == "translate_term":
        # the model's extracted term may not be a glossary key ("Tawhid / Oneness of God"): fall back to code matching
        info["term"] = info.get("term") if lookup(info.get("term") or "") else find_term(question)
        r = term_response(info, lang)
        if r:
            return r
        info["intent"] = "ask"
    if info["intent"] == "verify":
        from core.verifier_mode import verify_text
        out = verify_text(question)
        return _resp("verified", info, lang, verify=out, answer_text="", blocks=[])
    if info["level"] == "د":
        return personal_response(question, info, lang)

    fix_blocks, fix_ids = _corrections(question, lang)
    passages = R.retrieve(question, per_type=PER_TYPE_ASK)
    have = {p["id"] for p in passages}
    for pid in fix_ids:
        if pid not in have and R.get_passage(pid):
            passages.insert(0, R.get_passage(pid))
    trace["retrieved_ids"] = [p["id"] for p in passages]
    resp = _answer(question, info, lang, passages, trace)
    if fix_blocks:
        resp["blocks"] = fix_blocks + resp["blocks"]
        resp["corrections"] = fix_blocks
    return resp


def _answer(question, info, lang, passages, trace):
    if not passages:
        return abstain_response(info, lang, "no_passages")
    if not L.llm_available():
        cards = [_card(p, i + 1) for i, p in enumerate(passages[:8])]
        msg = TEXT["unavailable"]["en" if lang == "en" else "ar"]
        return _resp("retrieval_only", info, lang, blocks=[{"kind": "notice", "text": msg}], sources=cards, answer_text=msg)

    errors, attempts, last, raw = None, 0, None, None
    for attempts in (1, 2):
        try:
            raw = generate(question, info["level"], lang, passages, error=errors, previous=raw)
        except Exception as e:  # API outage, rate limit, ...: fail closed with a warm abstain, never fabricate
            trace["attempts"].append({"n": attempts, "raw": None, "errors": [str(e)[:200]], "ok": False})
            return abstain_response(info, lang, "llm_error", errors=[str(e)[:200]])
        vr = verify_answer(raw, passages, lang)
        support = None
        if vr.ok and not vr.abstain and S.mode() != "off":
            # semantic check that each cited stretch is backed by the passage(s) it cites (see core/support.py)
            try:
                s_errs, support = S.unsupported(raw, passages)
                if S.mode() == "enforce" and s_errs:
                    vr = VerifyResult(ok=False, errors=s_errs)
            except Exception as e:  # the check is an extra safety net: never let a model-loading problem break answering
                support = [{"error": str(e)[:120]}]
        trace["attempts"].append({"n": attempts, "raw": raw, "errors": vr.errors, "ok": vr.ok, "support": support})
        last = vr
        if vr.ok:
            if vr.abstain:
                return abstain_response(info, lang, "model_insufficient_evidence")
            return _resp("answered", info, lang, blocks=vr.blocks, sources=vr.sources, answer_text=vr.answer_text, attempts=attempts)
        errors = "\n".join(f"- {e}" for e in vr.errors)
    return abstain_response(info, lang, "verification_failed", errors=last.errors if last else [])
