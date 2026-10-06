"""ASK pipeline: route -> (term | referral | verify | retrieve -> generate -> verify -> retry once -> abstain)."""
import os

from core import llm as L
from core import retrieve as R
from core import relevance as rel
from core import rewrites
from core import support as S
from core.generate import generate
from core.glossary import GLOSSARY, lookup
from core.router import detect_language, find_term, needs_context, needs_empathy, needs_medical_note, route
from core.verify import NOTES, VerifyResult, normalize_citations, verify_answer

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
EXTRA_FROM_ENGLISH = 4  # passages from the English-wording search kept next to the Arabic-search results
EXTRA_FROM_ORIGINAL = 3  # extra passages taken from a search with the user's own wording
PER_TYPE_ASK = {"quran": 4, "hadith": 4, "qa": 10, "tafsir": 2, "term": 2}  # qa: 4 curated + 3 fiqh encyclopedia + 2 books + 1 reference card (core/retrieve.py QA_GROUPS)

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
    "empathy": {
        "ar": "أتفهّم أن ما تمرّ به قد يكون صعبًا، وأشكرك على مشاركته معي. سأعرض لك فيما يلي ما ورد في المصادر المعتمدة بلطف ووضوح.",
        "en": "I understand that what you are going through may be difficult, and I thank you for sharing it. Below is what the approved sources say, offered gently and clearly.",
    },
    "service": {
        "ar": "تعذّر الوصول إلى خدمة توليد الإجابة مؤقتًا (النموذج غير متاح أو لم يُضبط). هذا لا يعني أن المصادر لا تحتوي على جواب؛ يرجى المحاولة بعد قليل أو الرجوع إلى المراجع الموثوقة أدناه.",
        "en": "The answer-generation service could not be reached right now (the model is unavailable or not configured). This does not mean the sources have no answer; please try again shortly or use the trusted references below.",
    },
    "clarify": {
        "ar": "لم أفهم ما الذي تريد مني أن أُثبته؛ فلم تذكر نصّ الكلام أو الحكم المقصود. يمكنك كتابة العبارة أو الحكم الذي تسأل عنه، وسأبحث له في المصادر المعتمدة. ولن أذكر لك حديثًا أو آية لا صلة لها بما تقصده.",
        "en": "I could not tell what you want me to prove: the statement or ruling you mean was not included. Please write it out and I will look for it in the approved sources. I will not show you a hadith or verse that has no clear connection to what you mean.",
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
            # HadeethEnc's terms forbid deleting part of a hadith: hadith cards carry the full text; other long passages are excerpts
            "text": (p.get("text") or "") if p["type"] == "hadith" else (p.get("text") or "")[:400],
            "text_en": (p.get("text_en") or "") if p["type"] == "hadith" else ((p.get("text_en") or "")[:300] or None)}


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


SERVICE_REASONS = ("llm_error", "llm_empty", "llm_truncated")


def abstain_response(info, lang, reason="insufficient_evidence", sources=None, errors=None, text_key="abstain"):
    if reason in SERVICE_REASONS:  # an outage must not read as "the sources do not cover this"
        text_key = "service"
    msg = TEXT[text_key]["en" if lang == "en" else "ar"]
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


def _with_empathy(out: dict, question: str) -> dict:
    """Acknowledge the person first when they share a struggle. A code-owned notice (fixed text, never model-written, never a citation)."""
    if out.get("status") not in ("answered", "abstained", "referral") or not needs_empathy(question):
        return out
    lang = "en" if out.get("language") == "en" else "ar"
    text = TEXT["empathy"][lang]
    out["blocks"] = [{"kind": "notice", "text": text, "note": "empathy"}] + list(out.get("blocks") or [])
    out["answer_text"] = text + chr(10) * 2 + (out.get("answer_text") or "")
    out["empathy"] = True
    return out


def _with_medical(out: dict, question: str) -> dict:
    """Health questions: a code-owned reminder that this is general information, not medical advice. Added once, never to verify results."""
    if out.get("status") not in ("answered", "abstained", "referral") or not needs_medical_note(question):
        return out
    if any(b.get("note") == "medical" for b in out.get("blocks") or []):
        return out
    lang = "en" if out.get("language") == "en" else "ar"
    note = NOTES["medical"][lang]
    out["blocks"] = list(out.get("blocks") or []) + [{"kind": "notice", "text": note, "note": "medical"}]
    out["answer_text"] = (out.get("answer_text") or "") + chr(10) * 2 + note
    return out


def ask(question: str, lang: str | None = None, debug: bool = False) -> dict:
    """`debug=True` (evals/dev only, never exposed by the API) adds out["debug"]: route, retrieved ids, every generation
    attempt with its raw model output and verifier errors, and the abstention reason."""
    trace = {"retrieved_ids": [], "attempts": []}
    out = _ask(question, lang, trace)
    out = _with_empathy(out, question)
    out = _with_medical(out, question)
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

    rw = rewrites.lookup(question)  # a human-approved rewrite of this exact phrasing beats the model's restatement
    if rw:
        info["canonical_question"], info["rewrite"] = rw["canonical"], "approved"
        if rw.get("claim"):
            info["claim"] = rw["claim"]
        if info["intent"] == "verify" and not rw.get("keep_verify"):
            info["intent"] = "ask"

    if info["intent"] == "ask" and needs_context(question):
        # "give me a hadith that proves this" with nothing to prove: ask for the statement instead of searching and showing unrelated texts
        return abstain_response(info, lang, "needs_clarification", text_key="clarify")
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
        # a result made only of unidentifiable quotes ("not found" for a quoted phrase that is not scripture) verified nothing
        meaningful = [c for c in out["claims"] if not (c["claim_type"] == "quote" and c["verdict"] in ("not_found", "unverifiable"))]
        if meaningful:
            return _resp("verified", info, lang, verify=out, answer_text="", blocks=[])
        # Nothing to verify (no verse, hadith or attributed saying): this is a question about a topic, e.g. 'is this ruling
        # correct?', not a pasted message. Answer it like any question instead of dead-ending with "nothing found".
        info["intent"], info["fallback"] = "ask", "verify_found_no_claims"
    if info["level"] == "د":
        return personal_response(question, info, lang)
    if info.get("claim") and info["level"] == "أ":
        info["level"] = "ب"  # judging whether a statement/ruling is right is explanation-level at least, never "stable basics"

    search_q = info.get("canonical_question") or question  # the model's clean restatement of what is asked
    if search_q != question:
        info["original"] = question
    trace["canonical_question"] = search_q
    fix_blocks, fix_ids = _corrections(question, lang)
    passages = R.retrieve(search_q, per_type=PER_TYPE_ASK)
    ar_q = info.get("search_ar") if lang != "ar" else None
    if ar_q:  # the sources and the keyword index are Arabic: a non-Arabic question is also searched in Arabic (answer language is unchanged)
        trace["search_ar"] = ar_q
        ar_ps = R.retrieve(ar_q, per_type=PER_TYPE_ASK)
        have = {p["id"] for p in ar_ps}
        passages = ar_ps + [p for p in passages if p["id"] not in have][:EXTRA_FROM_ENGLISH]
    if search_q != question:  # the user's own wording can find sources the restatement misses: add a few new ones
        have = {p["id"] for p in passages}
        passages += [p for p in R.retrieve(question, per_type=PER_TYPE_ASK) if p["id"] not in have][:EXTRA_FROM_ORIGINAL]
    have = {p["id"] for p in passages}
    for pid in fix_ids:
        if pid not in have and R.get_passage(pid):
            passages.insert(0, R.get_passage(pid))
    gloss = find_term(question)  # an approved glossary term in the user's own wording always brings its glossary card, even if the restatement lost it
    if gloss:
        gid = f"term:glossary:{[g['ar'] for g in GLOSSARY].index(gloss) + 1}"
        if gid not in {p["id"] for p in passages} and R.get_passage(gid):
            passages.insert(0, R.get_passage(gid))
    trace["retrieved_ids"] = [p["id"] for p in passages]
    resp = _answer(search_q, info, lang, passages, trace)
    if fix_blocks:
        resp["blocks"] = fix_blocks + resp["blocks"]
        resp["corrections"] = fix_blocks
    return resp


class _Empty(Exception):
    pass


def _generate_once(question, info, lang, passages, errors, previous):
    """One generation with a single RUNTIME retry (not a verifier attempt): a truncated or empty reply (a local model that spent its
    budget on hidden reasoning) or a dropped connection is retried once, with a bigger budget after a truncation."""
    import time
    kw = dict(error=errors, previous=previous, claim=info.get("claim"), original=info.get("original"))
    budget = int(os.environ.get("LLM_GEN_MAX_TOKENS", 6000))
    for retry in (False, True):
        try:
            raw = generate(question, info["level"], lang, passages, max_tokens=budget, **kw)
            if raw.strip():
                return raw
            err = _Empty()
        except L.LLMTruncated as e:
            err, budget = e, min(budget * 2, 3000 if budget <= 1500 else budget)
        except Exception as e:
            if type(e).__name__ not in ("APIConnectionError", "APITimeoutError"):
                raise
            err = e
            time.sleep(2)
        if retry:
            raise err
    raise err  # unreachable


def _answer(question, info, lang, passages, trace):
    if not passages:
        return abstain_response(info, lang, "no_passages")
    if not L.llm_available():
        cards = [_card(p, i + 1) for i, p in enumerate(passages[:8])]
        msg = TEXT["unavailable"]["en" if lang == "en" else "ar"]
        return _resp("retrieval_only", info, lang, blocks=[{"kind": "notice", "text": msg}], sources=cards, answer_text=msg)

    by_id = {p["id"]: p for p in passages}
    errors, last, raw = None, None, None
    max_attempts = int(os.environ.get("LLM_MAX_ATTEMPTS", 2))
    for attempts in range(1, max_attempts + 1):
        try:
            raw_model = _generate_once(question, info, lang, passages, errors, raw)
        except L.LLMTruncated as e:
            trace["attempts"].append({"n": attempts, "raw": None, "errors": [str(e)[:200]], "ok": False})
            return abstain_response(info, lang, "llm_truncated", errors=[str(e)[:200]])
        except _Empty:
            trace["attempts"].append({"n": attempts, "raw": "", "errors": ["empty answer"], "ok": False})
            return abstain_response(info, lang, "llm_empty", errors=["empty answer"])
        except Exception as e:  # API outage, rate limit, ...: fail closed with a warm abstain, never fabricate
            trace["attempts"].append({"n": attempts, "raw": None, "errors": [str(e)[:200]], "ok": False})
            return abstain_response(info, lang, "llm_error", errors=[str(e)[:200]])
        raw = normalize_citations(raw_model, by_id)  # citation SYNTAX only; every id is still verified below
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
        trace["attempts"].append({"n": attempts, "raw": raw_model, "normalized": raw != raw_model, "errors": vr.errors, "ok": vr.ok, "support": support})
        last = vr
        if vr.ok:
            if vr.abstain:
                return abstain_response(info, lang, "model_insufficient_evidence")
            blocks, answer_text = vr.blocks, vr.answer_text
            if rel.mode() != "off":
                # on-topic check: do the CITED passages contain what the question asks? (the verifier checks form, not relevance)
                cited = [by_id[x["id"]] for x in vr.sources if x["id"] in by_id]
                decision = rel.decide(question, cited)
                trace["attempts"][-1]["relevance"] = decision
                if rel.mode() == "enforce":
                    if decision["action"] == "abstain":
                        return abstain_response(info, lang, "sources_not_on_topic",
                                                errors=[f"cited passages do not address the question (model: {decision['verdict']}, similarity {decision.get('sim')})"])
                    if decision["action"] == "note" and not any(b.get("note") in ("no_ruling", "not_direct") for b in blocks):
                        note = NOTES["not_direct"]["en" if lang == "en" else "ar"]
                        blocks = list(blocks) + [{"kind": "notice", "text": note, "note": "not_direct"}]
                        answer_text = answer_text + chr(10) * 2 + note
            return _resp("answered", info, lang, blocks=blocks, sources=vr.sources, answer_text=answer_text, attempts=attempts)
        errors = list(vr.errors)
    return abstain_response(info, lang, "verification_failed", errors=last.errors if last else [])
