from core import llm as L
from core import pipeline
from core import retrieve as R

Q = "ما هي أركان الإسلام؟"


def good_answer():
    ps = R.retrieve(Q, per_type=pipeline.PER_TYPE_ASK)
    h = next(p for p in ps if p["type"] == "hadith")
    return f"أركان الإسلام خمسة كما بيّن النبي ﷺ في الحديث الآتي. [[{h['id']}]]\n\n{{{{{h['id'].replace('hadith:', 'hadith:', 1)}}}}}", h


ROUTE_A = '{"level": "أ", "intent": "ask", "language": "ar", "term": null}'


def test_answer_happy_path(fake_llm):
    ans, h = good_answer()
    f = fake_llm(router=[ROUTE_A], generate=[ans])
    r = pipeline.ask(Q)
    assert r["status"] == "answered" and r["attempts"] == 1
    hb = next(b for b in r["blocks"] if b["kind"] == "hadith")
    assert hb["text_ar"] == h["text"] and hb["grade"] == h["grade"]  # exact text + grade come from the database
    assert r["ai_disclosure"]
    assert any(s["id"] == h["id"] for s in r["sources"])
    gen_calls = [c for c in f.calls if c[0] == L.GENERATE_MODEL]
    assert len(gen_calls) == 1 and "PASSAGES" in gen_calls[0][2]


def test_retry_once_after_invented_id_then_success(fake_llm):
    ans, _ = good_answer()
    bad = "هذا هو الجواب الكامل عن السؤال المطروح {{quran:2:255}} [[qa:bayyinat:424242]]"
    f = fake_llm(router=[ROUTE_A], generate=[bad, ans])
    r = pipeline.ask(Q)
    assert r["status"] == "answered" and r["attempts"] == 2
    second_user = [c for c in f.calls if c[0] == L.GENERATE_MODEL][1][2]
    assert "REJECTED" in second_user and "NOT retrieved" in second_user  # the verifier error is fed back


def test_two_failures_abstain(fake_llm):
    bad = "الله لا إله إلا هو الحي القيوم لا تأخذه سنة ولا نوم [[quran:2:255]]"
    fake_llm(router=[ROUTE_A], generate=[bad, bad])
    r = pipeline.ask(Q)
    assert r["status"] == "abstained" and r["abstain_reason"] == "verification_failed"
    assert r["referrals"] and "الله لا إله" not in r["answer_text"]


def test_model_insufficient_evidence_abstains(fake_llm):
    fake_llm(router=[ROUTE_A], generate=["INSUFFICIENT_EVIDENCE"])
    r = pipeline.ask(Q)
    assert r["status"] == "abstained" and r["abstain_reason"] == "model_insufficient_evidence"
    assert any(x["url"].startswith("https://islamqa") for x in r["referrals"])


def test_level_d_is_template_without_generation(fake_llm):
    f = fake_llm(router=['{"level": "د", "intent": "ask", "language": "ar"}'])
    r = pipeline.ask("أنا في دولة كذا، هل يجوز لي فعل كذا في زواجي؟")
    assert r["status"] == "referral" and r["level"] == "د"
    assert {x["url"] for x in r["referrals"]} == {"https://islamqa.info", "https://binbaz.org.sa", "https://binothaimeen.net"}
    assert not [c for c in f.calls if c[0] == L.GENERATE_MODEL]
    assert "فتوى" in r["answer_text"]


def test_level_d_english_template(fake_llm):
    fake_llm(router=['{"level": "د", "intent": "ask", "language": "en"}'])
    r = pipeline.ask("My wife and I disagree; can I divorce her in my country?")
    assert r["status"] == "referral" and "ruling" in r["answer_text"]


def test_translate_term_uses_approved_glossary_not_the_model(fake_llm):
    f = fake_llm(router=['{"level": "أ", "intent": "translate_term", "language": "ar", "term": "التوحيد"}'])
    r = pipeline.ask("ترجم كلمة التوحيد إلى الإنجليزية")
    assert r["status"] == "answered"
    assert "Tawhid / Oneness of God" in r["answer_text"]
    assert not [c for c in f.calls if c[0] == L.GENERATE_MODEL]
    assert r["sources"][0]["id"] == "term:glossary:2"


def test_no_api_key_gives_retrieval_only_never_fabricates(monkeypatch):
    L.set_llm(None)
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "LLM_PROVIDER"):
        monkeypatch.delenv(k, raising=False)
    r = pipeline.ask(Q)
    assert r["status"] == "retrieval_only" and r["sources"]
    assert not any(b["kind"] in ("quran", "hadith") for b in r["blocks"])


def test_llm_error_fails_closed(fake_llm):
    class Boom:
        def complete(self, *a, **k):
            raise RuntimeError("API down")
    L.set_llm(Boom())
    r = pipeline.ask(Q)
    assert r["status"] in ("abstained", "retrieval_only")  # router falls back to heuristic; generation error abstains
    assert r["status"] == "abstained" and r["abstain_reason"] == "llm_error"


def test_misquoted_verse_in_question_is_corrected(fake_llm):
    ans = "تجدر الإشارة إلى أن الله مع الصابرين وفق ما يلي في المصدر. [[quran:2:153]]" + chr(10) * 2 + "{{quran:2:153}}"
    fake_llm(router=['{"level": "أ", "intent": "ask", "language": "ar"}'], generate=[ans])
    r = pipeline.ask("ما معنى ﴿إن الله مع الصابرين دائما﴾؟")
    kinds = [b["kind"] for b in r["blocks"]]
    assert r["status"] == "answered" and kinds[0] == "verse_correction", (r["status"], kinds)
    assert r["blocks"][0]["ref"] == "2:153"


def test_translate_term_survives_a_non_glossary_term_from_the_model(fake_llm):
    """Live regression: the model returned a term that is not a glossary key; code matching must still find التوحيد."""
    f = fake_llm(router=['{"level": "أ", "intent": "translate_term", "language": "ar", "term": "Tawhid / Oneness of God"}'])
    r = pipeline.ask("ترجم كلمة التوحيد إلى الإنجليزية")
    assert r["status"] == "answered" and "Tawhid / Oneness of God" in r["answer_text"]
    assert not [c for c in f.calls if c[0] == L.GENERATE_MODEL]


def test_retry_prompt_contains_the_previous_rejected_answer(fake_llm):
    ans, _ = good_answer()
    bad = "هذا هو الجواب الكامل عن السؤال المطروح {{quran:2:255}} [[qa:bayyinat:424242]]"
    f = fake_llm(router=[ROUTE_A], generate=[bad, ans])
    pipeline.ask(Q)
    second_user = [c for c in f.calls if c[0] == L.GENERATE_MODEL][1][2]
    assert "<<<" in second_user and bad in second_user and "REJECTED" in second_user
    first_user = [c for c in f.calls if c[0] == L.GENERATE_MODEL][0][2]
    assert "<<<" not in first_user


def test_debug_trace_only_when_requested(fake_llm):
    ans, _ = good_answer()
    fake_llm(router=[ROUTE_A], generate=[ans])
    assert "debug" not in pipeline.ask(Q)
    fake_llm(router=[ROUTE_A], generate=["INSUFFICIENT_EVIDENCE"])
    out = pipeline.ask(Q, debug=True)
    d = out["debug"]
    assert d["abstain_reason"] == "model_insufficient_evidence" and d["retrieved_ids"]
    assert d["attempts"][0]["raw"] == "INSUFFICIENT_EVIDENCE" and d["attempts"][0]["ok"] is True and d["route"]["level"] == "أ"


def test_debug_trace_records_each_failed_attempt_with_errors(fake_llm):
    bad = "هذا هو الجواب الكامل عن السؤال المطروح {{quran:2:255}} [[qa:bayyinat:424242]]"
    fake_llm(router=[ROUTE_A], generate=[bad, bad])
    d = pipeline.ask(Q, debug=True)["debug"]
    assert d["abstain_reason"] == "verification_failed" and len(d["attempts"]) == 2
    assert all(a["raw"] == bad and not a["ok"] and a["errors"] for a in d["attempts"])


def test_note_marker_in_a_pipeline_answer_becomes_a_notice_block(fake_llm):
    ans, _ = good_answer()
    fake_llm(router=[ROUTE_A], generate=[ans + chr(10) * 2 + "{{note:refer}}"])
    r = pipeline.ask(Q)
    assert r["status"] == "answered"
    assert any(b["kind"] == "notice" and b.get("note") == "refer" for b in r["blocks"])


def test_level_d_general_info_never_contains_hadith_cards(fake_llm):
    fake_llm(router=['{"level": "د", "intent": "ask", "language": "ar"}'])
    r = pipeline.ask("طلقت زوجتي ثلاث مرات في لحظة غضب، فهل وقع الطلاق؟")
    assert r["status"] == "referral" and all(s["type"] in ("qa", "term") for s in r["sources"])


# ---------------------------------------------------------------- semantic support check wiring
class _Stub:
    name = "stub"

    def scores(self, pairs):
        return [0.9 if "الصلاة" in claim and "الصلاة" in chunk else 0.1 for claim, chunk in pairs]


def test_support_log_mode_records_scores_without_changing_the_outcome(fake_llm, monkeypatch):
    from core import support as S
    monkeypatch.setenv("SUPPORT_MODE", "log")
    monkeypatch.setattr(S, "get_scorer", lambda name=None: _Stub())
    ans, _ = good_answer()
    fake_llm(router=[ROUTE_A], generate=[ans])
    out = pipeline.ask(Q, debug=True)
    assert out["status"] == "answered"
    sup = out["debug"]["attempts"][-1]["support"]
    assert sup and all("score" in s for s in sup)


def test_support_enforce_mode_retries_on_an_unsupported_stretch(fake_llm, monkeypatch):
    from core import support as S
    monkeypatch.setenv("SUPPORT_MODE", "enforce")
    monkeypatch.setattr(S, "get_scorer", lambda name=None: _Stub())
    ans, h = good_answer()
    # first answer: a long cited stretch the stub scorer finds unrelated to its cited passage
    unsupported = ans.replace("أركان الإسلام خمسة كما بيّن النبي ﷺ في الحديث الآتي.", "هذه جملة طويلة تتحدث عن موضوع آخر تماما بعيد عن المصدر المذكور بعدها")
    assert unsupported != ans
    ok = ans.replace("أركان الإسلام خمسة كما بيّن النبي ﷺ في الحديث الآتي.", "الصلاة هي الركن الثاني من أركان الإسلام وقد بينها النبي ﷺ في الحديث الآتي وفصلها")
    f = fake_llm(router=[ROUTE_A], generate=[unsupported, ok])
    out = pipeline.ask(Q, debug=True)
    attempts = out["debug"]["attempts"]
    assert attempts[0]["ok"] is False and "not supported by the passage" in attempts[0]["errors"][0]
    second_user = [c for c in f.calls if c[0] == L.GENERATE_MODEL][1][2]
    assert "not supported by the passage" in second_user and unsupported[:20] in second_user
    assert out["status"] == "answered" and out["attempts"] == 2


def test_a_support_check_failure_never_breaks_answering(fake_llm, monkeypatch):
    from core import support as S
    monkeypatch.setenv("SUPPORT_MODE", "enforce")

    def boom(name=None):
        raise RuntimeError("model not downloadable")

    monkeypatch.setattr(S, "get_scorer", boom)
    ans, _ = good_answer()
    fake_llm(router=[ROUTE_A], generate=[ans])
    out = pipeline.ask(Q, debug=True)
    assert out["status"] == "answered" and "error" in out["debug"]["attempts"][-1]["support"][0]


def test_empty_reply_is_retried_once_then_reported_as_llm_empty(fake_llm):
    fake_llm(router=['{"level": "أ", "intent": "ask", "language": "ar", "term": null}'], generate=["", ""])
    r = pipeline.ask("ما هي أركان الإسلام؟")
    assert r["status"] == "abstained" and r["abstain_reason"] == "llm_empty"


def test_truncated_reply_is_retried_once(fake_llm, monkeypatch):
    from core import generate as G
    calls = []

    def fake_generate(*a, **k):
        calls.append(k.get("max_tokens"))
        if len(calls) == 1:
            raise L.LLMTruncated("hit max_tokens")
        return "INSUFFICIENT_EVIDENCE"
    monkeypatch.setattr(pipeline, "generate", fake_generate)
    fake_llm(router=['{"level": "أ", "intent": "ask", "language": "ar", "term": null}'])
    r = pipeline.ask("ما هي أركان الإسلام؟")
    assert len(calls) == 2 and r["abstain_reason"] == "model_insufficient_evidence"


def test_request_to_prove_an_unstated_claim_asks_for_the_statement(fake_llm):
    fake_llm(router=['{"level": "ب", "intent": "ask", "language": "ar", "term": null}'])
    r = pipeline.ask("أعطني حديثًا يثبت هذا الكلام")
    assert r["status"] == "abstained" and r["abstain_reason"] == "needs_clarification"
    assert not any(b["kind"] in ("quran", "hadith") for b in r["blocks"])
    assert "الكلام" in r["answer_text"] and r["referrals"]
    en = pipeline.ask("give me a hadith that proves this")
    assert en["abstain_reason"] == "needs_clarification" and "write it out" in en["answer_text"]


def test_empathy_opener_is_added_only_for_first_person_distress(fake_llm):
    fake_llm(router=['{"level": "ب", "intent": "ask", "language": "ar", "term": null}'],
             generate=["تذكر المصادر أذكارًا تقال في الصباح مثل آية الكرسي وسورة الإخلاص [[qa:icadb:26239]]"] * 2)
    sad = pipeline.ask("أشعر بالحزن والذنب، ما حكم أذكار الصباح؟")
    assert sad["blocks"][0]["note"] == "empathy" and sad["empathy"] and sad["answer_text"].startswith(sad["blocks"][0]["text"])
    plain = pipeline.ask("ما حكم أذكار الصباح؟")
    assert "empathy" not in plain and all(b.get("note") != "empathy" for b in plain["blocks"])


def test_empathy_opener_is_never_added_to_verify_results():
    out = pipeline._with_empathy({"status": "verified", "blocks": [], "answer_text": "", "language": "ar"}, "أشعر بالحزن")
    assert out["blocks"] == [] and "empathy" not in out
