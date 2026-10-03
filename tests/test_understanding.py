"""Question understanding: canonical question + claim, verify->ask fallback, partial answers with the no_ruling note, Arabic purity."""
from core import llm as L
from core import pipeline
from core import retrieve as R
from core import router
from core.generate import build_prompts
from core.verify import NOTES, verify_answer

CLAIM_Q = 'هل هذا الحكم "حكم اذكار الصباح واجبه" صحيح؟'
PLAIN_Q = "ما حكم اذكار الصباح؟"
ROUTE_VERIFY = ('{"level": "أ", "intent": "verify", "language": "ar", "term": null, '
                '"canonical_question": "ما حكم أذكار الصباح، وهل هي واجبة؟", "claim": "حكم اذكار الصباح واجبه"}')
ROUTE_ASK = '{"level": "أ", "intent": "ask", "language": "ar", "term": null, "canonical_question": "ما حكم أذكار الصباح؟", "claim": null}'
KURSI_ID = "qa:icadb:26239"
ANSWER = ("تذكر المصادر أذكارًا تقال في الصباح والمساء مثل قراءة آية الكرسي وسورة الإخلاص ثلاث مرات في الصباح [[qa:icadb:26239]]"
          + chr(10) * 2 + "{{note:no_ruling}}")


# ---------------------------------------------------------------- router fields
def test_router_returns_canonical_question_and_claim(fake_llm):
    fake_llm(router=[ROUTE_VERIFY])
    r = router.route(CLAIM_Q)
    assert r["canonical_question"] == "ما حكم أذكار الصباح، وهل هي واجبة؟" and r["claim"] == "حكم اذكار الصباح واجبه"


def test_router_falls_back_to_a_code_extracted_claim_when_the_model_gives_none(fake_llm):
    fake_llm(router=['{"level": "ب", "intent": "ask", "language": "ar"}'])
    r = router.route(CLAIM_Q)
    assert r["claim"] == "حكم اذكار الصباح واجبه" and r["canonical_question"]


def test_heuristic_router_extracts_the_claim_without_any_model():
    r = router.route(CLAIM_Q)
    assert r["intent"] == "ask" and r["claim"] == "حكم اذكار الصباح واجبه" and r["canonical_question"] == "حكم اذكار الصباح واجبه؟"
    assert router.extract_claim(PLAIN_Q) is None and router.extract_claim("هل هذا الحديث صحيح؟") is None


def test_router_prompt_keeps_verify_for_pasted_scripture_only():
    assert "NOT verify" in router.SYSTEM and "canonical_question" in router.SYSTEM


# ---------------------------------------------------------------- pipeline
def test_verify_with_nothing_to_verify_becomes_an_ordinary_answer(fake_llm):
    f = fake_llm(router=[ROUTE_VERIFY], generate=[ANSWER])
    out = pipeline.ask(CLAIM_Q, debug=True)
    assert out["status"] == "answered" and out["route"]["intent"] == "ask" and out["route"]["fallback"] == "verify_found_no_claims"
    assert out["route"]["level"] == "ب"                      # judging a ruling is never 'stable basics'
    assert any(b["kind"] == "notice" and b.get("note") == "no_ruling" for b in out["blocks"])
    assert out["debug"]["canonical_question"] == "ما حكم أذكار الصباح، وهل هي واجبة؟"
    gen_user = [c for c in f.calls if c[0] == L.GENERATE_MODEL][0][2]
    assert "حكم اذكار الصباح واجبه" in gen_user and "WHETHER THIS STATEMENT IS CORRECT" in gen_user and CLAIM_Q in gen_user


def test_real_verify_requests_still_go_to_verify_mode(fake_llm, monkeypatch):
    from core import verifier_mode as V
    fake_llm(router=[ROUTE_VERIFY])
    monkeypatch.setattr(V, "verify_text", lambda t: {"claims": [{"claim_type": "verse", "verdict": "verified", "color": "green"}], "summary": {}})
    assert pipeline.ask("قال تعالى: ﴿إن الله مع الصابرين﴾ تحقق")["status"] == "verified"


def test_retrieval_uses_the_canonical_question_and_adds_a_few_from_the_original(fake_llm, monkeypatch):
    seen = []
    orig = R.retrieve
    monkeypatch.setattr(R, "retrieve", lambda q, per_type=None, vectors=False, dorar=False: (seen.append(q), orig(q, per_type=per_type, vectors=False))[1])
    fake_llm(router=[ROUTE_ASK.replace("ما حكم أذكار الصباح؟", "ما الحكم في تلاوة أذكار الصباح؟")], generate=[ANSWER])
    pipeline.ask(PLAIN_Q, debug=True)
    assert seen[0] == "ما الحكم في تلاوة أذكار الصباح؟" and PLAIN_Q in seen


def test_partial_answer_instead_of_refusal_when_passages_are_relevant_but_do_not_state_the_ruling(fake_llm):
    fake_llm(router=[ROUTE_ASK], generate=[ANSWER])
    out = pipeline.ask(PLAIN_Q)
    assert out["status"] == "answered" and any(s["id"] == KURSI_ID for s in out["sources"])
    assert any(b.get("note") == "no_ruling" for b in out["blocks"])


def test_generator_prompt_allows_partial_answers_and_names_the_note():
    system, user = build_prompts(PLAIN_Q, "ب", "ar", [R.get_passage(KURSI_ID)])
    assert "do NOT refuse" in system and "{{note:no_ruling}}" in system and "INSUFFICIENT_EVIDENCE" in system
    assert "ORIGINAL WORDING" not in user and "STATEMENT IS CORRECT" not in user
    _, user2 = build_prompts("سؤال منقح", "ب", "ar", [R.get_passage(KURSI_ID)], claim="حكم X واجب", original="سؤال أصلي")
    assert "ORIGINAL WORDING" in user2 and "سؤال أصلي" in user2 and "حكم X واجب" in user2


# ---------------------------------------------------------------- verifier
def test_no_ruling_note_renders_in_both_languages_and_is_not_a_citation():
    ps = [R.get_passage(KURSI_ID)]
    ok = verify_answer(ANSWER, ps)
    assert ok.ok, ok.errors
    assert ok.blocks[-1]["text"] == NOTES["no_ruling"]["ar"]
    assert verify_answer("{{note:no_ruling}}", ps).ok is False                      # a note alone is still no source
    en = verify_answer("The sources describe the adhkar to say in the morning, such as Ayat al-Kursi and the Ikhlas surah. [[qa:icadb:26239]]"
                       + chr(10) * 2 + "{{note:no_ruling}}", ps, "en")
    assert en.ok and en.blocks[-1]["text"] == NOTES["no_ruling"]["en"]


def test_english_words_inside_an_arabic_answer_are_rejected_but_parenthesised_terms_are_fine():
    ps = [R.get_passage(KURSI_ID)]
    bad = verify_answer("لا يثبت من هذا passage حكم الوجوب في الأذكار المذكورة في الصباح [[qa:icadb:26239]]", ps)
    assert not bad.ok and any("English word" in e for e in bad.errors)
    good = verify_answer("التوحيد (Tawhid) هو إفراد الله بالعبادة وهو أصل الأذكار المذكورة في الصباح [[qa:icadb:26239]]", ps)
    assert good.ok, good.errors
    en = verify_answer("The sources describe the adhkar to say in the morning, such as Ayat al-Kursi. [[qa:icadb:26239]]", ps, "en")
    assert en.ok, en.errors
