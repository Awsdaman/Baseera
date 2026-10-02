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
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
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
    ans = "تجدر الإشارة إلى أن الله مع الصابرين وفق ما يلي في المصدر. {{quran:2:153}}"
    fake_llm(router=['{"level": "أ", "intent": "ask", "language": "ar"}'], generate=[ans])
    r = pipeline.ask("ما معنى ﴿إن الله مع الصابرين دائما﴾؟")
    kinds = [b["kind"] for b in r["blocks"]]
    assert kinds[0] == "verse_correction", kinds
    assert r["blocks"][0]["ref"] == "2:153"
