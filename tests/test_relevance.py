"""On-topic check: real, correctly cited passages that do not address the question must not become a confident answer."""
from core import pipeline, relevance
from core import llm as L

ANSWER = "تذكر المصادر أذكارًا تقال في الصباح مثل آية الكرسي وسورة الإخلاص [[qa:icadb:26239]]"
ROUTE = '{"level": "ب", "intent": "ask", "language": "ar", "term": null}'


def _ask(fake_llm, monkeypatch, decision):
    monkeypatch.setenv("REL_MODE", "enforce")
    monkeypatch.setattr(relevance, "decide", lambda q, ps: decision)
    fake_llm(router=[ROUTE], generate=[ANSWER])
    return pipeline.ask("ما حكم أذكار الصباح؟", debug=True)


def test_answers_verdict_changes_nothing(fake_llm, monkeypatch):
    r = _ask(fake_llm, monkeypatch, {"verdict": "answers", "action": "keep"})
    assert r["status"] == "answered" and all(b.get("note") != "not_direct" for b in r["blocks"])
    assert r["debug"]["attempts"][-1]["relevance"]["action"] == "keep"


def test_partial_adds_the_code_owned_not_direct_notice_once(fake_llm, monkeypatch):
    r = _ask(fake_llm, monkeypatch, {"verdict": "partial", "action": "note"})
    notes = [b for b in r["blocks"] if b.get("note") == "not_direct"]
    assert r["status"] == "answered" and len(notes) == 1 and r["answer_text"].endswith(notes[0]["text"])


def test_unrelated_with_low_similarity_abstains_instead_of_showing_the_answer(fake_llm, monkeypatch):
    r = _ask(fake_llm, monkeypatch, {"verdict": "unrelated", "sim": 0.31, "action": "abstain"})
    assert r["status"] == "abstained" and r["abstain_reason"] == "sources_not_on_topic" and r["referrals"]
    assert not any(b["kind"] == "explanation" for b in r["blocks"])


def test_off_mode_skips_the_check(fake_llm, monkeypatch):
    monkeypatch.setenv("REL_MODE", "off")
    monkeypatch.setattr(relevance, "decide", lambda *a: (_ for _ in ()).throw(AssertionError("must not run")))
    fake_llm(router=[ROUTE], generate=[ANSWER])
    assert pipeline.ask("ما حكم أذكار الصباح؟")["status"] == "answered"


def test_a_failing_check_never_breaks_answering(fake_llm):
    class Boom:
        def complete(self, *a, **k):
            raise RuntimeError("server down")
    L.set_llm(Boom())
    out = relevance.check("سؤال", [{"id": "qa:x:1", "type": "qa", "text": "نص"}])
    assert out["verdict"] == "answers" and "server down" in out["error"]


def test_decide_abstains_only_when_both_signals_agree(monkeypatch):
    monkeypatch.setattr(relevance, "check", lambda q, ps: {"verdict": "unrelated"})
    monkeypatch.setattr(relevance, "question_similarity", lambda q, ps: 0.40)
    assert relevance.decide("q", [{"id": "a"}])["action"] == "abstain"
    monkeypatch.setattr(relevance, "question_similarity", lambda q, ps: 0.60)
    assert relevance.decide("q", [{"id": "a"}])["action"] == "note"        # similar enough: keep the answer with a note
    monkeypatch.setattr(relevance, "check", lambda q, ps: {"verdict": "partial"})
    assert relevance.decide("q", [{"id": "a"}])["action"] == "note"
    monkeypatch.setattr(relevance, "check", lambda q, ps: {"verdict": "answers"})
    assert relevance.decide("q", [{"id": "a"}])["action"] == "keep"
