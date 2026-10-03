"""The eval runner itself: one ask per case with the full trace stored, and --reverify at zero cost."""
import json

import pytest

from core import pipeline
from evals import run_evals as E

ROUTE_A = '{"level": "أ", "intent": "ask", "language": "ar", "term": null}'
Q = "ما هي أركان الإسلام؟"


def good(fake_llm):
    from core import retrieve as R
    h = next(p for p in R.retrieve(Q, per_type=pipeline.PER_TYPE_ASK) if p["type"] == "hadith")
    return f"أركان الإسلام خمسة كما بيّن النبي ﷺ في الحديث الآتي. [[{h['id']}]]" + chr(10) * 2 + "{{" + h["id"][len("hadith:"):].join(["hadith:", ""]) + "}}"


def test_run_one_routes_once_and_stores_the_trace(fake_llm):
    ans = good(fake_llm)
    f = fake_llm(router=[ROUTE_A], generate=[ans])
    g = {"id": "t1", "input": Q, "expected_level": "أ", "expected_intent": "ask", "expected_behavior": "answer", "must_retrieve_title": "بني الإسلام على خمس"}
    r = E.run_one(g, live=True, do_judge=False)
    assert len([c for c in f.calls if c[0] == pipeline.L.ROUTER_MODEL]) == 1       # not routed twice any more
    assert r["router_ok"] and r["behavior_ok"] and r["retrieval_ok"] and r["attempts"] == 1
    assert r["trace"]["attempts"][0]["raw"] == ans and r["trace"]["retrieved_ids"] and r["abstain_reason"] is None
    assert r["_resp"]["referrals"] == [] and "language" in r["_resp"]


def test_failed_attempts_are_kept_with_their_errors(fake_llm):
    bad = "هذا هو الجواب الكامل عن السؤال المطروح {{quran:2:255}} [[qa:bayyinat:424242]]"
    fake_llm(router=[ROUTE_A], generate=[bad, bad])
    g = {"id": "t2", "input": Q, "expected_level": "أ", "expected_intent": "ask", "expected_behavior": "answer"}
    r = E.run_one(g, live=True, do_judge=False)
    assert r["abstain_reason"] == "verification_failed" and r["attempts"] == 2
    assert all(a["errors"] for a in r["trace"]["attempts"])
    s = E.summarize([r])
    assert s["verifier_forced_abstention_rate"][0] == 100.0 and s["false_abstention_rate"][0] == 100.0


def test_reverify_replays_saved_outputs_with_the_current_verifier(fake_llm, tmp_path, capsys):
    fake_llm(router=[ROUTE_A], generate=["معنى لا يستند الى مصدر ولا يحتوي اي احالة صحيحة على المصادر المسترجعة ابدا في هذا النص"] * 2)
    g = {"id": "t3", "input": Q, "expected_level": "أ", "expected_intent": "ask", "expected_behavior": "answer"}
    r = E.run_one(g, live=True, do_judge=False)
    # pretend the first attempt had been recorded as a pass under an older, laxer verifier
    r["trace"]["attempts"][0]["ok"] = True
    p = tmp_path / "results.json"
    p.write_text(json.dumps({"results": [r]}, ensure_ascii=False), encoding="utf-8")
    E.reverify(str(p))
    out = capsys.readouterr().out
    assert "re-verified" in out and "attempt 1: pass -> reject" in out and "t3" in out
