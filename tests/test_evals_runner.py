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


def test_token_usage_is_attributed_per_thread_and_totals_are_exact():
    """Parallel evals: each case must see only its own tokens, and the global counter must not lose updates."""
    import threading

    from core import llm as L
    L.USAGE.clear()
    seen = {}

    class U:
        prompt_tokens, completion_tokens = 10, 3

    def work(k):
        L.thread_usage_reset()
        for _ in range(200):
            L._count("m", U, "prompt_tokens", "completion_tokens")
        seen[k] = L.thread_usage()["m"]

    ts = [threading.Thread(target=work, args=(k,)) for k in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert all(v == {"calls": 200, "input": 2000, "output": 600} for v in seen.values())
    assert L.USAGE["m"] == {"calls": 1600, "input": 16000, "output": 4800}


def test_retrieval_connection_is_per_thread():
    import threading

    from core import retrieve as R
    errs, ids = [], []

    def work():
        try:
            ids.append(R.get_passage("quran:2:255")["id"])
        except Exception as e:  # sqlite3.ProgrammingError if a connection leaked across threads
            errs.append(e)

    ts = [threading.Thread(target=work) for _ in range(6)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errs and ids == ["quran:2:255"] * 6


def test_run_one_works_in_parallel_threads(fake_llm):
    from concurrent.futures import ThreadPoolExecutor
    fake_llm(router=['{"level": "د", "intent": "ask", "language": "ar"}'])
    gs = [{"id": f"p{i}", "input": "طلقت زوجتي ثلاث مرات في لحظة غضب، فهل وقع الطلاق؟", "expected_level": "د",
           "expected_intent": "ask", "expected_behavior": "referral"} for i in range(6)]
    with ThreadPoolExecutor(3) as ex:
        rs = list(ex.map(lambda g: E.run_one(g, live=True, do_judge=False), gs))
    assert all(r["behavior_ok"] and r["router_ok"] for r in rs)


def _res(i, status="answered", errors=(), source="llm", behavior="answer", judge=None):
    return {"id": i, "status": status, "verification_errors": list(errors), "trace": {"attempts": [], "retrieved_ids": []},
            "route": {"level": "أ", "intent": "ask", "source": source}, "expected": {"expected_behavior": behavior}, "judge": judge}


def test_is_tainted_detects_provider_outages_judge_failures_and_router_fallback():
    out_of_credit = "Error code: 429 - {'error': {'message': 'You have no credits remaining.'}}"
    assert E.is_tainted(_res("a", "abstained", [out_of_credit]))
    assert E.is_tainted(_res("b", "abstained", ["Error code: 503 - overloaded"]))
    assert E.is_tainted(_res("c", judge={"error": "429"}))
    assert E.is_tainted(_res("d", source="heuristic"))                      # model router failed, rules stood in
    assert E.is_tainted(_res("e", "error"))
    assert not E.is_tainted(_res("f"))
    assert not E.is_tainted(_res("g", source="heuristic", behavior="referral"))   # level-د / term / verify never need the model router
    assert not E.is_tainted(_res("h", "abstained", ["Explanation without an explicit [[passage-id]] citation"]))  # a real verdict, keep it


def test_resume_reruns_only_tainted_cases_and_merges_in_golden_order(tmp_path, monkeypatch, fake_llm):
    import json as _json
    fake_llm()
    args = type("A", (), {"only": None, "limit": 4, "workers": 2, "no_judge": True})()
    ids = [g["id"] for g in E.load_golden(args)]
    results = {i: _res(i) for i in ids}
    results[ids[1]] = _res(ids[1], "abstained", ["Error code: 429 - no credits remaining"])
    results[ids[3]] = _res(ids[3], source="heuristic")
    old = tmp_path / "results-old.json"
    old.write_text(_json.dumps({"meta": "LIVE run", "results": [results[i] for i in ids]}, ensure_ascii=False), encoding="utf-8")
    ran = []

    def fake_run_cases(todo, live, do_judge, workers):
        ran.extend(g["id"] for g in todo)
        return [dict(_res(g["id"]), marker="rerun") for g in todo]

    monkeypatch.setattr(E, "run_cases", fake_run_cases)
    monkeypatch.setattr(E, "REPORTS", tmp_path)
    monkeypatch.setattr(E, "summarize", lambda rs: {"x": (1, len(rs))})
    monkeypatch.setattr(E, "failure_patterns", lambda rs: {})
    monkeypatch.setattr(E, "html_report", lambda *a: "<html></html>")
    E.resume(str(old), args)
    assert ran == [ids[1], ids[3]]                                       # only the tainted ones were re-run
    out = _json.loads(next(tmp_path.glob("results-2*.json")).read_text(encoding="utf-8"))
    assert [r["id"] for r in out["results"]] == ids                      # golden order preserved
    assert [r.get("marker") for r in out["results"]] == [None, "rerun", None, "rerun"]
    assert "RESUMED" in out["meta"] and "re-ran 2" in out["meta"]


def test_abstain_or_clarify_accepts_abstention_or_a_scripture_free_answer_and_rejects_invented_hadith():
    g = {"expected_behavior": "abstain_or_clarify"}
    assert E.behavior_ok(g, {"status": "abstained", "blocks": []})
    assert E.behavior_ok(g, {"status": "answered", "blocks": [{"kind": "explanation", "text": "أرسل لي الكلام المقصود"}]})
    assert not E.behavior_ok(g, {"status": "answered", "blocks": [{"kind": "hadith", "text_ar": "..."}]})
    assert not E.behavior_ok(g, {"status": "answered", "blocks": [{"kind": "quran", "text_ar": "..."}]})
