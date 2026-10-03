"""Opt-in reports (privacy), approved rewrites and the reviewer tool."""
import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

import api.main as m
from core import llm as L
from core import pipeline, reports, rewrites
from tools import review

client = TestClient(m.app)
OUTCOME = {"status": "abstained", "level": "ب", "intent": "ask", "language": "ar", "abstain_reason": "model_insufficient_evidence",
           "answer_text": "لم أجد ...", "source_ids": ["qa:icadb:26239"]}


@pytest.fixture(autouse=True)
def isolated_files(tmp_path, monkeypatch):
    monkeypatch.setenv("REPORTS_DB", str(tmp_path / "reports.sqlite"))
    monkeypatch.setattr(review, "GOLDEN", tmp_path / "golden.jsonl")
    review.GOLDEN.write_text(json.dumps({"id": "x-1", "input": "q"}) + chr(10), encoding="utf-8")
    monkeypatch.setattr(review, "GAPS", tmp_path / "gaps.json")
    monkeypatch.setattr(rewrites, "PATH", tmp_path / "rewrites.json")
    rewrites._cache["mtime"] = None


# ---------------------------------------------------------------- privacy
def test_nothing_is_stored_without_explicit_consent():
    for consent in (False, None, "yes", 1):
        with pytest.raises(ValueError):
            reports.add_report("سؤال", "wrong_answer", None, OUTCOME, consent)
    assert reports.list_reports(None) == []


def test_api_refuses_a_report_without_consent_and_stores_with_it():
    body = {"question": "ما حكم اذكار الصباح؟", "reason": "should_have_answered", "comment": "x", "outcome": OUTCOME}
    assert client.post("/api/report", json=body | {"consent": False}).status_code == 400
    assert client.post("/api/report", json=body).status_code == 400                      # consent defaults to False
    assert reports.list_reports(None) == []
    assert client.post("/api/report", json=body | {"consent": True}).json() == {"ok": True}
    assert len(reports.list_reports(None)) == 1


def test_only_the_declared_fields_are_stored():
    reports.add_report("سؤال", "other", "ملاحظة", OUTCOME, True)
    cols = {r[1] for r in sqlite3.connect(reports.db_path()).execute("PRAGMA table_info(reports)")}
    assert cols == {"id", "created", "question", "language", "status", "level", "intent", "abstain_reason", "answer_excerpt", "source_ids",
                    "reason", "comment", "review", "review_note"}                    # no ip, user agent, session or account columns


def test_emails_phones_and_links_are_scrubbed_before_saving():
    rid = reports.add_report("اتصل بي على 0501234567 أو a.b@mail.com انظر https://x.com/page", "other",
                             "رقمي +966 50 123 4567 وبريدي me@site.org", OUTCOME, True)
    r = reports.get_report(rid)
    blob = r["question"] + r["comment"]
    for secret in ("0501234567", "a.b@mail.com", "https://x.com", "me@site.org", "123 4567"):
        assert secret not in blob
    assert "[email]" in blob and "[number]" in blob and "[link]" in blob


def test_unknown_reason_and_empty_question_are_rejected():
    with pytest.raises(ValueError):
        reports.add_report("سؤال", "made-up-reason", None, OUTCOME, True)
    with pytest.raises(ValueError):
        reports.add_report("   ", "other", None, OUTCOME, True)


def test_retention_purges_old_reports_only():
    old = reports.add_report("قديم", "other", None, OUTCOME, True)
    new = reports.add_report("جديد", "other", None, OUTCOME, True)
    con = reports.connect()
    con.execute("UPDATE reports SET created='2020-01-01 00:00' WHERE id=?", (old,))
    con.commit()
    assert reports.purge(90) == 1
    assert [r["id"] for r in reports.list_reports(None)] == [new]


def test_privacy_endpoint_states_the_policy():
    j = client.get("/api/privacy").json()
    assert j["retention_days"] == 90 and "IP address" in j["never_stores"] and "ar" in j["policy"] and "en" in j["policy"]


# ---------------------------------------------------------------- approved rewrites
def test_rewrite_matches_near_exact_phrasings_only():
    rewrites.add('هل هذا الحكم "حكم اذكار الصباح واجبه" صحيح؟', "ما حكم أذكار الصباح، وهل هي واجبة؟", claim="حكم اذكار الصباح واجبه")
    assert rewrites.lookup('هل هذا الحكم "حكم أذكار الصباح واجبة" صحيح؟')["canonical"] == "ما حكم أذكار الصباح، وهل هي واجبة؟"   # hamza / ta marbuta
    assert rewrites.lookup("ما حكم الصلاة؟") is None and rewrites.lookup("") is None


def test_approved_rewrite_is_used_by_the_pipeline_before_search(fake_llm, monkeypatch):
    from core import retrieve as R
    seen = []
    orig = R.retrieve
    monkeypatch.setattr(R, "retrieve", lambda q, per_type=None, vectors=False, dorar=False: (seen.append(q), orig(q, per_type=per_type, vectors=False))[1])
    rewrites.add("سؤال غريب الصياغة عن الاذكار", "ما حكم أذكار الصباح؟")
    fake_llm(router=['{"level": "ب", "intent": "verify", "language": "ar", "term": null, "canonical_question": "شيء آخر", "claim": null}'],
             generate=["تذكر المصادر أذكارًا تقال في الصباح مثل آية الكرسي وسورة الإخلاص [[qa:icadb:26239]]"])
    out = pipeline.ask("سؤال غريب الصياغة عن الاذكار", debug=True)
    assert out["route"]["rewrite"] == "approved" and out["route"]["intent"] == "ask"      # also overrides a wrong 'verify' routing
    assert out["debug"]["canonical_question"] == "ما حكم أذكار الصباح؟" and seen[0] == "ما حكم أذكار الصباح؟"


# ---------------------------------------------------------------- reviewer tool
def _report(**kw):
    return reports.add_report(kw.get("q", "ما حكم اذكار الصباح؟"), kw.get("reason", "should_have_answered"), "تعليق", OUTCOME, True)


def test_approve_test_appends_a_valid_golden_case_and_marks_the_report():
    rid = _report()
    case = review.approve_test(rid, "ب", "answer", retrieve=["qa:icadb:26239"])
    lines = [json.loads(x) for x in review.GOLDEN.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert lines[-1] == case and case["id"] == f"r-{rid}" and case["must_retrieve_any"] == ["qa:icadb:26239"]
    assert {"input", "expected_level", "expected_intent", "expected_behavior"} <= set(case)
    assert reports.get_report(rid)["review"] == "approved_test"
    with pytest.raises(SystemExit):
        review.approve_test(rid, "ب")                                                      # no duplicates
    with pytest.raises(SystemExit):
        review.approve_test(_report(q="س2"), "ز")                                          # invalid level


def test_approve_rewrite_makes_the_phrasing_active():
    rid = _report(q="هل هذا صحيح ان الاذكار فرض؟")
    review.approve_rewrite(rid, "ما حكم أذكار الصباح، وهل هي فرض؟")
    assert rewrites.lookup("هل هذا صحيح ان الاذكار فرض؟")["canonical"].startswith("ما حكم أذكار")
    assert reports.get_report(rid)["review"] == "approved_rewrite"


def test_source_gap_dismiss_summary_and_purge():
    a, b = _report(q="سؤال أ"), _report(q="سؤال ب", reason="wrong_source")
    g = review.gap(a, topic="حكم أذكار الصباح (الوجوب أو الاستحباب)")
    assert json.loads(review.GAPS.read_text(encoding="utf-8"))[0]["topic"] == g["topic"] and reports.get_report(a)["review"] == "source_gap"
    review.main(["dismiss", str(b), "--note", "ليست مشكلة"])
    assert reports.get_report(b)["review"] == "dismissed"
    s = review.summary()
    assert "2 reports (0 pending)" in s and "source gaps recorded: 1" in s
    assert reports.list_reports("pending") == []
