from fastapi.testclient import TestClient

import api.main as m
from core import pipeline

client = TestClient(m.app)


def test_ask_never_exposes_internal_fields(monkeypatch):
    seen = {}

    def fake_ask(q, lang=None, debug=False):
        seen["debug"] = debug
        return {"status": "abstained", "verification_errors": ["secret model text"], "debug": {"attempts": ["raw"]}, "blocks": []}

    monkeypatch.setattr(pipeline, "ask", fake_ask)
    r = client.post("/api/ask", json={"question": "سؤال"})
    assert r.status_code == 200 and seen["debug"] is False
    assert "verification_errors" not in r.json() and "debug" not in r.json()


def test_health_reports_llm_and_data():
    j = client.get("/api/health").json()
    assert j["ok"] and j["verses"] == 6236 and "provider" in j["llm"] and "configured" in j["llm"]


def test_ask_validates_input():
    assert client.post("/api/ask", json={"question": ""}).status_code == 422
    assert client.post("/api/ask", json={"question": "x" * 2001}).status_code == 422
    assert client.post("/api/verify", json={"text": ""}).status_code == 422


def test_ui_pages_are_never_served_from_a_stale_browser_cache():
    for path in ("/", "/retrieval"):
        r = client.get(path)
        assert r.status_code == 200 and r.headers["cache-control"] == "no-cache"


def test_rate_limit_stops_a_flood_of_asks_but_not_normal_use(monkeypatch):
    import api.main as m
    from fastapi.testclient import TestClient
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "3")
    m._hits.clear()
    c = TestClient(m.app)
    codes = [c.post("/api/verify", json={"text": "قال تعالى: ﴿قل هو الله أحد﴾"}).status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200] and codes[3] == 429
    m._hits.clear()


def test_oversized_report_outcome_and_audio_are_refused():
    import api.main as m
    from fastapi.testclient import TestClient
    m._hits.clear()
    c = TestClient(m.app)
    big = {"question": "سؤال", "reason": "other", "consent": True, "outcome": {"x": "a" * 30000}}
    assert c.post("/api/report", json=big).status_code == 413
    assert c.post("/api/transcribe", content=b"0" * (16000 * 2 * 40)).status_code == 413
    m._hits.clear()


def test_health_hides_internal_addresses_and_reports_readiness():
    import api.main as m
    from fastapi.testclient import TestClient
    j = TestClient(m.app).get("/api/health").json()
    assert "base_url" not in str(j) and "database_empty" in j and "vectors_ready" in j


def _wait_job(jid, tries=100):
    import time
    for _ in range(tries):
        j = client.get(f"/api/jobs/{jid}").json()
        if j["state"] in ("done", "error"):
            return j
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_job_lifecycle_hides_internal_fields_and_unknown_job_is_404(monkeypatch):
    monkeypatch.setattr(pipeline, "ask", lambda q, lang=None, debug=False: {"status": "answered", "debug": {"x": 1}, "verification_errors": ["secret"], "blocks": []})
    r = client.post("/api/jobs", json={"kind": "ask", "text": "سؤال"})
    assert r.status_code == 200 and r.json()["position"] == 0
    j = _wait_job(r.json()["job"])
    assert j["state"] == "done" and j["result"]["status"] == "answered" and "debug" not in j["result"] and "verification_errors" not in j["result"]
    assert client.get("/api/jobs/doesnotexist").status_code == 404


def test_job_failure_is_reported_without_the_text(monkeypatch):
    def boom(q, lang=None, debug=False):
        raise RuntimeError("secret question text")

    monkeypatch.setattr(pipeline, "ask", boom)
    j = _wait_job(client.post("/api/jobs", json={"kind": "ask", "text": "سؤال"}).json()["job"])
    assert j == {"state": "error", "error": "failed"}


def test_job_queue_refuses_when_full_and_validates(monkeypatch):
    import threading
    gate = threading.Event()
    monkeypatch.setattr(pipeline, "ask", lambda q, lang=None, debug=False: (gate.wait(5), {"status": "answered", "blocks": []})[1])
    monkeypatch.setenv("MAX_QUEUE", "1")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "0")
    m._jobs.clear()
    codes = [client.post("/api/jobs", json={"kind": "ask", "text": f"q{i}"}).status_code for i in range(m._capacity() + 2)]
    gate.set()
    assert codes[:-1] == [200] * (len(codes) - 1) and codes[-1] == 503
    assert client.post("/api/jobs", json={"kind": "other", "text": "x"}).status_code == 422
    assert client.post("/api/jobs", json={"kind": "ask", "text": "x" * 2001}).status_code == 422


def test_rate_limit_uses_the_real_client_address_behind_the_tunnel(monkeypatch):
    monkeypatch.setenv("TRUST_CF_IP", "1")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "2")
    m._hits.clear()
    monkeypatch.setattr(pipeline, "ask", lambda q, lang=None, debug=False: {"status": "answered", "blocks": []})
    post = lambda ip: client.post("/api/ask", json={"question": "س"}, headers={"cf-connecting-ip": ip}).status_code
    assert [post("1.1.1.1"), post("1.1.1.1"), post("1.1.1.1")] == [200, 200, 429]
    assert post("2.2.2.2") == 200                    # another judge is not affected


def test_usage_stats_count_without_storing_text_and_are_operator_only(monkeypatch):
    from core import usage_stats
    usage_stats.reset()
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "0")
    monkeypatch.setattr(pipeline, "ask", lambda q, lang=None, debug=False: {"status": "abstained", "blocks": []})
    client.post("/api/ask", json={"question": "نص سري"})
    snap = usage_stats.snapshot()
    assert snap["questions"] == 1 and snap["outcomes"] == {"abstained": 1} and "سري" not in str(snap)
    assert client.get("/api/stats").status_code in (200, 404)             # TestClient host is not 127.0.0.1
    assert client.get("/api/stats", headers={"cf-connecting-ip": "1.2.3.4"}).status_code == 404   # never through the tunnel
