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
