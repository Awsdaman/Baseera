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
