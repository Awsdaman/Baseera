import json
from pathlib import Path

from core.dorar import parse

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "samples" / "dorar_search.json"


def test_parse_sample():
    html = json.loads(SAMPLE.read_text(encoding="utf-8"))["ahadith"]["result"]
    hits = parse(html)
    assert len(hits) >= 10
    h = hits[0]
    assert h["source"] == "dorar" and h["id"].startswith("hadith:dorar:")
    assert "الأعمال" in h["text"] and not h["text"].startswith("1")
    assert h["grader"] and h["book"] and h["grade"]
    assert h["narrator"]
    assert len({x["id"] for x in hits}) == len(hits)


def test_parse_empty():
    assert parse("") == []


def test_dorar_circuit_breaker_skips_the_network_after_one_failure(monkeypatch):
    import time
    from core import dorar

    calls = []

    def boom(*a, **k):
        calls.append(1)
        raise RuntimeError("403 Cloudflare")
    monkeypatch.setattr(dorar, "get_json", boom)
    monkeypatch.setattr(dorar, "_down_until", 0.0)
    monkeypatch.setattr(dorar, "_cached", lambda params: False)
    for _ in range(2):
        try:
            dorar.search("اتقوا النار ولو بشق تمرة")
        except RuntimeError:
            pass
    assert len(calls) == 1 and dorar._down_until > time.time()          # the second call never reached the network
