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
