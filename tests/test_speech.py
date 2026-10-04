import numpy as np
import pytest
from fastapi.testclient import TestClient

import api.main as m
from core import speech

client = TestClient(m.app)


def pcm(seconds, amp=8000):
    t = np.arange(int(seconds * speech.SAMPLE_RATE)) / speech.SAMPLE_RATE
    return (np.sin(2 * np.pi * 220 * t) * amp).astype("<i2").tobytes()


def test_rejects_bad_audio():
    for data in (b"", pcm(0.1), pcm(40), bytes(32000)):
        with pytest.raises(speech.AudioError):
            speech.transcribe(data)


def test_transcribe_endpoint_passes_language_and_returns_text(monkeypatch):
    seen = {}
    monkeypatch.setattr(speech, "transcribe", lambda d, lang=None: seen.update(n=len(d), lang=lang) or "ما هي أركان الإسلام؟")
    r = client.post("/api/transcribe?language=ar", content=pcm(2))
    assert r.status_code == 200 and r.json() == {"text": "ما هي أركان الإسلام؟"} and seen["lang"] == "ar"


def test_transcribe_endpoint_errors(monkeypatch):
    assert client.post("/api/transcribe", content=b"").status_code == 400
    assert client.post("/api/transcribe?language=fr", content=pcm(2)).status_code == 422
    assert client.post("/api/transcribe", content=pcm(45)).status_code == 413
    monkeypatch.setattr(speech, "transcribe", lambda d, lang=None: 1 / 0)
    assert client.post("/api/transcribe", content=pcm(2)).status_code == 503
