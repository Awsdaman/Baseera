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


# ---- model choice, preprocessing and backend wiring (fake backend: no model download)
class FakeModel:
    def __init__(self, reply="  قل هو الله   أحد "):
        self.calls, self.reply = [], reply

    def __call__(self, audio, language):
        self.calls.append((audio, language))
        return self.reply


def test_default_model_is_the_quran_hadith_whisper(monkeypatch):
    for k in ("SPEECH_PROVIDER", "WHISPER_MODEL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(speech, "PROVIDER", "whisper")
    assert speech.provider() == "whisper" and speech.model_name() == "Dr-AliGomaa/whisper-large-v3-ar"
    monkeypatch.setenv("WHISPER_MODEL", "openai/whisper-base")
    assert speech.model_name() == "openai/whisper-base"
    monkeypatch.setenv("SPEECH_PROVIDER", "cohere")
    assert speech.model_name() == "CohereLabs/cohere-transcribe-arabic-07-2026"


def test_transcribe_prepares_audio_and_cleans_text(monkeypatch):
    fake = FakeModel()
    monkeypatch.setattr(speech, "_backend", lambda: fake)
    quiet = np.concatenate([np.zeros(speech.SAMPLE_RATE, dtype="<i2"),
                            np.frombuffer(pcm(1, amp=800), dtype="<i2"), np.zeros(speech.SAMPLE_RATE, dtype="<i2")])
    assert speech.transcribe(quiet.tobytes(), "ar") == "قل هو الله أحد"
    audio, lang = fake.calls[0]
    assert lang == "ar" and audio.dtype == np.float32
    assert 1.0 <= len(audio) / speech.SAMPLE_RATE <= 1.5  # 2 s of silence trimmed (+ 100 ms pads + margin)
    assert abs(20 * np.log10(np.sqrt(np.mean(audio ** 2))) + 16) < 1.5  # loudness normalised to about -16 dBFS


def test_silence_never_reaches_the_model(monkeypatch):
    fake = FakeModel()
    monkeypatch.setattr(speech, "_backend", lambda: fake)
    with pytest.raises(speech.AudioError):
        speech.transcribe(bytes(32000))
    assert not fake.calls


def test_health_reports_the_speech_model():
    s = client.get("/api/health").json()["speech"]
    assert s["model"] and s["max_seconds"] == speech.MAX_SECONDS and s["loaded"] in (True, False)
