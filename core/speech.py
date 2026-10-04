"""Speech-to-text for "ask by voice" (Whisper, local). Audio is processed in memory and never written to disk or logged.

The browser sends 16 kHz mono PCM16 (it decodes/resamples itself, so the server needs no ffmpeg). The transcript is only
put in the question box for the user to review and edit; it is never submitted automatically."""
import os
import threading
from functools import lru_cache

import numpy as np

MODEL_NAME = os.environ.get("WHISPER_MODEL", "openai/whisper-base")  # whisper-tiny is faster but noticeably worse at Arabic
SAMPLE_RATE = 16000
MAX_SECONDS = 30  # one Whisper window
MIN_SECONDS = 0.4
_LANG = {"ar": "arabic", "en": "english"}
_LOCK = threading.Lock()


class AudioError(ValueError):
    pass


@lru_cache(maxsize=1)
def _pipe():
    import truststore
    truststore.inject_into_ssl()
    from transformers import pipeline
    return pipeline("automatic-speech-recognition", model=MODEL_NAME, device="cpu")


def pcm16_to_float(data: bytes) -> np.ndarray:
    if len(data) < 2:
        raise AudioError("empty audio")
    arr = np.frombuffer(data[: len(data) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
    secs = len(arr) / SAMPLE_RATE
    if secs > MAX_SECONDS + 1:
        raise AudioError(f"audio longer than {MAX_SECONDS}s")
    if secs < MIN_SECONDS:
        raise AudioError("audio too short")
    return arr[: MAX_SECONDS * SAMPLE_RATE]


def transcribe(data: bytes, language: str | None = None) -> str:
    """language 'ar'/'en' pins Whisper's language (it mis-detects short Arabic clips); None lets it detect."""
    audio = pcm16_to_float(data)
    if float(np.abs(audio).max()) < 0.005:
        raise AudioError("silent audio")
    kwargs = {"task": "transcribe"}
    if language in _LANG:
        kwargs["language"] = _LANG[language]
    with _LOCK:  # one inference at a time, same reasoning as core.embedding
        out = _pipe()({"raw": audio, "sampling_rate": SAMPLE_RATE}, generate_kwargs=kwargs)
    return " ".join(out["text"].split())
