"""Speech-to-text for "ask by voice" (open Arabic models, local). Audio is processed in memory and never written to disk or logged.

The browser sends 16 kHz mono PCM16 (it decodes/resamples itself, so the server needs no ffmpeg). The transcript is only
put in the question box for the user to review and edit; it is never submitted automatically.

Models (see docs/VOICE_HANDOFF.md for the comparison and the reasons):
    SPEECH_PROVIDER=whisper (default) + WHISPER_MODEL (default Dr-AliGomaa/whisper-large-v3-ar): Whisper large-v3 fine-tuned on
        Quran recitation, hadith and MSA (reported WER 0.33% Quran, 3.6% hadith, 9.5% MSA). Writes Imla'i text without tashkeel,
        the form our verse/hadith matchers compare. Weak on dialects. Apache-2.0. Any Hugging Face Whisper id works
        (openai/whisper-base is the old, much weaker default: fine for a quick CPU demo).
    SPEECH_PROVIDER=cohere: CohereLabs/cohere-transcribe-arabic-07-2026 (2B, Arabic + English, stronger on dialects, no published
        Quran numbers). Apache-2.0 but GATED: accept its terms on Hugging Face and set HF_TOKEN. Needs transformers >= 5.4.

    SPEECH_DEVICE   auto (CUDA if PyTorch sees a GPU, else CPU) | cpu | cuda
    SPEECH_BEAMS    beam size (default 5 on GPU, 1 on CPU: on CPU a 2B model with beams is too slow for interactive use)
    SPEECH_PRELOAD  1 = load the model in a background thread when the server starts (api/main.py), so the first
                    recording does not wait for the ~3 GB download / load
"""
import os
import threading

import numpy as np

PROVIDER = os.environ.get("SPEECH_PROVIDER", "whisper").strip().lower()
DEFAULT_WHISPER = "Dr-AliGomaa/whisper-large-v3-ar"
COHERE_MODEL = "CohereLabs/cohere-transcribe-arabic-07-2026"
MODEL_NAME = os.environ.get("WHISPER_MODEL") or DEFAULT_WHISPER
SAMPLE_RATE = 16000
MAX_SECONDS = 30  # the UI stops at 29 s (one Whisper window)
MIN_SECONDS = 0.4
_LANG = {"ar": "arabic", "en": "english"}
_LOCK = threading.Lock()  # one inference at a time, same reasoning as core.embedding
_LOAD_LOCK = threading.Lock()
_backend_cache: dict = {}


class AudioError(ValueError):
    pass


def provider() -> str:
    return os.environ.get("SPEECH_PROVIDER", PROVIDER).strip().lower()


def model_name() -> str:
    return COHERE_MODEL if provider() == "cohere" else (os.environ.get("WHISPER_MODEL") or DEFAULT_WHISPER)


def _device() -> str:
    import torch
    want = (os.environ.get("SPEECH_DEVICE") or "auto").lower()
    return "cuda:0" if want != "cpu" and torch.cuda.is_available() else "cpu"


class _Whisper:
    def __init__(self, name: str):
        import torch
        import truststore
        truststore.inject_into_ssl()
        from transformers import pipeline
        self.device = _device()
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.pipe = pipeline("automatic-speech-recognition", model=name, dtype=dtype, device=self.device, chunk_length_s=30)

    def __call__(self, audio: np.ndarray, language: str | None) -> str:
        beams = int(os.environ.get("SPEECH_BEAMS") or (5 if self.device.startswith("cuda") else 1))
        kwargs = {"task": "transcribe", "num_beams": beams}
        if language in _LANG:
            kwargs["language"] = _LANG[language]
        return self.pipe({"raw": audio, "sampling_rate": SAMPLE_RATE}, generate_kwargs=kwargs)["text"]


class _Cohere:
    def __init__(self, name: str):
        import torch
        import truststore
        truststore.inject_into_ssl()
        from transformers import AutoProcessor, CohereAsrForConditionalGeneration
        self.device = _device()
        token = os.environ.get("HF_TOKEN") or None
        self.processor = AutoProcessor.from_pretrained(name, token=token)
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.model = CohereAsrForConditionalGeneration.from_pretrained(name, token=token, dtype=dtype).to(self.device)

    def __call__(self, audio: np.ndarray, language: str | None) -> str:
        inputs = self.processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt", language=language or "ar")
        inputs = inputs.to(self.model.device, dtype=self.model.dtype)
        out = self.model.generate(**inputs, max_new_tokens=256)
        dec = self.processor.batch_decode(out, skip_special_tokens=True)
        return dec[0] if isinstance(dec, (list, tuple)) else str(dec)


def _backend():
    """The loaded model for the configured provider (loaded once; tests monkeypatch this)."""
    key = (provider(), model_name())
    if key not in _backend_cache:
        with _LOAD_LOCK:
            if key not in _backend_cache:
                _backend_cache[key] = _Cohere(key[1]) if key[0] == "cohere" else _Whisper(key[1])
    return _backend_cache[key]


def preload():
    """Load the model in the background (SPEECH_PRELOAD=1) so the first recording does not wait for it."""
    def run():
        try:
            _backend()
        except Exception:
            pass  # the first real request will report "speech recognition unavailable"
    threading.Thread(target=run, daemon=True, name="speech-preload").start()


def describe() -> dict:
    return {"provider": provider(), "model": model_name(), "loaded": (provider(), model_name()) in _backend_cache,
            "max_seconds": MAX_SECONDS}


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


def prepare(audio: np.ndarray) -> np.ndarray:
    """The Whisper-AR model card's input chain, simplified: trim leading/trailing silence, pad 100 ms, normalise loudness
    to about -16 dBFS (RMS, gain capped at 30x so near-silence is not blown up into noise)."""
    frame = SAMPLE_RATE // 50  # 20 ms
    n = len(audio) // frame
    if n:
        rms = np.sqrt(np.mean(audio[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
        voiced = np.where(rms > max(0.01, 0.1 * float(rms.max())))[0]
        if len(voiced):
            audio = audio[max(0, voiced[0] - 5) * frame: min(n, voiced[-1] + 6) * frame]
    pad = np.zeros(SAMPLE_RATE // 10, dtype=np.float32)
    audio = np.concatenate([pad, audio, pad]).astype(np.float32)
    level = float(np.sqrt(np.mean(audio ** 2) + 1e-12))
    gain = min(10 ** (-16 / 20) / level, 30.0)
    return np.clip(audio * gain, -1.0, 1.0).astype(np.float32)


def transcribe(data: bytes, language: str | None = None) -> str:
    """language 'ar'/'en' pins the model's language (Whisper mis-detects short Arabic clips); None lets it detect."""
    audio = pcm16_to_float(data)
    if float(np.abs(audio).max()) < 0.005:
        raise AudioError("silent audio")
    audio = prepare(audio)
    with _LOCK:
        text = _backend()(audio, language)
    return " ".join((text or "").split())
