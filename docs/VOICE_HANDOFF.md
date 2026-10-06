# HANDOFF — Ask by voice (speech-to-text) — 2026-10-06

> **ملخص بالعربية:** ميزة "اسأل بصوتك" تعمل بنموذج مفتوح **على خادم بصيرة نفسه** (لا يُرسل الصوت لأي جهة خارجية).
> رُقّي النموذج الافتراضي من `openai/whisper-base` (ضعيف في العربية) إلى `Dr-AliGomaa/whisper-large-v3-ar` المدرَّب على القرآن والحديث،
> مع خيار Cohere للهجات، وتشغيل تلقائي على كرت الشاشة إن وُجد، وتنظيف الصوت قبل التفريغ. جُرِّب على لابتوب (المعالج فقط) فكانت الجودة
> جيدة لكن **بطيئة قليلًا**؛ الحل الأهم: PyTorch بدعم CUDA. النص يوضع في مربع السؤال **ليراجعه المستخدم** ولا يُرسل تلقائيًا.

Main code: `core/speech.py`. Endpoint: `POST /api/transcribe` in `api/main.py`. UI: the `#mic` block at the end of `web/index.html`.
Tests: `tests/test_speech.py`.

## 1. What the feature does
- 🎤 button in the Ask composer (hidden in Verify mode). Records up to **29 s** (one Whisper window).
- The browser decodes and resamples to **16 kHz mono PCM16** itself and posts the raw bytes to `/api/transcribe?language=ar|en`
  (no ffmpeg on the server).
- The server returns `{"text": ...}`; the text is appended to the question box for the user to **review and fix**. It is never
  submitted automatically (speech recognition can misspell verses and names; the pipeline and its verification are unchanged).
- Errors: 400 bad/short/silent audio, 413 too long, 422 bad language, 503 model unavailable (never a model message).

## 2. Model choice (decision log)
Publishers' own numbers, not comparable with each other:

| Model | Strength | Weakness | Licence | In Baseera |
|---|---|---|---|---|
| `Dr-AliGomaa/whisper-large-v3-ar` | Whisper large-v3 fine-tuned on Quran + hadith + MSA: WER 0.33% Quran, 3.6% hadith, 9.5% MSA (MGB-2). Imla'i text without tashkeel, the form our matchers compare | Not trained on dialects | Apache-2.0, not gated | **Default** |
| `CohereLabs/cohere-transcribe-arabic-07-2026` | 2B, Arabic + English, dialects; Open Universal Arabic ASR avg WER 25.9% | No Quran numbers; gated (`HF_TOKEN`) | Apache-2.0 | `SPEECH_PROVIDER=cohere` |
| `audarai/Audar-ASR-V1-Turbo` | #1 on Open Universal Arabic ASR (23.2% avg WER), GGUF for CPU | Generative audio-LLM; no Quran numbers | Community licence with commercial limits | Not wired in (could be served by vLLM later) |
| `openai/whisper-base` | Small, fast on CPU | Weak Arabic | MIT | Old default; still usable via `WHISPER_MODEL` |
| Groq hosted Whisper (earlier experiment) | Free cloud | Many Arabic mistakes, audio leaves the server | — | Rejected |

Why the default: in Baseera the costly error is a wrong word inside a verse or hadith; the scripture-trained Whisper minimises it.
**Not done yet:** our own measurement on ~30 recorded clips (verses, hadiths, dialect questions) scored with
`core.verifier_mode.check_verse` / WER, before changing defaults again.

## 3. Configuration (`.env`)
| Variable | Default | Meaning |
|---|---|---|
| `SPEECH_PROVIDER` | `whisper` | `whisper` \| `cohere` |
| `WHISPER_MODEL` | `Dr-AliGomaa/whisper-large-v3-ar` | Any Hugging Face Whisper id (e.g. `openai/whisper-base` for a fast CPU demo) |
| `HF_TOKEN` | — | Only for the gated Cohere model |
| `SPEECH_DEVICE` | `auto` | `auto` (CUDA if PyTorch sees a GPU) \| `cpu` \| `cuda` |
| `SPEECH_BEAMS` | 5 on GPU, 1 on CPU | Whisper beam size |
| `SPEECH_PRELOAD` | off | `1` = load the model in a background thread at server start |

## 4. Code map (`core/speech.py`)
- `provider()`, `model_name()`, `describe()` (reported by `/api/health` as `speech`: provider, model, loaded, max_seconds).
- `pcm16_to_float` (length checks), `prepare` (trim leading/trailing silence, 100 ms pads, RMS loudness to about -16 dBFS with gain
  capped at 30x: a simplified version of the Whisper-AR card's chain), `transcribe(data, language)`.
- Backends: `_Whisper` (transformers ASR pipeline, `chunk_length_s=30`, fp16 on CUDA / fp32 on CPU) and `_Cohere`
  (`CohereAsrForConditionalGeneration`, needs transformers >= 5.4). Loaded once per (provider, model) in `_backend()`;
  `_LOCK` = one inference at a time (a 2B model per request would exhaust memory). `preload()` = daemon thread.
- `api/main.py`: `lifespan` calls `speech.preload()` when `SPEECH_PRELOAD=1`; `/api/health` includes `speech`.

## 5. Privacy (must stay true)
Audio and transcript are processed in memory only, never written to disk, cached or logged; nothing leaves the server. Only the
model weights are cached by Hugging Face (`%USERPROFILE%\.cache\huggingface`).

## 6. Status (2026-10-06)
- Tried live on a laptop (Ryzen + RTX 4050 6 GB, Windows, CPU-only PyTorch): quality good, **a bit slow** (2B model in fp32 on CPU).
  The first use also downloads ~3 GB unless preloaded.
- The same design was first built as a separate `core/stt.py` on the pre-reviewer copy; this version ports it onto the current
  `core/speech.py` API (same endpoint and UI, raw PCM16 body).

## 7. Next steps
1. **GPU:** install the CUDA build of PyTorch in the venv (pick the wheel index on pytorch.org that matches the installed torch and
   Python), check `torch.cuda.is_available()`; fp16 model ~3 GB fits a 6 GB card. Expected ~1 s per sentence. Measure before/after.
2. CPU-only machines: keep `SPEECH_BEAMS=1`, or convert the model to CTranslate2 int8 and add a `faster-whisper` backend (≈3-4x).
3. Labelled audio set + comparison (section 2); record the numbers in README.
4. UI: show "preparing the voice model…" while `/api/health` reports `speech.loaded=false`; show elapsed seconds while transcribing.
5. AMD desktop (RX 7600 XT): CUDA does not apply; CPU, or Audar GGUF via llama.cpp (Vulkan) behind a small server.
