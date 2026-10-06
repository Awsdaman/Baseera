# Baseera quick start (working copy)

Baseera is an Arabic-first Islamic Q&A and verification assistant that answers only from approved sources. This copy includes the ready-built sources database (`data/db`, about 300 MB: Quran, tafsir, hadith, Q&A, 40,000 passages of the organizers' books and the Dorar fiqh encyclopedia) and the cached Dorar hadith answers, so nothing needs to be ingested; only the retrieval model is downloaded on the first run (about 2.3 GB, internet needed). Open http://127.0.0.1:8000/api/health: `vectors_ready: true` means that model has loaded.

## 1. Install (Python 3.11-3.13)

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

The first question downloads the retrieval model `BAAI/bge-m3` (about 2.3 GB, once, internet needed).

## 2. Choose how answers are written (pick ONE; create a file named `.env` next to this file)

| Option | `.env` content | Notes |
|---|---|---|
| Free local model (no key) | `LLM_PROVIDER=local`<br>`LOCAL_BASE_URL=http://localhost:1234/v1`<br>`LOCAL_MODEL=gemma-4-12b-it`<br>`LOCAL_NO_THINK=1`<br>`LLM_GEN_MAX_TOKENS=1500`<br>`LLM_ROUTER_MAX_TOKENS=600` | Needs LM Studio with `gemma-4-12b-it` loaded and its server started; 16 GB GPU recommended; about 30-60 s per answer |
| OpenAI | `LLM_PROVIDER=openai`<br>`OPENAI_API_KEY=<your key>` | Your own key |
| Claude | `LLM_PROVIDER=anthropic`<br>`ANTHROPIC_API_KEY=<your key>` | Your own key |
| No model at all | (no `.env`) | Verify mode, glossary, referrals and source search still work; Ask shows the closest approved sources |

Never share or commit `.env`.

## 3. Run

```bash
uvicorn api.main:app
```

Open http://127.0.0.1:8000 : the **Ask** tab answers with citation cards; the **Verify** tab checks pasted verses and hadiths (try a hadith with a missing word).

## 4. Check it

```bash
python -m pytest -q                       # about 290 tests
python evals/run_evals.py --offline       # the part of the golden set that needs no model
```

How decisions are made (sources, citation mechanism, when it answers or declines): `docs/how_baseera_decides.md`. Architecture diagram: open `docs/pipeline.html`. Full results and limits: `README.md`.
