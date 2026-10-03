# Baseera (بصيرة)

Arabic-first grounded Islamic Q&A and verification assistant for the "AI Challenge Serving Islamic Content". 3-day build. Pitch: `docs/the_idea.pdf`. Official data pack (levels, standards, test cases, glossary, approved sources): `docs/data.pdf`.

## Working agreement
- Work in phases (0-5, see below). At the end of EACH phase: run tests, commit, give a short summary of what works and what doesn't, then STOP for approval.
- Never guess an API's shape: probe it first and adapt.
- Never commit `.env` or API keys.

## Modes
1. ASK: answer a question (Arabic/English) ONLY from retrieved approved sources, with a citation card per claim.
2. VERIFY: user pastes a message (e.g. viral WhatsApp text); extract every verse/hadith claim and report: verified verse / misquoted verse (correct text + word diff) / hadith grade from scholars / not found in approved sources.

## Non-negotiable rules (enforce in code, not only prompts)
1. The LLM NEVER writes Quran or hadith text. It emits placeholders `{{quran:SURAH:AYAH}}`, `{{quran:SURAH:AYAH-AYAH}}`, `{{hadith:SOURCE:ID}}`; code substitutes exact text from the local DB.
2. Every claim cites a source ID actually retrieved for this question. Unretrieved/nonexistent IDs fail verification.
3. Hadith grades come ONLY from Dorar / HadeethEnc data, never the model.
4. Insufficient evidence -> abstain with a warm, useful message suggesting where to look. Never fabricate.
5. Content levels (data.pdf p.2):
   - أ (Quran, sahih hadith, pillars, basic seerah): direct sourced answer
   - ب (concepts, maqasid, common doubts): sourced explanation, no false certainty on disputed points
   - ج (fiqh disagreement, detailed aqeedah, contested history): present positions with attribution, no automatic tarjih, or refer to a specialist
   - د (personal case, contract validity, family dispute, legal/medical): general info + referral ONLY, never a ruling
6. UI clearly separates Quran text, hadith text, tafsir, and generated explanation.
7. Approved glossary term translations (data.pdf p.7 + icadb) override machine translation.
8. Show a line disclosing Baseera is an AI tool. Store no personal data.

## Stack
Python 3.13 (3.11 was specified; 3.13 approved), FastAPI, SQLite + FTS5 (normalized Arabic), Chroma, bge-m3 or multilingual-e5-large, rapidfuzz, Anthropic SDK, python-dotenv, pytest.
Models: `Codex-haiku-4-5-20251001` (routing, claim extraction), `Codex-sonnet-5-5` (answer generation).
Frontend: single RTL Arabic page served by FastAPI (plain HTML/CSS/JS), Arabic/English toggle. Colors: navy #1B2D45, teal #1F7872, gold #C9A04A.

## Layout
`docs/ ingest/ core/ api/ web/ evals/ data/raw/ data/db/ data/cache/ data/samples/ tests/`

## Data sources
Local index: Quran Complex Hafs JSON (use `kfgqpc_hafs_v30`, real Unicode + tashkeel, 6236 verses; NOT `hafs_smart_v8` whose main text is private-use glyphs), HadeethEnc (ar+en), QuranEnc (`english_saheeh`, Arabic tafsir `arabic_moyassar`; local Muyassar zip also in data/raw/tafsir), Bayyinat Q&A PDF (`data/raw/bayyinat/bayyinat_qa.pdf`, chunk by question), glossary (data.pdf p.7 + icadb).
Live (cache everything in data/cache/): Dorar `dorar.net/dorar_api.json` (needs browser User-Agent; Cloudflare 403 otherwise; returns HTML snippets), icadb (`icadb.com/api/docs/?format=openapi`), mp3quran (`https://www.mp3quran.net/api/v3/...`, use `www`).
Referral links only (never scrape): islamqa.info, binbaz.org.sa, binothaimeen.net.

## Phases
0 setup + API probing | 1 ingestion + hybrid retrieval | 2 router/generate/verify | 3 verify mode | 4 evals | 5 polish + demo + README.

## Conventions
Keep original text for display; normalized text (no tashkeel/tatweel, أإآ->ا, ى->ي, ة->ه) for matching only. Ingest scripts are idempotent. Tests in `tests/`.

## API findings (Phase 0, verified)
- HadeethEnc (docs: documenter.getpostman.com/view/5211979/TVev3j7q): endpoints `/hadeeths/list/` (per_page=1000 works), `/one/`, `/multiple/?ids=a,b`, `/search/?phrase=`. `multiple` with language=en returns full records INCLUDING Arabic fields (`hadeeth_ar, grade_ar, explanation_ar, hints_ar, attribution_ar`), so one English pass gives both languages. License: NO modification/addition/deletion of content, and must clearly credit HadeethEnc.com in the UI. Grades are in `grade` / `grade_ar`.
- Dorar (docs: dorar.net/article/389): only documented param is `skey`; response `{"ahadith":{"result":"<html>"}}`; undocumented `page` seems to work. No documented rate limit; needs browser User-Agent. Parse `div.hadith` + `div.hadith-info` (الراوي, المحدث, المصدر, الصفحة أو الرقم, خلاصة حكم المحدث). No hadith ID.
- icadb: public GET, OpenAPI at `/api/docs/?format=openapi`. Q&A encyclopedias: id 2 (Muslims), 11 (non-Muslims); terminology: id 5.
- Bayyinat PDF text extraction is garbled (swapped letters/lost lam-alef); clean up or fall back to icadb Q&A.
- Python needs `truststore` for HTTPS on this machine.
