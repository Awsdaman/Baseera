# HANDOFF — where we stopped (2026-10-03)

Read this first in a new session, then CLAUDE.md and README.md.

## State of the project
- All 5 phases built and committed. Tests: **142 pass** (`python -m pytest -q`). DB + Chroma vectors exist locally in `data/db/` (git-ignored; rebuild with `python ingest/build_all.py`).
- LLM provider is switchable in `.env` (`LLM_PROVIDER=openai|anthropic|local`). Currently: OpenAI (router `gpt-5.4-mini`, generator `gpt-5.6-sol`, judge `gpt-5.5`).
  - `gpt-5.6-sol` was my pick among `gpt-5.6-luna|sol|terra` (user said only "5.6"); change `OPENAI_GENERATE_MODEL` in `.env` if they meant another.
  - Anthropic key in `.env` has **no credit** ("credit balance too low"). A Claude judge needs credit, then `JUDGE_PROVIDER=anthropic` and `python evals/run_evals.py --rejudge evals/reports/results-<ts>.json`.
- Secrets: `.env` (real file), `OpenAI API.txt` and `env-folder-original/` are in the project root and **git-ignored, never committed**. Move them out of the folder before zipping/submitting. `data/raw` is also git-ignored (re-fetch with `python ingest/download_raw.py`; two Quran Complex zips must be placed by hand, see README).
- Last full live eval (final build before the current work): router 94.2%, retrieval 100%, citations 100%, verse fidelity 100%, correct abstention 100%, false abstention 3.4% (b-09), behaviour 98.1%, verify mode 5/5, judge 4.76/5. One full run ~ 300k input / 40k output tokens.

## Status (updated 2026-10-03, end of session)
Everything in the Opus review plan and the four SAWB-inspired ideas is implemented and committed. **Final live eval (52 cases, complete and clean)**: router 100%, retrieval 12/12, citations 100% (39), verse fidelity 100% (40), correct abstention 100% (7), verify mode 5/5, false abstention 0/30, verifier-forced abstention 0/39, behaviour 52/52, judge 4.71/5; 40 answered / 6 referred / 5 verified / 1 expected abstention; 3 answers needed the retry; support check flagged 3/130 stretches. The run was produced with `--resume` (27 clean results from a run that ran out of credit + 25 re-run). 187 tests pass (186 + 1 slow). `docs/pipeline.html` shows the architecture and these numbers.

Open quality points the judge/support check raised (not fixed): contested-fiqh group scores lowest (c: 4.48); a-15 (partial list of Quran prophets) 3.83; b-02 and c-03 phrase sensitive claims too definitively; dp6-02 cites a weakly related tafsir passage (the support check also flagged it).

## Pipeline-enhancement ideas (from the SAWB reference photo), done one by one
1. DONE: semantic support check (`core/support.py`, log mode default, enforce opt-in); reranker rejected on data.
2. DONE (verify-mode part): synthetic data `evals/synth_verify.py` found and fixed a real shortlist bug (misquote recall 73% -> 99.8%). Not done on purpose: LLM-generated Q&A data (costs money; the router is already 100% on the golden set).
3. DONE: parallel eval workers (`--workers N`, default 4; 1 for a local model). Made retrieval/embedding/http/token counting thread-safe. 3.7x faster live (134 s vs 498 s on 16 cases), identical outcomes.
4. DECIDED NOT TO BUILD (evidence in `evals/router_check.py`): the heuristic router flags only 0.4% (2/543) of real icadb general questions as personal cases (level د) and an embedding kNN gets 77% leave-one-out on 47 labelled questions (needs hundreds of labels). Router cost is ~0.4% of an eval run; the rule fallback already exists. Revisit only if the chosen local model routes badly (then train a small classifier on bge-m3 embeddings with labelled data).

## Added after the eval round (user-driven)
- Question understanding: router returns `canonical_question` + `claim`; a verify request with nothing to verify (or only unidentifiable quotes) falls back to a normal answer; generator may give a partial answer with the code-owned `{{note:no_ruling}}` instead of refusing when passages are on-topic but do not state the ruling; Arabic answers may not contain English words (verifier).
- Verify mode for hadith now checks WORDING like verses: word-level alignment (ignoring a leading و/ف), shows the correct wording + diff + the curated HadeethEnc grade (Dorar remarks are never used as the overall grade), distinctive-word and rarity guards for short claims, "too generic" for boilerplate. Router forces verify for attributed quotes without a question; extractor falls back to rules when the model finds nothing. UI shows a highlighted correct-wording block; pages are served with `Cache-Control: no-cache`.
- Opt-in learning loop (see README "Learning over time"): `core/reports.py`, `/api/report`, `/api/privacy`, report box in the UI, `tools/review.py`, `core/rewrites.py`, `data/curated/`. 226 tests pass.
- Last full live eval after these changes (55 cases, before the last router/extractor stability fixes): 53/55; the two misses (v-02, v-03) were the router calling pasted hadiths "ask" and are fixed and re-run green. A fresh full run is advisable.

## Left to do
1. (Optional, ~$1) one more live run to confirm dp6-07 and get a clean 0 forced abstentions: `python evals/run_evals.py --only dp6-07 --no-judge` (cheap) or the full run.
2. **Local-model backend** — waiting for the user to pick the model. `core/llm.py` already supports `LLM_PROVIDER=local` (OpenAI-compatible server: Ollama/LM Studio/vLLM; `LOCAL_BASE_URL`, `LOCAL_MODEL`; tested with stubs only). Once chosen: try it live; add small-context limits (cap passages / `max_chars` in `core/generate.py` through env); check Arabic JSON routing quality (the heuristic router is the fallback); run the evals with `LLM_CACHE=1`.
3. Claude judge when Anthropic credit exists: `JUDGE_PROVIDER=anthropic` + `--rejudge evals/reports/results-<ts>.json`.
4. Replace the 4 provisional viral-hadith eval cases (v-01..v-04) with real examples from Dorar's widespread-hadith section.
5. Before zipping/submitting: move `OpenAI API.txt` and `env-folder-original/` out of the project folder (git-ignored, but they are in the folder).

## Known gaps (documented, not fixed)
- One citation still covers a whole list/paragraph block (accepted).
- `fidelity_ok` is True for answers with no Quran block (it also counts leak checks).
- Judge and generator are both OpenAI (self-preference bias) until a Claude judge is possible.
- The 4 provisional viral-hadith eval cases (v-01..v-04) should be replaced with real examples from Dorar's widespread-hadith section.
