# HANDOFF — where we stopped (2026-10-03)

Read this first in a new session, then CLAUDE.md and README.md.

## State of the project
- All 5 phases built and committed. Tests: **142 pass** (`python -m pytest -q`). DB + Chroma vectors exist locally in `data/db/` (git-ignored; rebuild with `python ingest/build_all.py`).
- LLM provider is switchable in `.env` (`LLM_PROVIDER=openai|anthropic|local`). Currently: OpenAI (router `gpt-5.4-mini`, generator `gpt-5.6-sol`, judge `gpt-5.5`).
  - `gpt-5.6-sol` was my pick among `gpt-5.6-luna|sol|terra` (user said only "5.6"); change `OPENAI_GENERATE_MODEL` in `.env` if they meant another.
  - Anthropic key in `.env` has **no credit** ("credit balance too low"). A Claude judge needs credit, then `JUDGE_PROVIDER=anthropic` and `python evals/run_evals.py --rejudge evals/reports/results-<ts>.json`.
- Secrets: `.env` (real file), `OpenAI API.txt` and `env-folder-original/` are in the project root and **git-ignored, never committed**. Move them out of the folder before zipping/submitting. `data/raw` is also git-ignored (re-fetch with `python ingest/download_raw.py`; two Quran Complex zips must be placed by hand, see README).
- Last full live eval (final build before the current work): router 94.2%, retrieval 100%, citations 100%, verse fidelity 100%, correct abstention 100%, false abstention 3.4% (b-09), behaviour 98.1%, verify mode 5/5, judge 4.76/5. One full run ~ 300k input / 40k output tokens.

## Status (updated): the Opus review plan is fully implemented and committed
Done: code-owned `{{note:...}}` markers, multi-id citations + fix recipes, retry that shows the previous answer, debug trace (dev/evals only; the API strips `verification_errors`/`debug`), router escalation (concepts/war topics -> ب, contested -> ج) and translate_term clarification, level-د cards without hadith, eval runner (one ask per case, traces, token use per case, verifier-forced-abstention metric, `--reverify`, `--rejudge`), `LLM_CACHE=1`, golden-set corrections (12 retrieval targets), stock-phrase handling for quoted formulas, docs.
Final live eval (52 cases): router 100%, retrieval 12/12, citations 100%, verse fidelity 100%, correct abstention 100%, verify mode 5/5, false abstention 1/30 (dp6-07, fixed afterwards and confirmed via `--reverify`, not re-run live), judge 4.71. 171 tests pass.

## Pipeline-enhancement ideas (from the SAWB reference photo), done one by one
1. DONE: semantic support check (`core/support.py`, log mode default, enforce opt-in); reranker rejected on data.
2. DONE (verify-mode part): synthetic data `evals/synth_verify.py` found and fixed a real shortlist bug (misquote recall 73% -> 99.8%). Not done on purpose: LLM-generated Q&A data (costs money; the router is already 100% on the golden set).
3. DONE: parallel eval workers (`--workers N`, default 4; 1 for a local model). Made retrieval/embedding/http/token counting thread-safe. 3.7x faster live (134 s vs 498 s on 16 cases), identical outcomes.
4. DECIDED NOT TO BUILD (evidence in `evals/router_check.py`): the heuristic router flags only 0.4% (2/543) of real icadb general questions as personal cases (level د) and an embedding kNN gets 77% leave-one-out on 47 labelled questions (needs hundreds of labels). Router cost is ~0.4% of an eval run; the rule fallback already exists. Revisit only if the chosen local model routes badly (then train a small classifier on bge-m3 embeddings with labelled data).

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
