# HANDOFF — where we stopped (2026-10-03)

Read this first in a new session, then CLAUDE.md and README.md. Voice input ("ask by voice"): `docs/VOICE_HANDOFF.md`.

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

## PENDING TODO (decided 2026-10-06, deadline that day 11:59 PM; saved so nothing is forgotten)
Independent review findings to fix (all with tests, no GPU needed), in this order:
1. English typed-scripture hole: verify.py checks only Arabic; add a check against the retrieved verses' English translation (text_en) / hadith English (H1).
2. Router: `_PERSONAL` matches "ابي" as a substring (Abu Bakr -> level د) and bare "can i/should i"; add first-person medical patterns ("أنا مريض", "I have diabetes", "I am pregnant") to _PERSONAL/_STRONG_PERSONAL (H6, H7).
3. Dorar: circuit breaker + retries=1 + short timeout + cap of ~8 claims; ship data/cache in the package (H2).
4. Distinct "service problem" message for llm_error / llm_empty / llm_truncated (not "not enough evidence"); startup DB-empty check; warm up bge-m3 at startup with a vectors_ready flag in /api/health (H3, H4, H9).
5. Commit the uncommitted search_ar (English->Arabic search) change; rebuild the zip from HEAD (git archive) incl. data/cache; link it in README/QUICKSTART (private hand-over to judges, not a public Release, because it contains third-party sources) (H4, H5).
6. Pin requirements.txt; add LICENSE + NOTICE (credits for HadeethEnc, KFGQPC, QuranEnc, Dorar, icadb, Bayyinat); do not truncate hadith cards (H8, H10).
7. Docs honesty: test count 267, remove the stale "live LLM unverified" paragraph, retrieval 13/14 + rerun, commit the 17-case rerun and a reliability summary, golden = 55 cases, HANDOFF stale step (L, section 3 of the review).
8. Small API hardening: transcribe body size check, drop public dorar=true on /api/retrieve, per-IP rate limit + a semaphore around LLM calls, health endpoint exposes less.
Then re-run golden + reliability once on the final build.

## Approved-sources list from the organizers (docs/approved_sources_links.txt): audit and gap-filling in progress
We want the best possible sources for the questions (it scores points). See docs/sources_audit.md once written.

## FINAL VERSION (2026-10-06): everything below was re-run on the final build
- Golden 55 (Gemma 4 12B local, judge gpt-5.5): behaviour 100%, citations/verse fidelity 100%, correct abstention 7/7, false abstention 0/32, forced abstention 0/41, judge 4.86. 17 glossary-related cases re-run after the last retrieval fix: 17/17.
- Reliability (evals/reliability.py, 51 q): should-answer 14/14, no-source 16/16, disputed 6/6, paraphrase outcome 5/5 (levels differ in 3 groups).
- 48 fiqh questions (evals/bulk_check.py): 40 answered (32 with a limits note), 8 declined; no confident wrong-source answer shown.
- Synthetic verify: verse macro-F1 about 0.98 (script suggests 0.80 threshold, 0.997; current 0.72 kept), hadith F1 0.963. Tests: 265 pass (+1 slow).
- New since the plan: on-topic check (core/relevance.py, REL_MODE), level-د demotion to ج for general worship questions, "prove this" clarification, shahada stock-formula exemption, glossary card always retrieved, empathy opener, evidence-order rule. Docs: docs/how_baseera_decides.md, docs/pipeline.html, README.
- Known limits: fiqh details are only as good as the approved sources (the system declines or adds a "not directly addressed" note); answer LEVEL labels can differ between paraphrases; LM Studio ignores context-length requests; the laptop is untested; 4 of 48 fiqh answers were rejected by the verifier (uncited text, a verse recited from memory) and shown as declines.
- Local files not in git: data/questions/, "The enhanced UI UX/", design-options/ (UI redesigns waiting for review), evals/reports/bulk-* and reliability-* (contain full answers).

## FINAL local result (2026-10-04, build faab13c): Gemma 4 12B Q4_K_M via LM Studio, judge gpt-5.5
55/55 behaviour, router 100% (LLM router used 48/48), citations 100%, verse fidelity 100%, correct abstention 7/7, false abstention 0/32,
verifier-forced abstention 0/41, runtime abstentions 0, first-attempt pass 92.7%, judge 4.86/5 (OpenAI run: 4.71), 33 min per full run, ~130 output tokens/call.
Only miss: dp6-07 retrieval (term:glossary:2 not retrieved; flaky, it passed on a rerun). Run with:
`LLM_PROVIDER=local LOCAL_BASE_URL=http://localhost:1234/v1 LOCAL_MODEL=gemma-4-12b-it LOCAL_NO_THINK=1 LLM_GEN_MAX_TOKENS=1500 LLM_ROUTER_MAX_TOKENS=600 JUDGE_PROVIDER=openai python evals/run_evals.py --workers 1`
Decision: Gemma 4 12B is the local model for the desktop. Not tested: the laptop (RTX 4050 6 GB; plan: use the desktop LM Studio over the network, or partial offload).

## Local model plan implemented (2026-10-04) - NOT yet re-evaluated
All 7 steps of docs/local_model_plan.md are in the code (242 tests pass). The eval has NOT been re-run yet.
- Findings behind it: the first live Gemma run never used the LLM router (hidden reasoning ate the 300-token cap), `/no_think` is Qwen-only,
  dp6-03 was `[[a], [b]]` citation syntax, v-06 a narration pattern gap.
- LM Studio honours `reasoning_effort="none"` for Gemma 4 (probe: 10 tokens vs 134); `chat_template_kwargs` is ignored. Context length requests are
  ignored by `lms load -c` and the REST API (still 262144).
- Code: `core/llm.py` (`no_thinking()`, `thinking_off()`, `LLMTruncated`, no empty cache, max_retries), `core/verify.py` (`normalize_citations`,
  Malformed citation error, de-dup), `core/pipeline.py` (`_generate_once` runtime retry, `llm_empty`/`llm_truncated`, `LLM_MAX_ATTEMPTS`),
  `core/generate.py` (local checklist profile, `fix_recipe`), `core/router.py` (`_NARRATION`), `evals/run_evals.py` (router_llm_rate, runtime_abstentions,
  first_attempt_pass_rate, taint warning). `--reverify` on the saved Gemma run now flips dp6-03, dp6-11 (att.1), a-02, b-01, c-02 (att.1) to pass.
- Next: run `LOCAL_NO_THINK=1 LLM_CACHE=1 LLM_GEN_MAX_TOKENS=1500 LLM_ROUTER_MAX_TOKENS=600 python evals/run_evals.py --no-judge --workers 1`
  (smoke subset first: --only dp6-03,dp6-11,c-02,v-06,a-01,a-02,a-03,a-07,a-10,a-14,b-01), then twice in full; gates in the plan.

## Local model selection (in progress, resume here tomorrow)
Hardware: this machine IS the desktop (Ryzen 5 7600X, 32 GB RAM, RX 7600 XT 16 GB, LM Studio with the Vulkan llama.cpp 2.51.0 engine selected; `lms` CLI works; `lms runtime survey` sees 15.98 GiB VRAM). Laptop (RTX 4050 6 GB) not tested yet.

Research (Oct 2026, web): Arabic-native Falcon-H1-Arabic 7B tops the ~10B class (OALL 71.7) but its Hugging Face repos return 401 (gated/private), so it is not downloadable; ALLaM-7B / Fanar-9B have weak GGUF support. Strong multilingual options that fit 16 GB: **Gemma 4 12B-it** (140 languages, MMMLU 83.4, ~7 GB Q4_K_M, 256K ctx) = my expected winner; Qwen3.5-9B (unsloth Q6_K 7.5 GB); Qwen3-14B (lmstudio-community Q4_K_M 9.0 GB); Qwen3.6/3.8-27B dense Q4 is ~16.5 GB (too tight with context); Qwen3.6-35B-A3B MoE needs ~21 GB (partial offload). Laptop (6 GB) candidates: a 4B-class model or Qwen3.5-9B at Q4 with a small context (untested).

User approved downloading ONLY Gemma 4 12B (unsloth/gemma-4-12b-it-GGUF, file gemma-4-12b-it-Q4_K_M.gguf, 7.12 GB). `lms get` could not resolve the name, so the file was fetched directly with curl to `~/.lmstudio/models/unsloth/gemma-4-12b-it-GGUF/` (check it is complete: size ~7.12e9 bytes).

Code is ready for local models: `LLM_PROVIDER=local`, `LOCAL_BASE_URL`, `LOCAL_MODEL`, `LOCAL_NO_THINK=1` (Qwen soft switch), `LLM_PASSAGE_CHARS` (shrink passages for small contexts), hidden `<think>` text is stripped, `--workers 1` is the default for local. 230 tests pass.

NEXT STEPS (tomorrow):
1. `lms ls` should list the model; load it with a 16K context and all layers on the GPU, e.g. `lms load unsloth/gemma-4-12b-it-GGUF --gpu max --context-length 16384 --identifier gemma4-12b` (check `lms load --help`), then `lms server start` (port 1234).
2. Smoke test + speed: set in `.env` (or the shell) `LLM_PROVIDER=local LOCAL_BASE_URL=http://localhost:1234/v1 LOCAL_MODEL=gemma4-12b JUDGE_PROVIDER=openai`; ask one question through `core.pipeline.ask(...)`; measure tokens/second and prompt-processing time (prompts are ~5-8K tokens).
3. Full eval on the local model WITHOUT spending credit: `python evals/run_evals.py --no-judge --workers 1` (about 40 min), with `LLM_CACHE=1`. Compare with the OpenAI numbers (router 100%, citations 100%, fidelity 100%, forced abstentions 0, false abstention 0). Then optionally judge it with `--rejudge` (small OpenAI cost) or a Claude judge.
4. Look at what breaks: the verifier retry rate, Arabic quality, placeholders/notes use, JSON from the router (the rule-based router is the fallback). If Gemma 4 12B is not good enough, ask the user before downloading the next candidate (Qwen3.5-9B Q6_K or Qwen3-14B Q4_K_M).
5. Decide the production split: local model for generation/routing on the desktop (can serve the team), OpenAI only as optional judge. Then repeat a smaller test on the laptop.

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
