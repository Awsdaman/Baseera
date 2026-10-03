# HANDOFF — where we stopped (2026-10-03)

Read this first in a new session, then CLAUDE.md and README.md.

## State of the project
- All 5 phases built and committed. Tests: **142 pass** (`python -m pytest -q`). DB + Chroma vectors exist locally in `data/db/` (git-ignored; rebuild with `python ingest/build_all.py`).
- LLM provider is switchable in `.env` (`LLM_PROVIDER=openai|anthropic|local`). Currently: OpenAI (router `gpt-5.4-mini`, generator `gpt-5.6-sol`, judge `gpt-5.5`).
  - `gpt-5.6-sol` was my pick among `gpt-5.6-luna|sol|terra` (user said only "5.6"); change `OPENAI_GENERATE_MODEL` in `.env` if they meant another.
  - Anthropic key in `.env` has **no credit** ("credit balance too low"). A Claude judge needs credit, then `JUDGE_PROVIDER=anthropic` and `python evals/run_evals.py --rejudge evals/reports/results-<ts>.json`.
- Secrets: `.env` (real file), `OpenAI API.txt` and `env-folder-original/` are in the project root and **git-ignored, never committed**. Move them out of the folder before zipping/submitting. `data/raw` is also git-ignored (re-fetch with `python ingest/download_raw.py`; two Quran Complex zips must be placed by hand, see README).
- Last full live eval (final build before the current work): router 94.2%, retrieval 100%, citations 100%, verse fidelity 100%, correct abstention 100%, false abstention 3.4% (b-09), behaviour 98.1%, verify mode 5/5, judge 4.76/5. One full run ~ 300k input / 40k output tokens.

## Current task: implementing the Opus review plan (uncommitted before this file; committed with it)
Cause found: the generator prompt told the model to write scope/referral sentences ("not exhaustive", "ask a scholar") which can never carry a citation, so the verifier rejected them -> abstentions on a-15, b-09, c-01, c-04. The retry was also blind (did not see its previous answer).

### DONE (in code, existing tests pass)
1. `core/verify.py`: code-owned `{{note:partial|refer|disputed}}` placeholder (fixed ar/en text in `NOTES`, rendered as a `notice` block, never counts as a citation; unknown note keys rejected); multi-id citations `[[a, b]]` parsed (`cite_ids`), malformed citations reported precisely; the "explanation without citation" error now includes a fix recipe and 200 chars of the rejected text.
2. `core/generate.py` rewritten: rule 2 says any listed id (incl. quran:/hadith:) can be cited and a placeholder is not a citation; rule 10 = use notes instead of writing disclaimers; format example; level ب/ج notes (scholarly classifications attributed, one-view -> `{{note:partial}}`); `build_prompts/generate(..., previous=raw)` puts the rejected answer verbatim in the retry prompt.
3. `core/pipeline.py`: `ask(question, lang, debug=False)` returns `out["debug"]` = route, retrieved_ids, per-attempt raw output + errors + ok, abstain_reason; retry passes `previous=raw`; level-د general-info cards no longer include hadith.

### LEFT TO DO (in this order)
1. **Tests for the above** (none written yet): note placeholder passes / unknown note fails / note-only answer fails "no citation" / free-text disclaimer still fails; `[[a, b]]` passes when both retrieved and fails if either is not; b-09 shape (`{{quran:3:45}}` then text then `[[quran:3:45]]`) passes and fails without the cite; second `generate` call's user prompt contains the first raw answer; `debug` trace present only when requested; د cards contain no hadith.
2. **Router** (`core/router.py`): escalate أ->ب when `_CONCEPT` matches or a new `_SENSITIVE` regex matches (جهاد|قتال|حدود|ردة|رق|jihad|apostasy|slavery) — escalation only; in `SYSTEM` say "what does X mean in Islam" is `ask` and `translate_term` only for an explicit request for a translation/English equivalent. Tests with a fake LLM returning أ for dp6-01, b-02, dp6-12.
3. **API** (`api/main.py`): never pass `debug`; strip `verification_errors` from `/api/ask` responses (they can contain model-written text).
4. **Eval runner** (`evals/run_evals.py`): call `pipeline.ask(q, debug=True)` once (today `run_one` routes twice and retrieval recall uses `DEFAULT_PER_TYPE`, not `PER_TYPE_ASK`) and take route + retrieved_ids from `debug`; store abstain_reason, attempts, referrals, trace and per-case token usage (diff of `L.USAGE`) in results (drop nothing in `_resp`); show the judge the referrals/cards; add metric "verifier-forced abstention" (`abstain_reason == verification_failed`, target 0); add `--reverify results.json` (re-run `verify_answer` on stored raw outputs against `get_passage(retrieved_ids)`, zero cost); optional `LLM_CACHE=1` (sha256 of provider+model+system+user+max_tokens+effort -> data/cache) wrapper in `core/llm.py`; fix the docstring that still says ANTHROPIC_API_KEY.
5. **Golden set** (`evals/golden.jsonl`): a-15 `answer_or_abstain` -> `answer`; add `must_retrieve_any` to b-09 (`quran:3:45`, `quran:19:30`) and dp6-01 (`quran:2:144`) — check with retrieval first; note on c-01/c-04/c-05 that they should present positions (abstain only via the model).
6. **Measure cheaply**: `python -m pytest -q`, then `python evals/run_evals.py --offline`, then
   `python evals/run_evals.py --only a-15,b-09,c-01,c-02,c-04,dp6-01,b-02,dp6-12,dp6-07,c-05 --no-judge` (expect a-15/b-09/c-01/c-04 answered with attempts<=2, 0 verifier-forced abstentions, c-02 shows a reason, dp6-01 and b-02 routed ب, dp6-12 ask), then `--rejudge` on that file, and only then one full run. Commit after each green step (`git add -A`; confirm `git ls-files | findstr /i ".env api.txt data/raw"` prints nothing).
7. **Docs**: README/CLAUDE.md: mention `{{note:...}}`, provider switching (`.env.example` already documents it), debug trace, the new eval metrics, and the final numbers.
8. **Local-model backend** (user will pick the model): `core/llm.py` already has `LLM_PROVIDER=local` using any OpenAI-compatible server (`LOCAL_BASE_URL`, `LOCAL_MODEL`; plain `max_tokens`/`temperature`, tested with stubs only). Still to do once a model is chosen: try it live, add small-context limits (cap passages/`max_chars` in `core/generate.py` via env), test Arabic JSON routing quality, run the evals on it. Rejected on purpose by the review: a third retry, auto-inserting citations in code, dropping uncited segments, relaxing the segment rule.

## Known gaps (documented, not fixed)
- One citation still covers a whole list/paragraph block (accepted).
- `fidelity_ok` is True for answers with no Quran block (it also counts leak checks).
- Judge and generator are both OpenAI (self-preference bias) until a Claude judge is possible.
- The 4 provisional viral-hadith eval cases (v-01..v-04) should be replaced with real examples from Dorar's widespread-hadith section.
