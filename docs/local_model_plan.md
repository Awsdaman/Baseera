# Plan: getting the most out of Gemma 4 12B (local) — written by Opus 5.5 from the 2026-10-04 live eval

Baseline (55 cases, Gemma 4 12B Q4_K_M, LM Studio): behaviour 96.4%, router 98.2%, citations 100%, verse fidelity 100%,
false abstention 6.2%, verifier-forced abstention 7.3% (target 0). OpenAI run: 100% / 0.

## What the raw data showed
1. The LLM router never ran: all 55 routes are `source=heuristic`. Router calls hit the `max_tokens=300` cap (hidden reasoning), JSON parsing failed, code silently fell back. The 98.2% is the regex router. canonical_question/claim were None everywhere.
2. Thinking is still on: `/no_think` is a Qwen switch, Gemma ignores it. 70-90% of generation tokens are reasoning. dp6-11 attempt 2 used all 6000 tokens and returned an empty visible answer.
3. dp6-03 is a citation-syntax problem, not English words: Gemma wrote `[[a], [b]]`; the CITE regex did not match, so ids were flagged as English words and the retry error was misleading.
4. Other Gemma slips: short ids `[[tafsir:2:255]]`, quotes/﴿﴾ around phrases, hadith wording copied into prose, placeholder followed by its own citation.
5. v-06: heuristic verify pattern does not know `سمعت النبي ... يقول`.
6. CachedLLM caches empty outputs (tainted replays).
7. LM Studio context shows 262144 whatever is requested (CLI and REST load ignored it) — see Step 1 note.

## Steps (ordered by impact)
0. Make the eval show the problems: store route source/router_error + extraction source per case; add metrics router_llm_rate, runtime_abstentions, first_attempt_pass_rate, gen_output_tokens_mean; warn when non-referral cases are tainted. CachedLLM: never cache empty output; add LOCAL_THINKING/LOCAL_MODEL to the key; clear the old local cache.
1. Thinking off + token budgets: probe which switch LM Studio honours (chat_template_kwargs enable_thinking=false, reasoning_effort, per-model UI toggle); `thinking="off"` for router/extractor; LLM_GEN_MAX_TOKENS (1500 local), LLM_ROUTER_MAX_TOKENS (600 local); raise LLMTruncated on finish_reason=length (also Anthropic stop_reason=max_tokens); local max_retries=1. Expect ~20-25 s/case, full run ~25 min. Context: unload, `lms load ... --context-length 16384 --identifier gemma-4-12b-it`, disable JIT loading in LM Studio dev settings, verify with `lms ps`.
2. `normalize_citations` in core/verify.py before verifying: rewrite `[[a], [b]]` to `[[a, b]]`, map short `tafsir:S:A` / verse ranges to retrieved full ids, move a placeholder's own trailing citation. Only bracket syntax; every id must still be retrieved. Add `،` to id splitters, a "Malformed citation" error, de-duplicate errors. Check free with `--reverify` (expect forced abstentions 3 -> 1). Unit tests required.
3. Empty/truncated output: one runtime retry (not a verifier attempt), then `llm_empty` / `llm_truncated` / `llm_error` reported as their own metrics.
4. Router: `_NARRATION` regex (adds `سمعت النبي ... يقول`, `رضي الله عنه`), a narration line in SYSTEM, "JSON only, one line"; optional LOCAL_JSON_SCHEMA=1 (probe first); EXTRACT_SYSTEM line about narrator chains.
5. Precise retry instructions: pass errors as a list; `fix_recipe(errors)` with an exact FIX per error type (hadith/verse wording, ornate brackets, unattributed quotes, missing citation, unretrieved id, malformed citation, English words). Optional: LLM_MAX_ATTEMPTS=3 for local; optional deletion-only repair behind a flag.
6. Prompt addendum for local models only: `LLM_PROMPT_PROFILE` (default "local" when LLM_PROVIDER=local), a CHECKLIST appended at the END of the user message (full ids, one pair of brackets, citation before the full stop, placeholders alone on a line, no quotes/﴿﴾, no Latin letters, BAD/GOOD pairs). Default prompts for Anthropic/OpenAI stay byte-identical (test). Promote clarifications to shared RULES only after an OpenAI `--only` regression run. Keep LLM_PASSAGE_CHARS=900 on the desktop (600 on the laptop).
7. Throughput: try LM Studio 2-3 concurrent predictions with `--workers 2`.

## Measurement
1. Free: `--reverify` + pytest. 2. Probe thinking switches and `lms ps`. 3. Smoke subset (fresh cache): dp6-03, dp6-11, c-02, v-06, a-01, a-02, a-03, a-07, a-10, a-14, b-01 — gate: router source=llm 11/11, 0 runtime abstentions, >=10/11 behaviour, median < 30 s.
4. Full 55 twice (take the worse). Hard gates: verse fidelity 100%, citations 100%, 0 model-written hadith grades, 0 rulings on level د, correct abstention 100%. Targets: behaviour >= 54/55, forced abstention <= 1/41, false abstention <= 1/32, LLM router accuracy >= 95%, first-attempt pass >= 80%, runtime abstentions 0, mean gen output <= 900 tokens, median <= 30 s. Then `--rejudge` with an OpenAI judge: avg >= 4.4.
5. Overfitting guard: freeze ~30 held-out answerable questions from icadb/bayyinat titles not in golden; run evals/router_check.py and evals/synth_verify.py; no golden-case topic words in prompts; shared RULES changes need an OpenAI regression run.

## Laptop (RTX 4050, 6 GB)
1. Best: thin client — LM Studio "Serve on Local Network" on the desktop, `LOCAL_BASE_URL=http://<desktop-ip>:1234/v1` on the laptop.
2. Offline: same Gemma 12B Q4_K_M with partial GPU offload (~30-36 layers), context 8192, LLM_PASSAGE_CHARS=600, LLM_GEN_MAX_TOKENS=1200, thinking off, 1 worker; expect 8-15 tok/s, 45-90 s/answer.
3. Small 4B-class model: needs its own full eval (more retries/abstentions; verifier keeps it safe, not helpful).
4. Last resort: retrieval-only mode, or OpenAI when online (cloud: the user's choice).

## Risks
Normalization could hide sloppiness (limited to bracket syntax + documented id mapping, ids still checked, trace records `normalized`). Thinking off may thin answers (judge gate; LOCAL_THINKING_GENERATE=on stays available). JSON schema may fail on Vulkan (flagged, fallbacks kept). Shared RULES edits invalidate the cloud cache (deferred).
