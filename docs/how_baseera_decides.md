# How Baseera decides: sources, citations, and when it answers, qualifies, or declines

One page for reviewers. Every statement here is enforced by code in this repository and covered by tests, not only requested in a prompt.

## 1. Sources (approved, local, traceable)

| Source | Used for | Where it lives |
|---|---|---|
| King Fahd Complex Mushaf (Hafs v3.0) | Quran text: ground truth, 6,236 verses | local DB |
| KFGQPC *Tafseer Muyassar* | Arabic tafsir, verse level (shown separately from the Quran text) | local DB |
| HadeethEnc (ar + en) | Hadith text and **grades** (credited in the UI) | local DB |
| icadb Q&A and terminology | Scholarly Q&A, approved term translations | local DB |
| Bayyinat Q&A (263 entries) | Doubts and objections | local DB |
| Dorar al-Saniyyah (live API, cached) | Hadith grades for Verify mode | cached |
| icadb books (about 335 of the organizers' 345), Q&A, dictionary of Quranic words, notable people, places, sects, Names of Allah | Scholarly explanations and reference cards | local DB (books: keyword search) |
| Dorar al-Saniyyah Fiqh Encyclopedia (2,489 articles) | Positions of the schools with references | local DB, links back to dorar.net |
| Challenge glossary (data pack) | Approved term translations, override machine translation | code |

Nothing else is searched: every source is on the organizers' approved list (`docs/approved_sources_links.txt`; the audit is in `docs/sources_audit.md`). Referral links (islamqa.info, binbaz.org.sa, binothaimeen.net) are only shown as links, never scraped. We did not add sources to cover more fiqh: coverage of detailed practical fiqh is limited, and Baseera says so instead (section 3).

## 2. How a citation works (the model never writes scripture)

1. The question is routed (level أ/ب/ج/د, intent) and the approved sources are searched (keyword + semantic, fused).
2. The model sees only the retrieved passages, each with an id such as `qa:icadb:26239`, `quran:2:255`, `hadith:hadeethenc:4560`.
3. It may only emit **placeholders** (`{{quran:2:255}}`, `{{hadith:hadeethenc:4560}}`, `{{tafsir:2:255}}`) and **citations** (`[[qa:icadb:26239]]`).
4. **Pure code** (`core/verify.py`, no model) then rejects the answer if: a cited or placeholder id was not retrieved for this question; the model typed Quran or hadith words itself (fuzzy scan against the whole Mushaf and the retrieved hadith; for English answers also against the English translations of the retrieved verses and hadiths); text is in Quran brackets or in unattributed quotation marks; any 8+ word stretch has no citation; Latin words appear in an Arabic answer.
5. A rejected answer gets one retry that names each error and its exact fix; a second rejection becomes an honest "not enough evidence" reply with referral links. Nothing unverified is ever shown.
6. On acceptance, code substitutes the **exact stored text** for each placeholder, numbers the citations, and the UI shows Quran, hadith (with grade), tafsir, and the generated explanation in separate blocks. Hadith grades come only from HadeethEnc / Dorar, never from the model.

## 3. When Baseera answers, qualifies, or declines

| Situation | What happens | Enforced by |
|---|---|---|
| Sources state the answer | Sourced answer with citation cards | verifier |
| Sources discuss the topic but not the exact point | Answer from what they say + a fixed note "the sources do not state the specific ruling" | `{{note:no_ruling}}` (prompt rule 5) |
| Cited passages do not address the question | Same answer + fixed note "the passages do not address your question directly" | on-topic check (`core/relevance.py`) |
| Cited passages are unrelated **and** semantically far from the question | Declines, with referral links (`sources_not_on_topic`) | on-topic check, two independent signals |
| Scholars differ | Positions with attribution, no tarjih, fixed "scholars differ" and "ask a scholar" notes (level ج) | level rules + notes |
| A personal case (marriage, contract, family dispute, medical) | General information + referral only, never a ruling (level د) | router rules (code can only escalate) |
| A request to prove "this" with nothing to prove | Asks for the statement; shows no unrelated text | `core/router.py` |
| Verse or hadith pasted for checking | Verified / misquoted (correct text + word diff) / graded / not found | Verify mode |
| Model or server fails, empty or truncated reply | One retry, then a clear "service problem" decline, counted separately | pipeline |

Scope notes are **code-owned text**: the model asks for a note by key and the system writes the sentence, so the model never has to write an uncitable "ask a scholar".

## 4. Privacy and honesty

An on-screen line states that Baseera is an AI tool and not a scholar. No personal data is stored; only an explicit, opt-in problem report is kept (scrubbed, 90-day retention), and nothing learns automatically.

## 5. Measured results (local Gemma 4 12B, final sources, 2026-10-06)

| Suite | Result |
|---|---|
| Golden set, 55 cases (`evals/run_evals.py`) | behaviour 100% (55/55), verse fidelity 100%, citations 100%, correct abstention 7/7, retrieval 14/14, false abstention 0/32, forced abstention 0/41, judge 4.89/5 (final build) |
| Reliability, 51 questions (`evals/reliability.py`) | should-answer 17/17, no-source 13/13 declined or qualified, disputed 6/6, paraphrase outcome 5/5 |
| 82 IslamQA questions in English (`evals/bulk_check.py`) | real sourced answers 41 before the Arabic search step and the new sources, 59 after; declined 34 to 16; every answered "don't know"-tier question carries a referral or limits note |
| Synthetic Verify mode (`evals/synth_verify.py`) | verse misquote/fabrication macro-F1 about 0.98, hadith matching F1 0.96 (2,856 verse and 1,000 hadith cases) |
| Unit and integration tests | 291 pass |

## 6. How it is measured

`evals/run_evals.py` (55-case golden set), `evals/reliability.py` (answer boundaries: should-answer, no-source, disputed, paraphrase consistency), `evals/bulk_check.py` (full answers to any question file), `evals/synth_verify.py` (thousands of synthetic misquotes). Tests: `python -m pytest -q`.
