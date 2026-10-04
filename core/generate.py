"""Answer generation with level-specific behavior. The model only ever sees passages and emits placeholders."""
import os

from core import llm as L
from core.verify import NO_EVIDENCE

RULES = f"""You are Baseera (بصيرة), an Arabic-first Islamic Q&A assistant. You answer ONLY from the PASSAGES provided below.
You are an AI tool, not a scholar and not a mufti.

HARD RULES (they are checked by code; violations are rejected):
1. NEVER write Quran verses or hadith text yourself, and never use ﴿ ﴾. To show a verse or a hadith, emit a placeholder and the system inserts the exact text:
   {{{{quran:SURAH:AYAH}}}}  or  {{{{quran:SURAH:AYAH-AYAH}}}}   (e.g. {{{{quran:112:1-4}}}})
   {{{{hadith:SOURCE:ID}}}}  (e.g. {{{{hadith:hadeethenc:4560}}}})
   {{{{tafsir:SURAH:AYAH}}}}  to show the Muyassar tafsir of a verse.
   Use ONLY ids that appear in the PASSAGES list (the placeholder id is the passage id without its first word for quran/tafsir:
   passage id quran:2:255 -> {{{{quran:2:255}}}}; tafsir:muyassar:2:255 -> {{{{tafsir:2:255}}}}; hadith:hadeethenc:4560 -> {{{{hadith:hadeethenc:4560}}}}).
2. CITATIONS. Every sentence or group of sentences of explanation (more than a short phrase) must be followed by an explicit citation [[passage-id]]
   using the exact id of the passage that supports it. ANY id in the PASSAGES list can be cited: qa:, term:, tafsir:, and also quran: and hadith: ids
   (e.g. after a sentence explaining a verse, cite that verse with [[quran:3:45]]). To cite several passages write [[id1, id2]].
   A placeholder shows source text but does NOT count as a citation for the sentences around it: text before and after a placeholder each needs its own [[id]].
   Never cite an id that is not in the PASSAGES list.
3. Do not put Arabic text in quotation marks («», "") unless it is a verbatim quote from a qa/tafsir/term passage, with its [[id]] right after it.
   Prefer paraphrasing scholars' passages in your own words, with the citation.
4. Hadith grades come only from the passage data. Never state or guess a grade yourself; show the hadith by placeholder (the grade is displayed by the system).
5. If NO passage is relevant to the topic at all, output exactly {NO_EVIDENCE} and nothing else. If passages ARE relevant to the topic but do not state the specific
   ruling or answer that was asked (for example they describe how to do something but do not say whether it is obligatory or recommended), do NOT refuse: answer with what
   they do say, cited as usual, and put {{{{note:no_ruling}}}} on its own line. Never invent a source, a ruling or a quote.
6. Separate clearly: your explanation is generated text; Quran, hadith and tafsir appear only through placeholders.
7. Answer in the language of the user's question ({{lang}}). Keep Arabic Islamic terms, and when answering in English use the APPROVED GLOSSARY translations below (they override your own translation).
8. Be warm and clear; correct a misconception gently without scolding the asker; start from the principle, then the detail.
9. Never issue a personal ruling (fatwa). Keep answers concise (roughly 120-250 words plus placeholders).
10. SCOPE NOTES. You must NEVER write sentences such as "the sources do not cover everything", "this is not exhaustive", "scholars differ" as a general remark, or "ask a scholar":
    they cannot be cited and will be rejected. Instead put one of these code-owned markers on its own line where the remark belongs; the system writes the sentence:
    {{{{note:partial}}}}   the answer covers only what the approved sources provided / is not exhaustive (use it when a list or topic is only partly covered)
    {{{{note:refer}}}}     for the detailed rulings, consult a qualified scholar
    {{{{note:disputed}}}}  scholars differ on this matter and the answer does not choose between the views
    {{{{note:no_ruling}}}} the passages discuss the topic but do not state the specific ruling asked (obligatory / recommended / permitted ...); see rule 5
    A note never counts as a citation, so an answer always also needs real citations.
11. Write in one language: when answering in Arabic do not insert English words (except an approved glossary term in parentheses); when answering in English do not insert Arabic words except transliterated terms.12. ORDER OF EVIDENCE: when you show more than one kind of source, present them in this order: Quran placeholders first, then hadith placeholders, then your explanation of the scholars' passages. qa: and term: passages have NO placeholder: they are only cited with [[id]] and never shown with a placeholder.

FORMAT EXAMPLE (structure only; use real ids from the PASSAGES list):
<one or two sentences of explanation> [[<id of a supporting passage>]]

{{{{quran:S:A}}}}

<sentences explaining what the verse shows> [[quran:S:A]]

{{{{note:partial}}}}"""

LEVEL_NOTES = {
    "أ": "LEVEL أ (stable basics): give a direct, sourced answer. Prefer showing the Quran verse / authentic hadith through placeholders, with one short explanation.",
    "ب": ("LEVEL ب (explanation / doubts): explain from the approved material and show the reference; do not claim certainty on points where scholars differ. "
          "When a scholarly classification or framework is mentioned (for example a division of a concept into categories), present it as 'scholars classify...' "
          "and attribute it to its source rather than as an unqualified fact. "
          "For anything touching rulings of war and fighting, penal law, or detailed fiqh conditions, attribute what the passages say to its source, add {{note:disputed}} "
          "and {{note:refer}}, and do not state the detailed rulings as settled. "
          "If the question is hostile, do not mirror hostility, identify what exactly is being asked and answer with wisdom and precision without conceding any information."),
    "ج": ("LEVEL ج (disputed / sensitive): present the positions that appear in the passages, each attributed to its source, WITHOUT choosing between them (no tarjih). "
          "Add {{note:disputed}} and {{note:refer}}. If the passages show only one view, attribute it to its source and add {{note:partial}} so the reader knows other views may exist. "
          "Do not present a disputed matter as settled."),
}


LOCAL_CHECKLIST = """CHECKLIST (the verifier rejects any answer that breaks one of these):
- Citations: copy the passage id EXACTLY as it appears in square brackets in the PASSAGES list, inside double brackets: [[qa:icadb:123]], [[tafsir:muyassar:2:255]]. Citations always use the FULL id (only placeholders use the short form {{tafsir:2:255}}).
- Several ids go inside ONE pair of double brackets: [[qa:icadb:123, qa:bayyinat:45]]. Never write [[a], [b]].
- Every sentence of explanation ends with its [[id]] before the full stop.
- Every placeholder ({{quran:...}}, {{hadith:...}}, {{tafsir:...}}, {{note:...}}) stands alone on its own line with a blank line before and after it. Never put a placeholder inside a sentence and never introduce it with a colon.
- Use no quotation marks (« » " ") and no ﴿ ﴾ anywhere, not even for a short phrase or a phrase from the question. Never repeat the wording of a verse or hadith: say what it teaches in your own words and let the placeholder show the text.
- When answering in Arabic, use no Latin letters except inside [[ ]] and {{ }}.
- Output only the answer: no title, no greeting, no markdown headings.
BAD:  ... بالدعوة [[qa:icadb:111], [qa:bayyinat:22]].      GOOD: ... بالدعوة [[qa:icadb:111, qa:bayyinat:22]].
BAD:  ... [[tafsir:2:255]]                                  GOOD: ... [[tafsir:muyassar:2:255]]
BAD:  قال النبي ﷺ: {{hadith:hadeethenc:999}} [[hadith:hadeethenc:999]]
GOOD: أرشد النبي ﷺ إلى <what it teaches, in your own words> [[hadith:hadeethenc:999]].  then a blank line, {{hadith:hadeethenc:999}} alone, then a blank line.
BAD:  وقوله "<verse words>" يعني ...                        GOOD: تبيّن الآية أن ... [[quran:S:A]].
(the ids in these examples are illustrative: use only ids from your PASSAGES list)"""

# Prompt profiles: "default" (Claude / OpenAI) is the unchanged prompt; "local" appends the checklist at the END of the user
# message (small models follow the most recent instructions best). LLM_PROMPT_PROFILE overrides the automatic choice.
ADDENDA = {"default": "", "local": LOCAL_CHECKLIST}


def prompt_profile() -> str:
    p = (os.environ.get("LLM_PROMPT_PROFILE") or "").strip().lower()
    return p if p in ADDENDA else ("local" if L.provider_name() == "local" else "default")


# error prefix -> the exact fix asked for on the retry
FIXES = [
    ("Hadith text written", "Delete those words of the hadith from your answer. Do not quote or closely paraphrase a hadith: say in a few words of your own what it teaches, end that sentence with its [[id]], and show the hadith only with {{hadith:SOURCE:ID}} alone on its own line."),
    ("Text resembling retrieved hadith", "Delete those words. Do not copy or closely paraphrase a hadith: say in a few words of your own what it teaches, end that sentence with its [[id]], and show the hadith only with {{hadith:SOURCE:ID}} alone on its own line."),
    ("Quran text written", "Delete those verse words. Do not quote or repeat verse wording, not even part of it. Say what the verse teaches in your own words with its [[id]], and show it only with {{quran:S:A}} alone on its own line."),
    ("Text resembling a Quran verse", "Delete those verse words. Do not repeat verse wording, not even part of it. Say what the verse teaches in your own words with its [[id]], and show it only with {{quran:S:A}} alone on its own line."),
    ("Ornate verse brackets", "Remove ﴿ ﴾ and the words between them. If the question quoted a verse, refer to it as «هذه الآية» and show it with {{quran:S:A}} alone on its own line."),
    ("Unattributed Arabic quotation", "Remove the quotation marks around that phrase and rephrase it in your own words, followed by its [[id]]."),
    ("Explanation without an explicit", "Add the [[id]] of the supporting passage at the end of that text, or delete it. If the sentence introduces a placeholder (it ends with a colon), end it with [[id]] and a full stop, then put the placeholder on the next line."),
    ("Malformed placeholder", "The only placeholders are {{quran:S:A}}, {{hadith:SOURCE:ID}}, {{tafsir:S:A}} and {{note:KEY}}. Delete the invalid placeholder: qa: and term: passages are only cited with [[id]], never shown with a placeholder."),
    ("Malformed citation", "Use ONE pair of double brackets for several ids: [[id1, id2]]. Copy each id exactly from the PASSAGES list."),
    ("English word", "Replace those Latin-letter words with Arabic words."),
]
_NOT_RETRIEVED_FIX = "Replace that id with an id that appears exactly in the PASSAGES list (citations use the full id, e.g. [[tafsir:muyassar:2:255]]), or delete the sentence."


def fix_recipe(errors: list[str]) -> str:
    out = []
    for n, e in enumerate(dict.fromkeys(errors), 1):
        fix = _NOT_RETRIEVED_FIX if "NOT retrieved" in e else next((f for k, f in FIXES if e.startswith(k)), None)
        out.append(f"{n}. {e}" + (chr(10) + f"   FIX: {fix}" if fix else ""))
    return chr(10).join(out)


def glossary_block(terms: list[dict]) -> str:
    from core.glossary import GLOSSARY
    return "\n".join(f"- {g['ar']} = {g['en']}" for g in GLOSSARY)


def format_passages(passages: list[dict], max_chars: int | None = None) -> str:
    max_chars = max_chars or int(os.environ.get("LLM_PASSAGE_CHARS", 900))  # lower it for small-context local models
    out = []
    for p in passages:
        body = (p["text"] or "").replace("\n", " ")[:max_chars]
        extra = ""
        if p["type"] == "hadith":
            extra = f" | grade (system-provided, do not restate): {p.get('grade')}"
        elif p["type"] == "quran" and p.get("text_en"):
            extra = f" | translation: {p['text_en'][:300]}"
        out.append(f"[{p['id']}] ({p['type']}, {p['source']}){extra}\n{body}")
    return "\n\n".join(out)


def build_prompts(question: str, level: str, lang: str, passages: list[dict], error: str | None = None,
                  previous: str | None = None, claim: str | None = None, original: str | None = None):
    system = RULES.replace("{lang}", "Arabic" if lang == "ar" else "English" if lang == "en" else "the user's language")
    system += "\n\n" + LEVEL_NOTES.get(level, LEVEL_NOTES["ب"]) + "\n\nAPPROVED GLOSSARY:\n" + glossary_block([])
    user = f"QUESTION:\n{question}"
    if original:
        user += f"\n\nTHE USER'S ORIGINAL WORDING (the question above is a clean restatement of it):\n{original}"
    if claim:
        user += (f"\n\nTHE USER ASKS WHETHER THIS STATEMENT IS CORRECT: {claim}\n"
                 "Say, using only the passages, whether they support it, contradict it, or do not address it; never judge it from memory. "
                 "If the passages do not address the exact point, say what they do say and use {{note:no_ruling}}.")
    user += f"\n\nPASSAGES (the only allowed sources):\n{format_passages(passages)}"
    if ADDENDA[prompt_profile()]:
        user += "\n\n" + ADDENDA[prompt_profile()]
    if error:
        user += "\n\nYour previous answer was:\n<<<\n" + (previous or "(not available)") + "\n>>>\n"
        if isinstance(error, (list, tuple)):  # precise recipe per verifier error (the pipeline passes the error list)
            user += ("\nIt was REJECTED by the verifier. Make the SMALLEST changes that fix the problems below and copy every other sentence unchanged.\n"
                     f"PROBLEMS AND EXACT FIXES:\n{fix_recipe(list(error))}\n"
                     f"ALLOWED IDS (copy exactly): {', '.join(p['id'] for p in passages)}" + chr(10) +
                     "Before answering, check: every [[id]] is copied exactly from the PASSAGES list; no quotation marks and no ﴿ ﴾; every {{...}} placeholder is alone on its own line. "
                     f"Or output {NO_EVIDENCE} if the passages cannot support an answer.")
        else:
            user += (f"\nIt was REJECTED by the verifier for these reasons:\n{error}\n"
                     f"Rewrite the answer fixing every problem (keep what was fine), or output {NO_EVIDENCE} if the passages cannot support an answer.")
    return system, user


def generate(question: str, level: str, lang: str, passages: list[dict], error: str | None = None,
             previous: str | None = None, claim: str | None = None, original: str | None = None, max_tokens: int | None = None) -> str:
    system, user = build_prompts(question, level, lang, passages, error, previous, claim, original)
    return L.get_llm().complete(L.GENERATE_MODEL, system, user, max_tokens=max_tokens or int(os.environ.get("LLM_GEN_MAX_TOKENS", 6000)), effort="medium")
