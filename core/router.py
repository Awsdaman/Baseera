"""Router (Haiku): {level, intent, language, term}. Code-level safety escalation always wins over the model.

level  أ basic sourced facts | ب explanation/concepts/doubts | ج disputed/sensitive | د personal ruling/case
intent ask | verify | translate_term
"""
import os
import re

from core import llm as L
from core.glossary import GLOSSARY, lookup
from core.normalize import normalize_ar

LEVELS = ("أ", "ب", "ج", "د")
INTENTS = ("ask", "verify", "translate_term")

SYSTEM = """You are the routing component of Baseera, an Islamic Q&A assistant. Classify the user message.
Return ONLY a JSON object: {"level": "أ|ب|ج|د", "intent": "ask|verify|translate_term", "language": "ar|en|other", "term": "<term or null>", "canonical_question": "<...>", "canonical_question_ar": "<...>", "claim": "<... or null>"}

Levels (content sensitivity):
- أ: stable basics: Quran, authentic hadith, pillars of Islam/faith, basic seerah, basic ethics, definitions.
- ب: explaining concepts, comparisons, objectives of Sharia (maqasid), general questions and common doubts/objections about Islam.
- ج: fiqh disagreements between schools, detailed aqeedah disputes, contested historical issues, questions that need specialist scholarly treatment, "do all Muslims agree on this?".
- د: a personal situation or ruling request about a specific person's case: validity of a particular contract/worship/marriage, family disputes, legal or medical matters with religious impact, "am I allowed to ... in my marriage / in my country".
Intents:
- verify: the user pastes a message that CONTAINS a Quran verse, a hadith or a saying attributed to the Prophet / a companion / a scholar, and wants that text checked. A question about whether a RULING or STATEMENT is correct (e.g. is "the ruling on X is obligatory" right?) is NOT verify: it is an ask about the topic.
- translate_term: ONLY an explicit request to translate a term or give its English equivalent (set "term"). "What does X mean in Islam?" is a plain ask, not translate_term.
- ask: everything else.
A hostile or accusatory question about Islam is still level ب (answer wisely), not د.
Language is the language of the message itself.
canonical_question: rewrite the message as ONE clear, self-contained question in the SAME language, naming the topic and what is asked, so a search engine can find the sources.
  Example: 'هل هذا الحكم "حكم أذكار الصباح واجبة" صحيح؟'  ->  'ما حكم أذكار الصباح، وهل هي واجبة؟'.  Keep an already-clear question as it is.
canonical_question_ar: the approved sources are Arabic. When the message is NOT Arabic, write the same self-contained question in Arabic (Islamic terms in their standard Arabic form, e.g. wudu = الوضوء, zakah = الزكاة, tawaf = الطواف) for searching; when the message is already Arabic, null.
claim: if the user quotes or asserts a statement and asks whether it is correct, put that statement here (without the surrounding question); otherwise null.
A message that only reports a narration with no question, e.g. 'عن فلان رضي الله عنه قال: سمعت النبي ﷺ يقول: «...»' or 'قال رسول الله ﷺ: «...»', is verify even without words like "تحقق" or "صحيح؟".
Output the JSON object only, on one line, with nothing before or after it."""

AR = "ء-ي"   # a standalone Arabic word: not glued to other letters (so "ابي" never matches inside "الصحابي")
_PERSONAL = re.compile(
    r"(ما حكم (?:زواجي|طلاقي|عقدي)|في زواجي|زوجتي|زوجي|طلقت|طلاقي|عقد(?:ي| العمل| الايجار)|انا في دوله|أنا في دولة|حالتي|مشكلتي|"
    r"(?<![" + AR + r"])(?:ابي|أبي) يرفض|اختلفت مع|"
    r"(?<![" + AR + r"])(?:انا|أنا) (?:مريض|مريضه|مريضة|حامل|مصاب|مصابه|مصابة|مدين|متزوج|متزوجه|متزوجة|مطلق|مطلقه|مطلقة)|"
    r"(?:عندي|لدي|اعاني من|أعاني من) (?:السكر|سكر|ضغط|مرض|سرطان|امراض|أمراض|فشل|ربو|صرع))")
# generic first-person openers are personal only together with a strong personal marker (see strong_personal)
_PERSONAL_OPENER = re.compile(r"(هل يجوز لي|هل يحق لي|هل يصح لي)")
_PERSONAL_EN = re.compile(
    r"\b(my (?:wife|husband|marriage|divorce|contract|landlord|boss|father|mother|case|medication|treatment|doctor)|"
    r"i live in|i am in (?:the )?[a-z]+ and|is my (?:marriage|contract|prayer|fast) valid|"
    r"i (?:have|am|was) (?:diabetes|diabetic|pregnant|sick|ill|diagnosed|on medication|taking medication)|i have (?:cancer|asthma|epilepsy|high blood pressure))\b", re.I)
_PERSONAL_OPENER_EN = re.compile(r"\b(am i allowed|can i|should i|may i)\b", re.I)

# Level د needs a real personal case: a family / marriage / money-contract / legal / medical situation OF THE ASKER, or an explicit "my case".
# A model router sometimes sends ordinary worship-practice questions ("a person forgot a pillar of Hajj, what must he do?", "does an injection
# break the fast?") to د, which would hide a perfectly answerable general question behind a referral. Such a د is demoted to ج (attributed
# positions + a scholar-referral note, never a personal ruling) unless the text shows a strong personal marker.
_STRONG_PERSONAL = re.compile(
    r"(زوج|طلاق|طلقت|خلع|ميراث|ورثه|وصيه|عقد|شركه|قرض|ديني|راتبي|وظيفتي|مديري|كفيلي|محكمه|قضيتي|سجن|حامل|حملي|ادويتي|طبيبي|مرضي|ابني|ابنتي|امي|(?<![ء-ي])ابي(?![ء-ي])|اخي|اختي|جاري|حالتي|مشكلتي|عندي|لدي|(?<![ء-ي])انا (?:مريض|حامل|مصاب|مدين|متزوج|مطلق)|(?:اعاني|اعاني) من)")
_STRONG_PERSONAL_EN = re.compile(r"(my (?:wife|husband|marriage|divorce|contract|landlord|boss|father|mother|son|daughter|brother|sister|case|doctor|job|salary)|i live in|am i allowed)", re.I)


def strong_personal(text: str) -> bool:
    return bool(_STRONG_PERSONAL.search(normalize_ar(text or "")) or _STRONG_PERSONAL_EN.search(text or ""))


_DISPUTE = re.compile(r"(اختلاف(?: العلماء| الفقهاء)?|ترجيح|المذاهب|اصح الاقوال|هل كل المسلمين|do all muslims agree|differ(?:ence|ent)? (?:between|among) (?:the )?(?:scholars|schools))", re.I)
_CONTESTED = re.compile(r"(البسمله (?:ايه|اية|من الفاتحه)|هل البسمله|الراجح|ايهما علي حق|الشيعه|معاويه|صفين|وقعه الجمل|المولد النبوي|الموسيقي|المعازف|الغناء|تصوير ذوات|تكفير|الخلاف بين)")
_BASIC = re.compile(r"(ما معني (?:ايه|اية|سوره|حديث|قوله تعالي|لا اله)|ما فضل|ما هي (?:سوره|اركان|الصلوات)|كم عدد|من هو النبي|اركان|ما هي سوره)")
_SENSITIVE = re.compile(r"(جهاد|قتال|الحدود|الرده|الرق|الاسترقاق|jihad|apostasy|slavery)")
_CONCEPT = re.compile(r"(مقاصد|الحكمه|لماذا|هل الاسلام|هل يتعارض|هل يظلم|ارهاب|عنف|تاليف (?:محمد|النبي|بشر)|(?:القران|الوحي) (?:مختلق|مفتري|من صنع)|do muslims|why do|is islam|did (?:muhammad|the prophet) (?:write|invent|make up)|(?:man-?made|written by (?:muhammad|a man)))")
_VERIFY = re.compile(r"(تحقق|هل هذا الحديث صحيح|هل هذه الايه|صحيح ام|fact.?check|is this (?:hadith|verse)|verify|authentic\?|ارسل لي|وصلني)", re.I)
_NARRATION = re.compile(r"(قال رسول الله|قال النبي|عن النبي|قال تعالى|سمعت (?:رسول الله|النبي)|(?:رسول الله|النبي)\s*(?:ﷺ|صلى الله عليه وسلم)?\s*(?:يقول|قال)|رضي الله عن(?:ه|ها|هما|هم))")
_TRANSLATE = re.compile(r"(ترجم|ترجمه|ما معنى كلمه .* بالانجليزيه|translate|how (?:do you|to) say|english (?:word|equivalent|for))", re.I)


def detect_language(text: str) -> str:
    ar = len(re.findall(r"[؀-ۿ]", text))
    la = len(re.findall(r"[A-Za-z]", text))
    if ar == 0 and la == 0:
        return "other"
    return "ar" if ar >= la else "en"


def find_term(text: str):
    t = normalize_ar(text)
    for g in GLOSSARY:
        if normalize_ar(g["ar"]) in t.split() or normalize_ar(g["ar"]) in t:
            return g["ar"]
    for tok in re.findall(r"[a-zA-Z'’‘]+", text):
        g = lookup(tok.lower())
        if g:
            return g["ar"]
    return None


def heuristic_route(text: str) -> dict:
    t = normalize_ar(text)
    lang = detect_language(text)
    level, intent, term = "ب", "ask", None
    if (_PERSONAL.search(text) or _PERSONAL_EN.search(text)
            or ((_PERSONAL_OPENER.search(text) or _PERSONAL_OPENER_EN.search(text)) and strong_personal(text))):
        level = "د"
    elif _DISPUTE.search(text) or _CONTESTED.search(t):
        level = "ج"
    elif _CONCEPT.search(t) or _CONCEPT.search(text.lower()):
        level = "ب"
    elif _BASIC.search(t) or (re.search(r"(ما هو|ما هي|من هو|what is|what are|who is|pillars)", text, re.I) and len(text) < 90):
        level = "أ"
    if _TRANSLATE.search(t) or _TRANSLATE.search(text):
        intent, term = "translate_term", find_term(text)
        level = "أ"
    elif ((_VERIFY.search(text) and has_claim_text(text)) or ("﴿" in text and not re.search("[؟?]", text)) or len(text) > 400
          or (_NARRATION.search(text) and re.search("[«\"“﴿]", text) and not re.search("[؟?]", text))):
        intent, level = "verify", "ب"
    return {"level": level, "intent": intent, "language": lang, "term": term, "source": "heuristic"}


_QUOTED = re.compile("[«»\"“”﴿﴾]|قال رسول الله|قال النبي|قال تعالى|قال الله|عن النبي")


def has_claim_text(text: str) -> bool:
    """A verify request needs something to verify: a quotation, an attributed saying, or a long pasted message."""
    return bool(_QUOTED.search(text)) or len(text) >= 80


_CLAIM_Q = re.compile(r"هل\s+(?:هذا\s+)?(?:الحكم|القول|الكلام|الكلام التالي)\s*[\"«“'](.+?)[\"»”']\s*(?:صحيح|صحيحه|خطأ|غلط|دقيق|ثابت)")


def extract_claim(text: str):
    """Heuristic: 'is this ruling "X" correct?' -> X (used when the model router is unavailable or returned nothing)."""
    m = _CLAIM_Q.search(text)
    return m.group(1).strip() if m else None


# "Give me a hadith that proves THIS": a request that points at a statement the user never pasted. Searching would only retrieve
# unrelated texts (and show them), so the pipeline asks for the statement instead. Deliberately narrow: request words + demonstrative only.
_DANGLING = re.compile(
    r"^(?:(?:اعطني|اعطيني|هات|اريد|ابي|ابغي|اذكر لي|اعطنا|هل يوجد|هل هناك)\s+)?(?:حديثا|حديث|دليلا|دليل|ايه|اية|مصدرا|مصدر|برهانا)?\s*"
    r"(?:(?:ال)?(?:يثبت|يدل علي|يؤيد|يويد|يؤكد|يوكد|يبين|يصحح|يثبتان)|الذي يثبت|الذي يدل علي)\s*(?:هذا|هذه|ذلك|تلك)\s*(?:الكلام|القول|الحكم|الامر|الادعاء|الشيء|الموضوع)?$")
_DANGLING_EN = re.compile(r"^(?:please\s+)?(?:give me|show me|find me|can you give me|is there)\s+(?:a|an|any)?\s*(?:hadith|verse|evidence|proof|source)\s+(?:that\s+)?(?:proves?|supports?|confirms?|shows?)\s+(?:this|that|it)(?:\s+(?:statement|claim|ruling))?$", re.I)


def needs_context(text: str) -> bool:
    """True when the message asks for evidence of 'this / that' without containing the statement."""
    t = normalize_ar(text or "").strip(" .؟?!،,؛;:")
    t = re.sub(r"\s+", " ", t)
    return bool(_DANGLING.fullmatch(t) or _DANGLING_EN.fullmatch(t))


# First-person distress cues ("I feel hopeless", "I committed a sin and regret it"): the answer opens with a code-owned line that
# acknowledges the person before any information. Fixed text we wrote, so it needs no citation and the model never has to write it.
_EMPATHY_AR = re.compile(r"(اشعر|احس |احسست|شعرت|اعاني|عانيت|تعبت|ندمت|نادم|خايف|اخاف|يائس|حزين|مكتئب|قلق|وحيد|ضائع|ضايع|ذنوبي|ذنبي|ارتكبت|ابتليت|لا استطيع التوقف|مهموم|مخنوق)")
_EMPATHY_EN = re.compile(r"\b(i feel|i am (?:so |very )?(?:sad|depressed|anxious|scared|lost|guilty|struggling|hopeless)|i'?m (?:so |very )?(?:sad|depressed|anxious|scared|lost|guilty|struggling|hopeless)|i(?:'ve| have) been struggling|i committed|my sins?|i regret|hopeless)\b", re.I)


def needs_empathy(text: str) -> bool:
    return bool(_EMPATHY_AR.search(normalize_ar(text or "") + " ") or _EMPATHY_EN.search(text or ""))


def _clean(v, limit=400):
    v = v.strip() if isinstance(v, str) else ""
    return v[:limit] if v and v.lower() not in ("null", "none") else None


def route(text: str) -> dict:
    h = heuristic_route(text)
    claim_h = extract_claim(text)
    h["claim"], h["canonical_question"] = claim_h, (claim_h + "؟") if claim_h else None
    if not L.llm_available():
        return h
    try:
        with L.no_thinking():  # short JSON: a local reasoning model must not burn the token budget on hidden thinking
            raw = L.get_llm().complete(L.ROUTER_MODEL, SYSTEM, text, max_tokens=int(os.environ.get("LLM_ROUTER_MAX_TOKENS", 300)))
        j = L.extract_json(raw)
        level = j.get("level") if j.get("level") in LEVELS else h["level"]
        intent = j.get("intent") if j.get("intent") in INTENTS else h["intent"]
        out = {"level": level, "intent": intent, "language": j.get("language") or h["language"],
               "term": j.get("term") if j.get("term") not in (None, "null", "") else h["term"], "source": "llm",
               "canonical_question": _clean(j.get("canonical_question")) or h["canonical_question"],
               "claim": _clean(j.get("claim")) or claim_h,
               "search_ar": _clean(j.get("canonical_question_ar")) if re.search("[؀-ۿ]", j.get("canonical_question_ar") or "") else None}
    except Exception as e:  # never fail the request because routing failed
        h["router_error"] = str(e)[:200]
        return h
    # Safety: code can only escalate the model's level (to د for personal cases, to ج for known contested topics).
    if h["level"] == "د":
        out["level"] = "د"
    elif out["level"] == "د" and not strong_personal(text):
        out["level"], out["demoted_from"] = "ج", "د"  # a general worship / fiqh question, not a personal case: answer with attribution + referral note
    elif _CONTESTED.search(normalize_ar(text)) and out["level"] in ("أ", "ب"):
        out["level"] = "ج"
    elif out["level"] == "أ" and (_CONCEPT.search(normalize_ar(text)) or _CONCEPT.search(text.lower())
                                  or _SENSITIVE.search(normalize_ar(text)) or _SENSITIVE.search(text.lower())):
        out["level"] = "ب"  # explanation/doubt/war-and-penal topics are never "stable basics"
    # A pasted message that attributes a quoted saying to Allah / the Prophet and asks no question is a verify request, whatever the
    # model said (the model's intent is not stable on these; found in the live evals). The reverse safety net is in the pipeline:
    # a verify request that finds nothing to verify is answered as a question.
    if out["intent"] == "ask" and h["intent"] == "verify" and not re.search("[؟?]", text) and has_claim_text(text):
        out["intent"] = "verify"
    # "verify" needs text to verify; "give me a hadith that proves this" has none -> it is an ask (and will abstain).
    if out["intent"] == "verify" and not has_claim_text(text):
        out["intent"] = "ask"
    if h["intent"] == "translate_term" and out["intent"] == "ask" and h["term"]:
        out["intent"], out["term"] = "translate_term", h["term"]
    return out
