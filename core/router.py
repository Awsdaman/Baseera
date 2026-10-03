"""Router (Haiku): {level, intent, language, term}. Code-level safety escalation always wins over the model.

level  أ basic sourced facts | ب explanation/concepts/doubts | ج disputed/sensitive | د personal ruling/case
intent ask | verify | translate_term
"""
import re

from core import llm as L
from core.glossary import GLOSSARY, lookup
from core.normalize import normalize_ar

LEVELS = ("أ", "ب", "ج", "د")
INTENTS = ("ask", "verify", "translate_term")

SYSTEM = """You are the routing component of Baseera, an Islamic Q&A assistant. Classify the user message.
Return ONLY a JSON object: {"level": "أ|ب|ج|د", "intent": "ask|verify|translate_term", "language": "ar|en|other", "term": "<term or null>"}

Levels (content sensitivity):
- أ: stable basics: Quran, authentic hadith, pillars of Islam/faith, basic seerah, basic ethics, definitions.
- ب: explaining concepts, comparisons, objectives of Sharia (maqasid), general questions and common doubts/objections about Islam.
- ج: fiqh disagreements between schools, detailed aqeedah disputes, contested historical issues, questions that need specialist scholarly treatment, "do all Muslims agree on this?".
- د: a personal situation or ruling request about a specific person's case: validity of a particular contract/worship/marriage, family disputes, legal or medical matters with religious impact, "am I allowed to ... in my marriage / in my country".
Intents:
- verify: the user pastes a message/verse/hadith and wants it checked or asks for proof of a claim.
- translate_term: ONLY an explicit request to translate a term or give its English equivalent (set "term"). "What does X mean in Islam?" is a plain ask, not translate_term.
- ask: everything else.
A hostile or accusatory question about Islam is still level ب (answer wisely), not د.
Language is the language of the message itself."""

_PERSONAL = re.compile(
    r"(هل يجوز لي|هل يحق لي|هل يصح لي|ما حكم (?:زواجي|طلاقي|عقدي)|في زواجي|زوجتي|زوجي|طلقت|طلاقي|عقد(?:ي| العمل| الايجار)|"
    r"انا في دوله|أنا في دولة|حالتي|مشكلتي|ابي|أبي يرفض|اختلفت مع)")
_PERSONAL_EN = re.compile(
    r"\b(am i allowed|can i|should i|my (?:wife|husband|marriage|divorce|contract|landlord|boss|father|mother|case)|"
    r"i live in|i am in (?:the )?[a-z]+ and|is my (?:marriage|contract|prayer|fast) valid)\b", re.I)
_DISPUTE = re.compile(r"(اختلاف(?: العلماء| الفقهاء)?|ترجيح|المذاهب|اصح الاقوال|هل كل المسلمين|do all muslims agree|differ(?:ence|ent)? (?:between|among) (?:the )?(?:scholars|schools))", re.I)
_CONTESTED = re.compile(r"(الراجح|ايهما علي حق|الشيعه|معاويه|صفين|وقعه الجمل|المولد النبوي|الموسيقي|المعازف|الغناء|تصوير ذوات|تكفير|الخلاف بين)")
_BASIC = re.compile(r"(ما معني (?:ايه|اية|سوره|حديث|قوله تعالي|لا اله)|ما فضل|ما هي (?:سوره|اركان|الصلوات)|كم عدد|من هو النبي|اركان|ما هي سوره)")
_SENSITIVE = re.compile(r"(جهاد|قتال|الحدود|الرده|الرق|الاسترقاق|jihad|apostasy|slavery)")
_CONCEPT = re.compile(r"(مقاصد|الحكمه|لماذا|هل الاسلام|هل يتعارض|هل يظلم|ارهاب|عنف|do muslims|why do|is islam)")
_VERIFY = re.compile(r"(تحقق|هل هذا الحديث صحيح|هل هذه الايه|صحيح ام|fact.?check|is this (?:hadith|verse)|verify|authentic\?|ارسل لي|وصلني)", re.I)
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
    if _PERSONAL.search(text) or _PERSONAL_EN.search(text):
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
          or (re.search(r"(قال رسول الله|قال النبي|عن النبي|قال تعالى)", text) and re.search("[«\"“﴿]", text) and not re.search("[؟?]", text))):
        intent, level = "verify", "ب"
    return {"level": level, "intent": intent, "language": lang, "term": term, "source": "heuristic"}


_QUOTED = re.compile("[«»\"“”﴿﴾]|قال رسول الله|قال النبي|قال تعالى|قال الله|عن النبي")


def has_claim_text(text: str) -> bool:
    """A verify request needs something to verify: a quotation, an attributed saying, or a long pasted message."""
    return bool(_QUOTED.search(text)) or len(text) >= 80


def route(text: str) -> dict:
    h = heuristic_route(text)
    if not L.llm_available():
        return h
    try:
        raw = L.get_llm().complete(L.ROUTER_MODEL, SYSTEM, text, max_tokens=120)
        j = L.extract_json(raw)
        level = j.get("level") if j.get("level") in LEVELS else h["level"]
        intent = j.get("intent") if j.get("intent") in INTENTS else h["intent"]
        out = {"level": level, "intent": intent, "language": j.get("language") or h["language"],
               "term": j.get("term") if j.get("term") not in (None, "null", "") else h["term"], "source": "llm"}
    except Exception as e:  # never fail the request because routing failed
        h["router_error"] = str(e)[:200]
        return h
    # Safety: code can only escalate the model's level (to د for personal cases, to ج for known contested topics).
    if h["level"] == "د":
        out["level"] = "د"
    elif _CONTESTED.search(normalize_ar(text)) and out["level"] in ("أ", "ب"):
        out["level"] = "ج"
    elif out["level"] == "أ" and (_CONCEPT.search(normalize_ar(text)) or _CONCEPT.search(text.lower())
                                  or _SENSITIVE.search(normalize_ar(text)) or _SENSITIVE.search(text.lower())):
        out["level"] = "ب"  # explanation/doubt/war-and-penal topics are never "stable basics"
    # "verify" needs text to verify; "give me a hadith that proves this" has none -> it is an ask (and will abstain).
    if out["intent"] == "verify" and not has_claim_text(text):
        out["intent"] = "ask"
    if h["intent"] == "translate_term" and out["intent"] == "ask" and h["term"]:
        out["intent"], out["term"] = "translate_term", h["term"]
    return out
