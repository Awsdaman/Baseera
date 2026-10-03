import pytest

from core import router

CASES = [  # the 12 test cases of docs/data.pdf p.6: (text, accepted levels, intent)
    ("لماذا يعبد المسلمون الكعبة؟", "ب", "ask"),
    ("هل القرآن من تأليف محمد ﷺ؟", "ب", "ask"),
    ("هل الإسلام انتشر بالسيف؟", "ب", "ask"),
    ("لماذا توجد أحكام مختلفة بين العلماء؟", "ب", "ask"),
    ("أنا في دولة كذا، هل يجوز لي فعل كذا في زواجي؟", "د", "ask"),
    ("أعطني حديثًا يثبت هذا الكلام", "أب", "ask"),
    ("ما معنى التوحيد لشخص لم يسمع بالمصطلح من قبل؟", "أب", "ask"),
    ("ترجم كلمة التوحيد إلى الإنجليزية", "أ", "translate_term"),
    ("لماذا يمنع الإسلام الحرية؟ أليس هذا دينا متخلفا؟", "ب", "ask"),
    ("هل كل المسلمين يتفقون في هذه المسألة؟", "ج", "ask"),
    ("ما معنى قوله تعالى ﴿إن الله مع الصابرين دائما﴾؟", "أب", "ask"),
    ("What does the word jihad mean in Islam?", "أب", "ask"),
]


@pytest.mark.parametrize("text,levels,intent", CASES)
def test_heuristic_route_on_challenge_cases(text, levels, intent):
    r = router.heuristic_route(text)
    assert r["level"] in levels, r
    assert r["intent"] == intent, r


def test_translate_term_extracts_glossary_term():
    assert router.heuristic_route("ترجم كلمة التوحيد إلى الإنجليزية")["term"] == "التوحيد"
    assert router.heuristic_route("translate the word Tawhid")["term"] == "التوحيد"


def test_llm_route_parsed(fake_llm):
    fake_llm(router=['```json\n{"level": "ج", "intent": "ask", "language": "ar", "term": null}\n```'])
    r = router.route("هل كل المسلمين يتفقون؟")
    assert (r["level"], r["intent"], r["source"]) == ("ج", "ask", "llm")


def test_code_escalates_personal_case_even_if_model_says_lower(fake_llm):
    fake_llm(router=['{"level": "أ", "intent": "ask", "language": "ar"}'])
    assert router.route("أنا في دولة كذا، هل يجوز لي فعل كذا في زواجي؟")["level"] == "د"


def test_bad_llm_output_falls_back_to_heuristic(fake_llm):
    fake_llm(router=["not json at all"])
    r = router.route("هل كل المسلمين يتفقون في هذه المسألة؟")
    assert r["level"] == "ج" and r["source"] == "heuristic"


def test_language_detection():
    assert router.detect_language("What is Islam?") == "en"
    assert router.detect_language("ما هو الإسلام؟") == "ar"


def test_verify_without_anything_to_verify_becomes_ask(fake_llm):
    fake_llm(router=['{"level": "أ", "intent": "verify", "language": "ar", "term": null}'])
    r = router.route("أعطني حديثًا يثبت هذا الكلام")
    assert r["intent"] == "ask"


def test_verify_with_a_quoted_hadith_stays_verify(fake_llm):
    fake_llm(router=['{"level": "أ", "intent": "verify", "language": "ar", "term": null}'])
    assert router.route("قال رسول الله ﷺ: «اطلبوا العلم ولو في الصين» تحقق")["intent"] == "verify"


def test_contested_topic_is_escalated_to_level_j(fake_llm):
    fake_llm(router=['{"level": "ب", "intent": "ask", "language": "ar", "term": null}'])
    assert router.route("ما حكم الموسيقى والغناء في الإسلام؟")["level"] == "ج"


def test_escalation_never_lowers_a_level(fake_llm):
    fake_llm(router=['{"level": "ج", "intent": "ask", "language": "ar", "term": null}'])
    assert router.route("ما هي أركان الإسلام؟")["level"] == "ج"


@pytest.mark.parametrize("text", ["لماذا يعبد المسلمون الكعبة؟", "ما معنى الجهاد في الإسلام؟", "What does the word jihad mean in Islam?"])
def test_basic_level_is_escalated_to_b_for_doubts_and_war_topics(fake_llm, text):
    fake_llm(router=['{"level": "أ", "intent": "ask", "language": "ar", "term": null}'])
    assert router.route(text)["level"] == "ب"


def test_plain_basics_stay_level_a(fake_llm):
    fake_llm(router=['{"level": "أ", "intent": "ask", "language": "ar", "term": null}'])
    assert router.route("ما هي أركان الإسلام؟")["level"] == "أ"


def test_router_prompt_limits_translate_term_to_explicit_requests():
    assert "ONLY an explicit request" in router.SYSTEM


@pytest.mark.parametrize("msg", ["انشر هذا الحديث: قال رسول الله ﷺ: «حب الوطن من الإيمان»",
                                 "يقول النبي ﷺ: «اختلاف أمتي رحمة» فتحققوا منه",
                                 "قال تعالى: ﴿إن الله مع الصابرين دائما وأبدا﴾ انشروها"])
def test_pasted_attributed_saying_without_a_question_is_always_verify_even_if_the_model_says_ask(fake_llm, msg):
    fake_llm(router=['{"level": "ب", "intent": "ask", "language": "ar", "term": null}'])
    assert router.route(msg)["intent"] == "verify"


def test_ruling_validity_question_with_quotes_stays_an_ask(fake_llm):
    fake_llm(router=['{"level": "ب", "intent": "ask", "language": "ar", "term": null}'])
    assert router.route('هل هذا الحكم "حكم اذكار الصباح واجبه" صحيح؟')["intent"] == "ask"
