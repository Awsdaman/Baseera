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
