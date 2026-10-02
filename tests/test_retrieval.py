"""Hybrid retrieval on 10+ real questions (uses the real Chroma index + bge-m3; run ingest/embed.py first)."""
import pytest

from core import retrieve as R
from core.normalize import normalize_ar

pytestmark = pytest.mark.vectors

REQUIRED_KEYS = {"source", "id", "text", "reference_url", "grade", "type"}


def ids(q):
    return [r["id"] for r in R.retrieve(q)]


def titles(q):
    return " ".join(normalize_ar((r["title"] or "") + " " + (r["text"] or "")[:150]) for r in R.retrieve(q))


CASES = [  # (question, any-of expected ids)
    ("هل الإسلام انتشر بالسيف؟", {"qa:bayyinat:229", "qa:icadb:36112"}),
    ("ما معنى التوحيد؟", {"term:glossary:2", "qa:icadb:26014", "qa:icadb:26020"}),
    ("ما حكم الصلاة؟", {"qa:icadb:26070", "qa:icadb:26071"}),
    ("ما معنى آية الكرسي؟", {"quran:2:255", "tafsir:muyassar:2:255"}),
    ("ما هي سورة الإخلاص؟", {"quran:112:1", "quran:112:2"}),
    ("ما معنى حديث إنما الأعمال بالنيات؟", {"hadith:hadeethenc:4560", "hadith:hadeethenc:66511"}),
    ("هل القرآن من تأليف محمد ﷺ؟", {"qa:bayyinat:26", "qa:bayyinat:27", "qa:icadb:36097"}),
    ("What does Islam say about patience?", {"quran:2:153", "hadith:hadeethenc:3295"}),
]


@pytest.mark.parametrize("q,expected", CASES)
def test_real_question_finds_the_right_source(q, expected):
    assert set(ids(q)) & expected, (q, ids(q))


@pytest.mark.parametrize("q", ["ما هي أركان الإسلام؟", "What are the five pillars of Islam?"])
def test_pillars_question_finds_the_pillars_hadith_arabic_and_english(q):
    assert normalize_ar("بني الإسلام على خمس") in titles(q)


def test_kaaba_question_returns_qa_material():
    res = R.retrieve("لماذا يعبد المسلمون الكعبة؟")
    assert any(r["type"] == "qa" for r in res)


def test_result_shape_and_grades():
    res = R.retrieve("ما معنى حديث إنما الأعمال بالنيات؟")
    for r in res:
        assert REQUIRED_KEYS <= set(r)
        assert r["reference_url"]
    h = next(r for r in res if r["type"] == "hadith")
    assert h["grade"]  # hadith grade always comes from the data


def test_vectors_add_cross_language_recall_that_keywords_alone_miss():
    q = "honoring one's parents"  # English query, Arabic hadith (no shared words)
    kw = {r["id"] for r in R.retrieve(q, vectors=False)}
    hy = {r["id"] for r in R.retrieve(q)}
    target = "hadith:hadeethenc:3260"  # "فارجع إلى والديك فأحسن صحبتهما" (Arabic-only title)
    assert target in hy and target not in kw


def test_dorar_live_results_are_optional_and_graded(monkeypatch):
    from core import dorar
    monkeypatch.setattr(dorar, "search", lambda q, page=1: [{"id": "hadith:dorar:abc", "book": "كتاب", "text": "نص", "reference_url": "https://dorar.net",
                                                              "grade": "صحيح", "grader": "الألباني", "narrator": "راو", "page_or_number": "1"}])
    res = R.retrieve("إنما الأعمال بالنيات", dorar=True)
    d = [r for r in res if r["source"] == "dorar"]
    assert d and d[0]["grade"] == "صحيح"
