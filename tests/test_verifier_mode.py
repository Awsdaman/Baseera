import json
from pathlib import Path

import pytest

from core import dorar
from core import verifier_mode as V

SAMPLE = Path(__file__).resolve().parent.parent / "data" / "samples" / "dorar_search.json"


@pytest.fixture
def dorar_sample(monkeypatch):
    html = json.loads(SAMPLE.read_text(encoding="utf-8"))["ahadith"]["result"]
    monkeypatch.setattr(dorar, "search", lambda q, page=1: dorar.parse(html))


@pytest.fixture
def dorar_empty(monkeypatch):
    monkeypatch.setattr(dorar, "search", lambda q, page=1: [])


# --- verses
def test_exact_verse_verified():
    r = V.check_verse("الله لا إله إلا هو الحي القيوم لا تأخذه سنة ولا نوم")
    assert r["verdict"] == "verified" and r["match"]["ref"] == "2:255" and r["color"] == "green"


def test_verse_with_tashkeel_and_uthmani_spelling_verified():
    assert V.check_verse("بِسۡمِ ٱللَّهِ ٱلرَّحۡمَٰنِ ٱلرَّحِيمِ")["verdict"] == "verified"


def test_misquoted_verse_returns_correct_text_and_word_diff():
    r = V.check_verse("الله لا إله إلا هو الحي القيوم لا تأخذه سنة ولا نوم له ما في السماء وما في الأرض")
    assert r["verdict"] == "misquoted" and r["color"] == "amber" and r["match"]["ref"] == "2:255"
    reps = [d for d in r["diff"] if d["op"] == "replace"]
    assert reps and reps[0]["claimed"] == "السماء" and reps[0]["correct"] == "السموات"
    from core import retrieve as R
    assert r["match"]["text_ar"] == R.get_passage("quran:2:255")["text"]  # exact Mushaf text from the DB


def test_extra_word_diff():
    r = V.check_verse("قل هو الله واحد احد الله الصمد")
    assert r["verdict"] == "misquoted" and any(d["op"] == "extra" for d in r["diff"])


def test_fabricated_verse_is_red():
    r = V.check_verse("إن الله يحب كل من يعمل بجد واجتهاد في عمله اليومي ويتقن صنعته")
    assert r["verdict"] == "fabricated" and r["color"] == "red"


def test_too_short_is_unverifiable():
    assert V.check_verse("الله")["verdict"] == "unverifiable"


# --- grades
@pytest.mark.parametrize("grade,cls", [
    ("صحيح", "sound"), ("حسن", "sound"), ("إسناده صحيح", "sound"), ("ضعيف", "weak"), ("ضعيف جدا", "weak"), ("منكر", "weak"),
    ("لا يصح", "weak"), ("ليس بصحيح", "weak"), ("موضوع", "fabricated"), ("لا أصل له", "fabricated"), ("باطل", "fabricated"),
    ("", "unknown"), (None, "unknown"), ("خطأ في إسناده", "unknown"),
])
def test_classify_grade(grade, cls):
    assert V.classify_grade(grade) == cls


# --- hadith
def test_sahih_hadith_green(dorar_sample):
    r = V.check_hadith("إنما الأعمال بالنيات وإنما لكل امرئ ما نوى")
    assert r["verdict"] == "sound" and r["color"] == "green"
    assert any(m["source"] == "hadeethenc" for m in r["matches"])
    assert all(m["grade"] for m in r["matches"])  # every grade is copied from data


def test_hadith_not_found_is_grey_no_guessing(dorar_empty):
    r = V.check_hadith("من قال سبحان الله مئة مرة في اليوم بنى الله له قصرا من الذهب في الجنة")
    assert r["verdict"] == "not_found" and r["color"] == "grey" and r["matches"] == []


def test_dorar_down_is_reported_not_treated_as_fabricated(monkeypatch):
    def boom(q, page=1):
        raise RuntimeError("blocked")
    monkeypatch.setattr(dorar, "search", boom)
    r = V.check_hadith("من قال سبحان الله مئة مرة في اليوم بنى الله له قصرا من الذهب في الجنة")
    assert r["color"] == "grey" and r["source_error"]


# --- extraction + full flow
MSG = "وصلني هذا: قال رسول الله ﷺ: «اطلبوا العلم ولو في الصين» وقال تعالى: ﴿إن الله مع الصابرين في كل حين وزمان﴾"


def test_heuristic_extraction():
    cl = V.heuristic_extract(MSG)
    kinds = {c["type"]: c["text"] for c in cl}
    assert kinds["hadith"] == "اطلبوا العلم ولو في الصين"
    assert kinds["verse"].startswith("إن الله مع الصابرين")


def test_llm_extraction_used_when_available(fake_llm):
    fake_llm(extract='[{"type": "verse", "text": "قل هو الله واحد احد", "claimed_ref": null, "attributed_to": null}]')
    claims, how = V.extract_claims("أي نص")
    assert how == "llm" and claims[0]["type"] == "verse"


def test_verify_text_end_to_end(dorar_empty):
    out = V.verify_text(MSG)
    assert out["extraction"].startswith("heuristic")
    by_type = {c["claim_type"]: c for c in out["claims"]}
    assert by_type["verse"]["verdict"] in ("misquoted", "fabricated")
    assert by_type["hadith"]["verdict"] == "not_found"
    assert sum(out["summary"].values()) == 2


def test_no_claims_message():
    out = V.verify_text("مرحبا كيف حالكم اليوم")
    assert out["claims"] == [] and out["message"]
