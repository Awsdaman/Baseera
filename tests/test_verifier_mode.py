import json
from pathlib import Path

import pytest

from core import dorar
from core.normalize import normalize_ar
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


def test_introducer_phrase_is_not_a_claim():
    cl = V.heuristic_extract("انشر هذا الحديث: قال رسول الله ﷺ: «حب الوطن من الإيمان»")
    assert [c["text"] for c in cl] == ["حب الوطن من الإيمان"]


def test_one_word_edit_of_a_verse_that_contains_a_short_verse_phrase_is_misquoted_not_fabricated():
    """Found with synthetic data: short verses ('ياويلنا إنا كنا ظالمين') fit inside the claim, scored 100 in partial_ratio and
    crowded the real verse (21:46) out of the shortlist, so a 1-word edit was reported as 'not in the Quran'."""
    edited = "ولئن مستهم نفحة الملأ عذاب ربك ليقولن ياويلنا إنا كنا ظالمين"          # من -> الملأ
    r = V.check_verse(edited)
    assert r["verdict"] == "misquoted" and r["match"]["ref"] == "21:46" and r["score"] > 0.85
    assert any(d["op"] == "replace" and d["claimed"] == "الملا" and d["correct"] == "من" for d in r["diff"])
    deleted = "ولئن مستهم نفحة عذاب ربك ليقولن ياويلنا إنا كنا ظالمين"
    assert V.check_verse(deleted)["match"]["ref"] == "21:46"


def test_claim_that_starts_one_verse_before_the_shortlisted_verse_is_still_found():
    r = V.check_verse("الله لا إله إلا هو الحي القيوم لا تأخذه سنة ولا نوم له ما في السماء وما في الأرض من ذا الذي يشفع عنده إلا بإذنه")
    assert r["verdict"] == "misquoted" and r["match"]["ref"] == "2:255"


# ---------------------------------------------------------------- hadith wording: correct version + grade (user-reported case)
SHORT_MISQUOTE = "اتقوا النار ولو تمرة"          # missing "بشق"
SHORT_EXACT = "اتقوا النار ولو بشق تمرة"


def test_misquoted_hadith_shows_the_correct_wording_with_its_grade_and_a_diff(dorar_empty):
    r = V.check_hadith(SHORT_MISQUOTE)
    assert r["verdict"] == "misquoted" and r["color"] == "amber" and r["grade_verdict"] == "sound"
    c = r["correct"]
    assert normalize_ar(c["text_ar"]) == "فاتقوا النار ولو بشق تمره"                 # the authentic wording, from HadeethEnc
    assert c["source"] == "hadeethenc" and c["grade"] == "صحيح" and "متفق" in (c["book"] or "")
    missing = [d for d in r["diff"] if d["op"] == "missing"]
    assert len(missing) == 1 and normalize_ar(missing[0]["correct"]) == "بشق"
    assert "الصواب" in r["note"] and "حديث ثابت" in r["note"]                        # tells BOTH: wording is wrong, hadith is sound


def test_exact_wording_is_not_flagged_even_without_the_connecting_particle(dorar_empty):
    r = V.check_hadith(SHORT_EXACT)
    assert r["verdict"] == "sound" and r["color"] == "green" and "correct" not in r


def test_dorar_remarks_on_one_narration_never_become_the_hadiths_overall_grade(dorar_sample):
    """Live bug: a Dorar critique of one narrator ('فيه غفلة') made a sound hadith look weak. Only the curated grade may decide."""
    r = V.check_hadith(SHORT_MISQUOTE)
    assert r["verdict"] == "misquoted" and r["grade_verdict"] != "weak"


def test_boilerplate_phrases_are_unverifiable_not_matched_to_a_random_hadith(dorar_empty):
    for t in ("قال رسول الله صلى الله عليه وسلم", "عن أبي هريرة رضي الله عنه قال", "صلى الله عليه وسلم"):
        assert V.check_hadith(t)["verdict"] == "unverifiable", t


def test_unrelated_short_text_is_not_found(dorar_empty):
    assert V.check_hadith("مررت بالسوق أمس واشتريت خضروات طازجة")["verdict"] == "not_found"


def test_extractor_recognises_i_heard_the_prophet_say_form():
    msg = "عن عدي بن حاتم رضي الله عنه قال: سمعت النبي صلى الله عليه وسلم يقول: «اتقوا النار ولو  تمرة»"
    cl = V.heuristic_extract(msg)
    assert [(c["type"], c["text"]) for c in cl] == [("hadith", SHORT_MISQUOTE)]


def test_verify_text_end_to_end_on_the_user_message(dorar_empty):
    out = V.verify_text("عن عدي بن حاتم رضي الله عنه قال: سمعت النبي صلى الله عليه وسلم يقول: «اتقوا النار ولو  تمرة»")
    c = out["claims"][0]
    assert c["claim_type"] == "hadith" and c["verdict"] == "misquoted" and c["correct"]["grade"] == "صحيح"
