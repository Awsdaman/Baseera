from core import retrieve as R
from core import support as S


class StubScorer:
    """Scores 1.0 when the claim and the chunk share a marker word, else 0.1."""
    name = "stub"

    def __init__(self, marker):
        self.marker = marker
        self.calls = 0

    def scores(self, pairs):
        self.calls += len(pairs)
        return [1.0 if self.marker in claim and self.marker in chunk else 0.1 for claim, chunk in pairs]


def test_stretches_keep_only_cited_stretches_of_six_or_more_words():
    text = ("هذه جملة طويلة عن الصلاة وفضلها في حياة المسلم [[term:glossary:2]]" + chr(10) * 2 +
            "{{quran:112:1}}" + chr(10) * 2 +
            "جملة قصيرة [[term:glossary:2]]" + chr(10) * 2 +
            "جملة طويلة بلا أي إحالة على مصدر ولا غيره من المراجع هنا" + chr(10) * 2 +
            "{{note:partial}}")
    sts = S.stretches(text)
    assert len(sts) == 1 and sts[0]["cites"] == ["term:glossary:2"] and "[[" not in sts[0]["text"]


def test_stretches_split_around_placeholders_and_strip_markdown():
    text = "**التوحيد** هو إفراد الله وحده بالعبادة والربوبية [[a:b:1]] {{quran:112:1}} ثم تفسير آخر طويل بعد الآية الكريمة [[c:d:2, c:d:3]]"
    sts = S.stretches(text)
    assert [s["cites"] for s in sts] == [["a:b:1"], ["c:d:2", "c:d:3"]] and "*" not in sts[0]["text"]


def test_passage_chunks_include_translation_title_and_split_long_text():
    p = R.get_passage("quran:2:255")
    chunks = S.passage_chunks(p)
    assert p["text_en"][:30] in " ".join(chunks) and p["title"] in chunks
    long = {"text": " ".join(["كلمة"] * 400) + ".", "text_en": None, "title": None}
    assert len(S.passage_chunks(long)) >= 1 and all(len(c) <= S.CHUNK_CHARS * 2 for c in S.passage_chunks(long))


def test_support_report_takes_the_best_cited_passage():
    ps = {"x:1": {"id": "x:1", "text": "نص عن الزكاة", "text_en": None, "title": None},
          "x:2": {"id": "x:2", "text": "نص عن الصلاة", "text_en": None, "title": None}}
    text = "الصلاة عماد الدين وهي أهم أركان الإسلام بعد الشهادتين [[x:1, x:2]]"
    rep = S.support_report(text, ps, StubScorer("الصلاة"))
    assert rep[0]["best_id"] == "x:2" and rep[0]["score"] == 1.0


def test_unsupported_flags_stretches_below_theta_with_a_fix_recipe():
    ps = [{"id": "x:1", "text": "نص عن الزكاة", "text_en": None, "title": None}]
    text = "الصلاة عماد الدين وهي أهم أركان الإسلام بعد الشهادتين [[x:1]]"
    errs, rep = S.unsupported(text, ps, th=0.5, scorer=StubScorer("الصلاة"))
    assert len(errs) == 1 and "x:1" in errs[0] and "Fix:" in errs[0] and rep[0]["score"] < 0.5
    ok_errs, _ = S.unsupported(text.replace("الصلاة", "الزكاة", 1), ps, th=0.5, scorer=StubScorer("الزكاة"))
    assert ok_errs == []


def test_uncited_or_unknown_ids_never_crash_and_score_zero():
    rep = S.support_report("جملة طويلة تحتوي على أكثر من ست كلمات هنا [[not:retrieved:1]]", {}, StubScorer("x"))
    assert rep[0]["score"] == 0.0 and rep[0]["best_id"] is None


def test_mode_and_theta_come_from_the_environment(monkeypatch):
    monkeypatch.delenv("SUPPORT_MODE", raising=False)
    assert S.mode() == "log"          # default: record scores, change nothing
    monkeypatch.setenv("SUPPORT_MODE", "off")
    assert S.mode() == "off"
    monkeypatch.setenv("SUPPORT_MODE", "enforce")
    assert S.mode() == "enforce"
    monkeypatch.setenv("SUPPORT_MODE", "bogus")
    assert S.mode() == "log"
    monkeypatch.setenv("SUPPORT_THETA", "0.77")
    assert S.theta() == 0.77
