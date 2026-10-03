"""Adversarial tests for core/verify.py: the model must never get unretrieved ids or self-written sacred text through."""
from core import retrieve as R
from core.verify import verify_answer


def P(*ids):
    out = [R.get_passage(i) for i in ids]
    assert all(out), ids
    return out


KURSI = "quran:2:255"


def test_valid_answer_substitutes_exact_text():
    ps = P(KURSI, "tafsir:muyassar:2:255")
    r = verify_answer("هذه أعظم آية في القرآن [[tafsir:muyassar:2:255]]\n\n{{quran:2:255}}", ps)
    assert r.ok, r.errors
    verse = next(b for b in r.blocks if b["kind"] == "quran")
    assert verse["text_ar"] == ps[0]["text"] and verse["ref"] == "2:255"
    assert ps[0]["text"] in r.answer_text
    assert {s["id"] for s in r.sources} == {KURSI, "tafsir:muyassar:2:255"}


def test_placeholder_for_unretrieved_verse_fails():
    r = verify_answer("انظر {{quran:112:1}}", P(KURSI))
    assert not r.ok and any("NOT retrieved" in e for e in r.errors)


def test_invented_hadith_id_fails():
    assert not verify_answer("{{hadith:hadeethenc:99999999}}", P(KURSI)).ok


def test_invented_citation_fails():
    r = verify_answer("معلومة مهمة جدا لكل مسلم يريد المعرفة الصحيحة [[qa:bayyinat:9999]]", P(KURSI))
    assert not r.ok and any("NOT retrieved" in e for e in r.errors)


def test_model_writes_verse_in_ornate_brackets_fails():
    r = verify_answer("قال الله: ﴿قُلْ هُوَ اللَّهُ أَحَدٌ﴾ [[quran:112:1]]", P("quran:112:1"))
    assert not r.ok


def test_model_writes_verse_text_unbracketed_fails():
    r = verify_answer("الله لا إله إلا هو الحي القيوم لا تأخذه سنة ولا نوم [[quran:2:255]]", P(KURSI))
    assert not r.ok and any("Quran verse" in e for e in r.errors)


def test_model_writes_verse_in_guillemets_fails():
    r = verify_answer("وقد قال «قل هو الله أحد الله الصمد» [[quran:112:1]]", P("quran:112:1"))
    assert not r.ok


def test_model_misquotes_verse_still_fails():
    r = verify_answer("الله لا إله إلا هو الحي القيوم لا تأخذه سنة ولا نوم له ما في السماء [[quran:2:255]]", P(KURSI))
    assert not r.ok


def test_unattributed_arabic_quote_fails():
    r = verify_answer("قال بعض العلماء «العبادة اسم جامع لكل ما يحبه الله» وهذا مهم جدا للمسلم [[term:glossary:3]]", P("term:glossary:3"))
    assert not r.ok and any("Unattributed" in e or "quotation" in e for e in r.errors)


def test_attributed_quote_of_retrieved_passage_ok():
    ps = P("term:glossary:3")
    quote = ps[0]["text"][:40]
    r = verify_answer(f"جاء في المرجع: «{quote}» [[term:glossary:3]]", ps)
    assert r.ok, r.errors


def test_short_term_in_quotes_allowed():
    r = verify_answer("معنى «التوحيد» عند أهل العلم هو إفراد الله بالعبادة [[term:glossary:2]]", P("term:glossary:2"))
    assert r.ok, r.errors


def test_uncited_paragraph_fails():
    r = verify_answer("هذه فقرة طويلة بلا أي مصدر ولا إحالة على الإطلاق في هذا الكلام كله.\n\nفقرة ثانية [[term:glossary:2]]", P("term:glossary:2"))
    assert not r.ok and any("without an explicit" in e for e in r.errors)


def test_no_citation_at_all_fails():
    assert not verify_answer("كلام قصير", P(KURSI)).ok


def test_malformed_placeholder_fails():
    r = verify_answer("{{quran:abc}} [[quran:2:255]]", P(KURSI))
    assert not r.ok and any("Malformed" in e for e in r.errors)
    assert not verify_answer("{{poem:1:1}} [[quran:2:255]]", P(KURSI)).ok


def test_range_requires_every_verse_retrieved():
    assert not verify_answer("{{quran:112:1-4}}", P("quran:112:1", "quran:112:2")).ok
    ps = P("quran:112:1", "quran:112:2", "quran:112:3", "quran:112:4")
    r = verify_answer("{{quran:112:1-4}}", ps)
    assert r.ok and r.blocks[0]["ref"] == "112:1-4"
    assert r.blocks[0]["text_ar"] == " ".join(p["text"] for p in ps)


def test_hadith_placeholder_carries_grade_from_data():
    ps = P("hadith:hadeethenc:4560")
    r = verify_answer("{{hadith:hadeethenc:4560}}", ps)
    assert r.ok
    b = r.blocks[0]
    assert b["kind"] == "hadith" and b["grade"] == ps[0]["grade"] and b["grade_source"] == "HadeethEnc.com"
    assert b["text_ar"] == ps[0]["text"]


def test_insufficient_evidence_is_abstain():
    r = verify_answer("INSUFFICIENT_EVIDENCE", P(KURSI))
    assert r.ok and r.abstain


def test_empty_fails():
    assert not verify_answer("", P(KURSI)).ok


def test_tafsir_placeholder_needs_the_tafsir_passage():
    assert not verify_answer("{{tafsir:2:255}}", P(KURSI)).ok


def test_ordinary_explanation_sentences_are_not_mistaken_for_verses():
    """Regression: live Sonnet output was rejected because tiny verses fuzzy-matched inside normal sentences."""
    paras = ["معناهما الإقرار بأن الله وحده هو المعبود بحق، وأن محمدًا ﷺ عبده ورسوله [[term:glossary:1]]",
             "والزكاة هي إنفاق جزء يسير من المال على الفقراء والمساكين، وتعين المسلم على تغليب البذل والعطاء على الشح [[term:glossary:1]]",
             "والحج تفرغ للخالق في وقت ومكان معينين بمناسك واحدة لكل المؤمنين، وهذا شرح عام مبني على النصوص المذكورة أعلاه [[term:glossary:1]]"]
    text = "\n\n".join(paras)
    r = verify_answer(text, P("term:glossary:1"))
    assert r.ok, r.errors


def test_placeholder_does_not_count_as_citation_for_surrounding_explanation():
    ps = P("hadith:hadeethenc:4560", "term:glossary:2")
    bad = "هذا شرح طويل عن الحديث ومعناه عند أهل العلم قبل عرض النص نفسه.\n\n{{hadith:hadeethenc:4560}}"
    r = verify_answer(bad, ps)
    assert not r.ok and any("explicit" in e for e in r.errors)
    good = "هذا شرح طويل عن الحديث ومعناه عند أهل العلم قبل عرض النص نفسه [[term:glossary:2]]\n\n{{hadith:hadeethenc:4560}}"
    assert verify_answer(good, ps).ok


def test_each_segment_around_a_placeholder_needs_its_own_citation():
    ps = P("hadith:hadeethenc:4560", "term:glossary:2")
    before_only = "شرح طويل قبل الحديث يتضمن معلومات كثيرة ومهمة جدا [[term:glossary:2]] {{hadith:hadeethenc:4560}} وشرح طويل آخر بعد الحديث بلا أي إحالة على مصدر"
    assert not verify_answer(before_only, ps).ok
    both = before_only.replace("بلا أي إحالة على مصدر", "بإحالة صحيحة [[term:glossary:2]]")
    assert verify_answer(both, ps).ok


def test_short_stock_phrase_from_a_verse_is_allowed_but_a_long_run_is_not():
    """Live regression: 'حج البيت لمن استطاع إليه سبيلا' (a 6-word phrase of 3:97 that explanations reuse) must not block the
    pillars answer, while a long unquoted copy of a verse is still rejected."""
    ps = P("term:glossary:2", "quran:2:255")
    ok = "ومن أركانه الحج وهو حج البيت لمن استطاع إليه سبيلا عند أهل العلم [[term:glossary:2]]"
    assert verify_answer(ok, ps).ok, verify_answer(ok, ps).errors
    long_copy = "وقد قال بعضهم إن الله لا إله إلا هو الحي القيوم لا تأخذه سنة ولا نوم [[term:glossary:2]]"
    assert not verify_answer(long_copy, ps).ok
