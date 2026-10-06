"""Adversarial tests for core/verify.py: the model must never get unretrieved ids or self-written sacred text through."""
from core import retrieve as R
from core.normalize import normalize_ar
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


# ---------------------------------------------------------------- code-owned notes, multi-id citations, citation recipes
def test_note_placeholder_is_rendered_by_code_in_both_languages():
    from core.verify import NOTES
    ps = P("term:glossary:2")
    text = "التوحيد إفراد الله بالعبادة وهو أصل الدين الذي جاء به الأنبياء جميعا [[term:glossary:2]]\n\n{{note:partial}}"
    r = verify_answer(text, ps)
    assert r.ok, r.errors
    note = [b for b in r.blocks if b["kind"] == "notice"][0]
    assert note["note"] == "partial" and note["text"] == NOTES["partial"]["ar"]
    assert verify_answer(text, ps, "en").blocks[-1]["text"] == NOTES["partial"]["en"]
    assert NOTES["partial"]["ar"] in r.answer_text


def test_unknown_note_key_is_rejected():
    r = verify_answer("{{note:foo}} [[term:glossary:2]]", P("term:glossary:2"))
    assert not r.ok and any("Unknown note" in e for e in r.errors)


def test_a_note_alone_is_not_a_source():
    r = verify_answer("{{note:partial}}", P("term:glossary:2"))
    assert not r.ok and any("no citation" in e for e in r.errors)


def test_free_text_disclaimer_is_still_rejected():
    text = "معنى التوحيد إفراد الله بالعبادة [[term:glossary:2]]\n\nهذه الإجابة لا تغطي كل ما يتعلق بالمسألة وهي ليست حصرا شاملا لها"
    r = verify_answer(text, P("term:glossary:2"))
    assert not r.ok and any("without an explicit" in e for e in r.errors)


def test_a_note_does_not_cite_the_text_around_it():
    text = "نص طويل يتكون من أكثر من ثمان كلمات بلا إحالة ولا مصدر {{note:partial}} [[term:glossary:2]]"
    assert not verify_answer(text, P("term:glossary:2")).ok


def test_multi_id_citation_checks_every_id():
    ps = P("term:glossary:2", "term:glossary:3")
    text = "شرح طويل عن معنى التوحيد والعبادة عند أهل العلم وبيان أركانهما [[term:glossary:2, term:glossary:3]]"
    r = verify_answer(text, ps)
    assert r.ok, r.errors
    assert {s["id"] for s in r.sources} == {"term:glossary:2", "term:glossary:3"} and "[1][2]" in r.blocks[0]["text"]
    bad = verify_answer(text.replace("term:glossary:3", "term:glossary:9999"), ps)
    assert not bad.ok and any("NOT retrieved" in e for e in bad.errors)


def test_malformed_citation_is_reported_precisely():
    r = verify_answer("شرح طويل عن معنى التوحيد والعبادة عند أهل العلم وبيان أركانهما [[hello world]]", P("term:glossary:2"))
    assert not r.ok and any("Malformed citation" in e for e in r.errors)


def test_text_after_a_verse_placeholder_needs_its_own_citation_of_that_verse():
    ps = P("quran:3:45")
    bad = "{{quran:3:45}}\nThe Quran explicitly includes Jesus among the prophets and messengers of God according to the sources."
    assert not verify_answer(bad, ps, "en").ok
    good = bad + " [[quran:3:45]]"
    assert verify_answer(good, ps, "en").ok


def test_citation_error_carries_a_fix_recipe_and_the_rejected_text():
    r = verify_answer("هذه فقرة طويلة بلا أي مصدر ولا إحالة على الإطلاق في هذا الكلام كله [[term:glossary:2]]\n\nوهذه فقرة ثانية طويلة جدا بلا إحالة على أي مصدر من المصادر", P("term:glossary:2"))
    msg = next(e for e in r.errors if "without an explicit" in e)
    assert "Fix:" in msg and "{{note:partial}}" in msg and "وهذه فقرة ثانية" in msg


def test_quoting_the_shahada_as_a_cited_stock_phrase_is_allowed():
    """Live regression (dp6-07): «لا إله إلا الله» occurs inside several verses but is a formula, not typed scripture."""
    ps = P("qa:icadb:26014")
    assert "لا اله الا الله" in normalize_ar(ps[0]["text"])
    text = "وكلمة التوحيد هي «لا إله إلا الله» وهي أصل الإسلام عند أهل العلم جميعا [[qa:icadb:26014]]"
    r = verify_answer(text, ps)
    assert r.ok, r.errors


def test_quoting_a_whole_short_verse_is_still_rejected():
    ps = P("quran:112:1", "qa:icadb:26014")
    for q in ("«قل هو الله أحد»", "«بسم الله الرحمن الرحيم»"):
        r = verify_answer(f"وقد جاء في القرآن {q} وهو واضح في معناه عند أهل العلم [[qa:icadb:26014]]", ps)
        assert not r.ok, q


# ---------------------------------------------------------------- citation syntax repair (small local models)
from core.verify import normalize_citations  # noqa: E402


def _by_id(*ids):
    return {p["id"]: p for p in P(*ids)}


def test_split_citation_list_is_merged_into_one_bracket():
    by = _by_id(KURSI, "tafsir:muyassar:2:255")
    out = normalize_citations("بيان [[quran:2:255], [tafsir:muyassar:2:255]].", by)
    assert "[[quran:2:255, tafsir:muyassar:2:255]]" in out
    assert verify_answer(out, list(by.values())).ok


def test_short_tafsir_id_is_expanded_only_when_retrieved():
    by = _by_id(KURSI, "tafsir:muyassar:2:255")
    assert normalize_citations("بيان [[tafsir:2:255]]", by) == "بيان [[tafsir:muyassar:2:255]]"
    assert normalize_citations("بيان [[tafsir:3:7]]", by) == "بيان [[tafsir:3:7]]"
    r = verify_answer("هذه أعظم آية في القرآن الكريم وفيها بيان [[tafsir:3:7]]", list(by.values()))
    assert not r.ok and any("NOT retrieved" in e for e in r.errors)


def test_verse_range_citation_expands_only_when_all_ayat_retrieved():
    by = _by_id("quran:112:1", "quran:112:2")
    assert normalize_citations("[[quran:112:1-2]]", by) == "[[quran:112:1, quran:112:2]]"
    assert normalize_citations("[[quran:112:1-4]]", by) == "[[quran:112:1-4]]"


def test_placeholder_followed_by_its_own_citation_is_reordered():
    by = _by_id(KURSI)
    out = normalize_citations("انظر الآية: {{quran:2:255}} [[quran:2:255]].", by)
    assert out.index("[[quran:2:255]]") < out.index("{{quran:2:255}}")
    assert normalize_citations("{{quran:2:255}} [[tafsir:muyassar:2:255]]", by) == "{{quran:2:255}} [[tafsir:muyassar:2:255]]"  # different ids: untouched


def test_unrepaired_split_citation_gets_a_precise_error_not_an_english_one():
    ps = P(KURSI, "tafsir:muyassar:2:255")
    r = verify_answer("بيان طويل [[quran:2:255], [tafsir:muyassar:2:255]].", ps)
    assert any(e.startswith("Malformed citation") for e in r.errors)
    assert not any("English word" in e for e in r.errors)


def test_arabic_comma_separates_ids_in_one_bracket():
    ps = P(KURSI, "tafsir:muyassar:2:255")
    assert verify_answer("هذه أعظم آية في القرآن [[quran:2:255،tafsir:muyassar:2:255]]", ps).ok


def test_duplicate_errors_are_reported_once():
    ps = P(KURSI)
    r = verify_answer("بيان [[tafsir:2:255]] وبيان آخر [[tafsir:2:255]]", ps)
    assert len(r.errors) == len(set(r.errors))


def test_invented_placeholder_kind_is_one_clear_error_not_english_or_citation_noise():
    ps = P("tafsir:muyassar:2:255")
    r = verify_answer("بيان طويل عن الآية وفضلها [[tafsir:muyassar:2:255]]\n\n{{qa:icadb:26024}}", ps)
    assert any(e.startswith("Malformed placeholder") for e in r.errors)
    assert not any(e.startswith("Malformed citation") or "English word" in e for e in r.errors)


def test_evidence_order_rule_does_not_invite_qa_placeholders():
    from core import generate as G
    rule = G.RULES[G.RULES.index("12. ORDER OF EVIDENCE"):]
    assert "NO placeholder" in rule and "{{qa" not in rule


# ---------------------------------------------------------------- the shahada is a stock formula, not typed scripture
def test_shahada_in_a_pillars_answer_is_not_flagged_as_copied_hadith():
    ps = P("hadith:hadeethenc:65000") if R.get_passage("hadith:hadeethenc:65000") else P(KURSI)
    text = "الركن الأول من أركان الإسلام هو الشهادة بأن لا إله إلا الله وأن محمداً عبده ورسوله [[%s]]" % ps[0]["id"]
    r = verify_answer(text, ps)
    assert not any("resembling" in e or "Unattributed" in e for e in r.errors), r.errors


def test_quoted_shahada_alone_is_allowed_but_a_verse_next_to_it_is_still_caught():
    ps = P(KURSI)
    ok = verify_answer("وكلمة التوحيد هي «لا إله إلا الله» وهي أساس الدين وعليها يقوم الإسلام كله [[quran:2:255]]", ps)
    assert not any("Unattributed" in e for e in ok.errors), ok.errors
    verse = R.get_passage(KURSI)["text"]
    bad = verify_answer(f"قال: «لا إله إلا الله {normalize_ar(verse)}» [[quran:2:255]]", ps)
    assert not bad.ok                                   # the shahada exemption does not hide a typed verse


# ---------------------------------------------------------------- English answers may not type scripture either
def test_english_answer_quoting_the_saheeh_translation_is_rejected():
    ps = P(KURSI)
    en = R.get_passage(KURSI)["text_en"]
    r = verify_answer(f'Allah says "{en}" and this is the greatest verse of the Quran. [[quran:2:255]]', ps, "en")
    assert not r.ok and any("Quran text written" in e for e in r.errors), r.errors


def test_english_answer_that_copies_a_long_run_without_quotes_is_rejected():
    ps = P(KURSI)
    en = R.get_passage(KURSI)["text_en"]
    r = verify_answer(f"The verse teaches that {en[:260]} [[quran:2:255]]", ps, "en")
    assert not r.ok


def test_english_paraphrase_and_placeholders_are_allowed():
    ps = P(KURSI, "tafsir:muyassar:2:255")
    ok = verify_answer("This verse describes Allah's greatness and His knowledge of everything, and is among the most beloved verses to recite for protection. "
                       "[[tafsir:muyassar:2:255]]\n\n{{quran:2:255}}", ps, "en")
    assert ok.ok, ok.errors
    short = verify_answer('The Quran calls Him "the Ever-Living, the Sustainer of existence" in this verse, as the tafsir explains at length for readers. [[quran:2:255]]', ps, "en")
    assert not any("Quran text written" in e for e in short.errors)       # a 5-word attribute inside quotes is below the recitation threshold


def test_separator_slip_in_a_citation_id_is_repaired_only_when_unambiguous():
    by = {"qa:icadb-aalam:10387": {"id": "qa:icadb-aalam:10387"}, "qa:icadb:36072": {"id": "qa:icadb:36072"}}
    assert normalize_citations("نص [[qa:icadb:aalam:10387, qa:icadb:36072]]", by) == "نص [[qa:icadb-aalam:10387, qa:icadb:36072]]"
    assert normalize_citations("نص [[qa:icadb:99999]]", by) == "نص [[qa:icadb:99999]]"          # not a separator slip: still rejected later
    two = {"qa:x-1:5": {}, "qa:x:1-5": {}}
    assert normalize_citations("[[qa:x:1:5]]", two) == "[[qa:x:1:5]]"                           # ambiguous: left alone
