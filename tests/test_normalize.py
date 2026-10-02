from core.normalize import (light_stem, normalize_ar, search_form, strip_aya_end,
                            strip_tashkeel, strip_tatweel, tokens)


def test_strip_tashkeel():
    assert strip_tashkeel("بِسۡمِ ٱللَّهِ") == "بسم ٱلله"
    assert strip_tashkeel("مُحَمَّدٌ") == "محمد"


def test_tatweel():
    assert strip_tatweel("الــلــه") == "الله"
    assert normalize_ar("اللـــه") == "الله"


def test_alef_unification():
    for c in "أإآٱ":
        assert normalize_ar(c + "مر") == "امر"


def test_ya_and_ta_marbuta():
    assert normalize_ar("على") == "علي"
    assert normalize_ar("مدرسة") == "مدرسه"
    assert normalize_ar("مؤمن") == "مومن"


def test_quran_text_matches_emlaey():
    assert normalize_ar("بِسۡمِ ٱللَّهِ ٱلرَّحۡمَٰنِ ٱلرَّحِيمِ") == normalize_ar("بسم الله الرحمن الرحيم")
    # superscript alef must vanish: الرحمٰن -> الرحمن
    assert normalize_ar("ٱلرَّحۡمَٰنِ") == "الرحمن"


def test_aya_end_marker():
    assert strip_aya_end("ٱلرَّحِيمِ ۝١") == "ٱلرَّحِيمِ"
    assert strip_aya_end("ٱلنَّاسِ ۝٦") == "ٱلنَّاسِ"
    assert strip_aya_end("no marker") == "no marker"


def test_digits_and_punctuation():
    assert normalize_ar("﴿الله﴾ ١٢٣،") == "الله 123"
    assert normalize_ar("(Hello,  World!)") == "hello world"


def test_whitespace_and_zero_width():
    assert normalize_ar("  الله‏   اكبر\n") == "الله اكبر"


def test_empty():
    assert normalize_ar("") == ""
    assert normalize_ar(None) == ""


def test_presentation_forms_fold():
    assert normalize_ar("ﻟﺎ") == "لا"  # lam-alef presentation form pieces


def test_idempotent():
    s = "إِنَّمَا الْأَعْمَالُ بِالنِّيَّاتِ"
    assert normalize_ar(normalize_ar(s)) == normalize_ar(s) == "انما الاعمال بالنيات"


def test_light_stem():
    assert light_stem("الاعمال") == "اعمال"
    assert light_stem("بالنيات") == "نيات"
    assert light_stem("الي") == "الي"  # too short to strip
    assert search_form("الصلاة والزكاة") == "صلاه زكاه"
    assert tokens("الله اكبر") == ["الله", "اكبر"]


def test_allah_not_stemmed():
    assert light_stem("الله") == "الله"
    assert search_form("بسم الله") == "بسم الله"
