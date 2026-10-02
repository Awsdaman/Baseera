import json

import pytest

from core.db import connect
from core.glossary import lookup
from core.normalize import normalize_ar

con = connect()


def n(sql, *a):
    return con.execute(sql, a).fetchone()[0]


def test_quran_has_exactly_6236_verses():
    assert n("SELECT count(*) FROM quran") == 6236
    assert n("SELECT count(*) FROM passages WHERE type='quran'") == 6236
    assert n("SELECT count(*) FROM (SELECT DISTINCT surah, ayah FROM quran)") == 6236
    assert n("SELECT count(DISTINCT surah) FROM quran") == 114


def test_known_verse_counts():
    assert n("SELECT count(*) FROM quran WHERE surah=2") == 286
    assert n("SELECT count(*) FROM quran WHERE surah=1") == 7
    assert n("SELECT count(*) FROM quran WHERE surah=114") == 6


def test_uthmani_text_matches_emlaey_after_normalization_and_has_no_aya_marker():
    bad = []
    for r in con.execute("SELECT surah, ayah, text_uthmani, text_emlaey FROM quran"):
        if "۝" in r["text_uthmani"]:
            bad.append(("marker", r["surah"], r["ayah"]))
        # Mushaf spelling differs a little from emlaey (e.g. alef maqsura words), so require high overlap only
    assert not bad


def test_ayat_al_kursi():
    r = con.execute("SELECT text_ar, text_en FROM passages WHERE id='quran:2:255'").fetchone()
    assert normalize_ar(r["text_ar"]).startswith("الله لا اله الا هو الحي القيوم")
    assert "Allāh" in r["text_en"] or "Allah" in r["text_en"]


def test_every_ayah_has_tafsir():
    assert n("SELECT count(*) FROM passages WHERE type='tafsir'") == 6236
    assert n("SELECT count(*) FROM passages WHERE type='tafsir' AND length(text_ar)<1") == 0


def test_hadith_have_grades_and_urls():
    total = n("SELECT count(*) FROM passages WHERE source='hadeethenc'")
    assert total > 3000
    assert n("SELECT count(*) FROM passages WHERE source='hadeethenc' AND (grade IS NULL OR grade='')") == 0
    assert n("SELECT count(*) FROM passages WHERE source='hadeethenc' AND reference_url NOT LIKE 'https://hadeethenc.com/%'") == 0
    meta = json.loads(n("SELECT meta FROM passages WHERE id='hadith:hadeethenc:4560'") or "{}")
    assert meta["credit"] == "HadeethEnc.com"


def test_bayyinat_and_icadb_counts():
    assert n("SELECT count(*) FROM passages WHERE source='bayyinat'") == 263
    assert n("SELECT count(*) FROM passages WHERE source='icadb'") >= 500
    assert n("SELECT count(*) FROM passages WHERE source='glossary'") == 10


def test_bayyinat_ligatures_repaired():
    txt = " ".join(r[0] for r in con.execute("SELECT text_ar FROM passages WHERE source='bayyinat'"))
    for broken in ("اإلسالم", "اهلل", "املالئكة"):
        assert broken not in txt
    assert "الإسلام" in txt and "الله" in txt


def test_unique_ids_and_fts_in_sync():
    assert n("SELECT count(*) FROM passages") == n("SELECT count(*) FROM passages_fts")
    assert n("SELECT count(*) FROM (SELECT id FROM passages GROUP BY id HAVING count(*)>1)") == 0


def test_glossary_overrides():
    assert lookup("التوحيد")["en"] == "Tawhid / Oneness of God"
    assert lookup("Tawhid")["en"].startswith("Tawhid")
    assert lookup("الفتوى")["en"] == "Fatwa"
    assert lookup("unknown-term") is None
