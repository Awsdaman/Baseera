from core.aliases import named_refs
from core import retrieve as R


def test_ayat_al_kursi_alias():
    assert named_refs("ما معنى آية الكرسي؟") == ["quran:2:255"]
    assert named_refs("Explain the Throne Verse") == ["quran:2:255"]


def test_short_surah_by_name():
    assert named_refs("ما هي سورة الإخلاص؟") == [f"quran:112:{a}" for a in (1, 2, 3, 4)]
    assert named_refs("سورة الفاتحة") == [f"quran:1:{a}" for a in range(1, 8)]
    assert named_refs("tell me about surah al-ikhlas")[:1] == ["quran:112:1"]


def test_long_surah_returns_first_verses_only():
    assert named_refs("سورة البقرة") == ["quran:2:1", "quran:2:2", "quran:2:3"]


def test_last_two_verses_of_baqarah():
    assert named_refs("آخر آيتين من سورة البقرة") == ["quran:2:285", "quran:2:286"]


def test_no_false_positive():
    assert named_refs("ما هي أركان الإسلام؟") == []


def test_retrieve_resolves_names_first_without_vectors():
    ids = [r["id"] for r in R.retrieve("ما هي سورة الإخلاص؟", vectors=False)]
    assert ids[:4] == ["quran:112:1", "quran:112:2", "quran:112:3", "quran:112:4"]
    assert "quran:2:255" == R.retrieve("آية الكرسي", vectors=False)[0]["id"]


def test_explicit_reference():
    assert [r["id"] for r in R.retrieve("tafsir of 2:255-256", vectors=False)][:2] == ["quran:2:255", "quran:2:256"]
