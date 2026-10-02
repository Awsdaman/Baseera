"""Direct references: famous verse names ("آية الكرسي") and surah names ("سورة الإخلاص") resolve to verse ids."""
import re
import unicodedata

from core.db import connect
from core.normalize import normalize_ar

ALIASES = {  # normalized phrase -> (surah, first ayah, last ayah)
    "ايه الكرسي": (2, 255, 255), "اية الكرسي": (2, 255, 255), "ايه الكرسي": (2, 255, 255),
    "اخر ايتين من سوره البقره": (2, 285, 286), "خواتيم سوره البقره": (2, 285, 286), "خواتيم البقره": (2, 285, 286),
    "ايه النور": (24, 35, 35), "ايه الدين": (2, 282, 282), "ايه الوضوء": (5, 6, 6), "ايه الصيام": (2, 183, 183),
    "ايه الحجاب": (24, 31, 31), "ايه المباهله": (3, 61, 61), "ايه الربا": (2, 275, 275),
    "ayat al kursi": (2, 255, 255), "ayat al-kursi": (2, 255, 255), "throne verse": (2, 255, 255), "verse of the throne": (2, 255, 255),
}
MAX_VERSES = 7
_names = None


def _ascii(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return re.sub(r"[^a-z]", "", "".join(c for c in s if not unicodedata.combining(c)).lower().replace("ʿ", "").replace("’", ""))


def _surahs():
    global _names
    if _names is None:
        rows = connect().execute("SELECT surah, surah_name_ar, surah_name_en, count(*) n FROM quran GROUP BY surah").fetchall()
        _names = []
        for r in rows:
            ar = normalize_ar(r["surah_name_ar"])
            ar = ar[2:] if ar.startswith("ال") and len(ar) > 4 else ar
            _names.append({"surah": r["surah"], "ar": ar, "en": _ascii(r["surah_name_en"]), "n": r["n"]})
    return _names


def named_refs(query: str) -> list[str]:
    """Verse ids for alias phrases and for 'سورة X' / 'surah X' (short surahs in full, long ones first verses)."""
    q = normalize_ar(query)
    out: list[str] = []
    for phrase, (s, a1, a2) in ALIASES.items():
        if normalize_ar(phrase) in q:
            out += [f"quran:{s}:{a}" for a in range(a1, a2 + 1)]
    if out:  # an explicit alias beats the generic surah-name rule
        return list(dict.fromkeys(out))
    for m in re.finditer(r"(?:سوره|surah|sura|surat)\s+(?:ال)?([ء-ي]+|[a-z'\-]+(?:\s+[a-z\-]+)?)", q):
        word = m.group(1).strip()
        en = _ascii(word)
        for sn in _surahs():
            if (sn["ar"] and word.replace("ال", "", 1) == sn["ar"]) or (en and len(en) > 3 and en in (sn["en"], sn["en"].removeprefix("al"))):
                last = sn["n"] if sn["n"] <= MAX_VERSES else 3
                out += [f"quran:{sn['surah']}:{a}" for a in range(1, last + 1)]
                break
    return list(dict.fromkeys(out))
