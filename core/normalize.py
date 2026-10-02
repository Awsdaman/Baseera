"""Arabic normalization for MATCHING only. Display always uses the original text."""
import re
import unicodedata

# Harakat, tanween, shadda, sukun, superscript alef (U+064B-065F, U+0670),
# Quranic annotation marks (U+06D6-06ED), extended Arabic marks (U+08D3-08FF), dagger/small marks.
_TASHKEEL = re.compile("[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed\u08d3-\u08ff]")
_TATWEEL = "\u0640"
_AYA_END = re.compile(r"\s*\u06dd\s*[\u0660-\u0669\d]*\s*$")
_ZW = re.compile("[\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]")
_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)
_WS = re.compile(r"\s+")
_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_KEEP = {"الله", "الذي", "الذين", "التي", "اللات", "الان", "الي"}
_LETTER_MAP = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ٲ": "ا", "ٳ": "ا",
    "ى": "ي", "ئ": "ي", "ی": "ي",
    "ة": "ه",
    "ؤ": "و",
    "ک": "ك",
})


def strip_tashkeel(text: str) -> str:
    return _TASHKEEL.sub("", text)


def strip_tatweel(text: str) -> str:
    return text.replace(_TATWEEL, "")


def strip_aya_end(text: str) -> str:
    """Remove the trailing ayah-end marker and number (۝١) from Mushaf text."""
    return _AYA_END.sub("", text)


def normalize_ar(text: str) -> str:
    """Strip tashkeel/tatweel/marks, unify alef forms, ى->ي, ة->ه (also ئ->ي, ؤ->و), collapse
    whitespace and punctuation. Latin text is lowercased so mixed strings behave sanely."""
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text)  # also folds Arabic presentation forms
    t = _ZW.sub("", t)
    t = strip_tashkeel(t)
    t = strip_tatweel(t)
    t = t.translate(_ARABIC_DIGITS).translate(_LETTER_MAP)
    t = _NON_WORD.sub(" ", t.lower())
    return _WS.sub(" ", t).strip()


def tokens(text: str) -> list[str]:
    return normalize_ar(text).split()


def light_stem(token: str) -> str:
    """Very light stemming for keyword search: drop a leading definite article / conjunction+article."""
    if token in _KEEP:
        return token
    for p in ("وال", "بال", "كال", "فال", "لل", "ال"):
        if token.startswith(p) and len(token) - len(p) >= 2:
            return token[len(p):]
    return token


def search_form(text: str) -> str:
    """Normalized + light-stemmed form used in the FTS index and for queries."""
    return " ".join(light_stem(t) for t in tokens(text))
