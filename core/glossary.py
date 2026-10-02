"""Approved glossary (docs/data.pdf p.7). These translations override machine translation."""
import re

from core.normalize import normalize_ar

GLOSSARY = [
    {"ar": "الإسلام", "en": "Islam",
     "note": "دين الاستسلام لله بالتوحيد والانقياد له بالطاعة، ويشرح بحسب السياق ولا يختزل في معنى ثقافي عام."},
    {"ar": "التوحيد", "en": "Tawhid / Oneness of God",
     "note": "يفضل إبقاء المصطلح مع شرح معناه: إفراد الله بالربوبية والألوهية ووصفه بما جاء الوحي به من أسمائه الحسنى، ولا يختزل في ترجمة قد توحي بمجرد الوحدانية العددية."},
    {"ar": "العبادة", "en": "Worship",
     "note": "تشمل أعمال القلب والقول والعمل التي يتقرب بها العبد إلى الله، ولا تحصر في الشعائر فقط."},
    {"ar": "النبوة", "en": "Prophethood",
     "note": "تستخدم للدلالة على اصطفاء الأنبياء بالوحي، مع التمييز بينها وبين القيادة الدينية البشرية."},
    {"ar": "الوحي", "en": "Revelation",
     "note": "يشرح بوصفه ما أوحاه الله إلى أنبيائه، مع تجنب استعمالات فضفاضة قد توهم الإلهام الشخصي."},
    {"ar": "الشريعة", "en": "Sharia / Islamic law and guidance",
     "note": "يشرح بحسب السياق ولا يختزل في العقوبات أو القانون الجنائي."},
    {"ar": "الحديث", "en": "Hadith",
     "note": "ما نقل عن النبي ﷺ من قول أو فعل أو تقرير ونحو ذلك، مع بيان درجة الثبوت عند الاستدلال."},
    {"ar": "السنة", "en": "Sunnah", "note": "هدي النبي ﷺ وطريقته، ويحدد المقصود بحسب السياق العلمي."},
    {"ar": "الفتوى", "en": "Fatwa", "note": "جواب شرعي يصدره مؤهل في واقعة أو سؤال، ولا يساوى بالمعلومة العامة."},
    {"ar": "الدعوة", "en": "Da‘wah / Invitation to Islam",
     "note": "التعريف بالإسلام والدعوة إليه بالحكمة، ويختار المقابل بحسب السياق والجمهور."},
]

# English aliases that should map to a glossary entry when a user writes in English / transliteration.
_ALIASES = {
    "islam": "الإسلام", "tawhid": "التوحيد", "tawheed": "التوحيد", "oneness of god": "التوحيد",
    "worship": "العبادة", "ibadah": "العبادة", "ibada": "العبادة", "prophethood": "النبوة", "nubuwwah": "النبوة",
    "revelation": "الوحي", "wahy": "الوحي", "sharia": "الشريعة", "shariah": "الشريعة", "islamic law": "الشريعة",
    "hadith": "الحديث", "hadeeth": "الحديث", "sunnah": "السنة", "sunna": "السنة", "fatwa": "الفتوى",
    "dawah": "الدعوة", "da'wah": "الدعوة", "dawa": "الدعوة",
}
_BY_AR = {normalize_ar(g["ar"]): g for g in GLOSSARY}


def lookup(term: str):
    """Find an approved glossary entry for an Arabic term or a common English/transliterated alias."""
    t = term.strip().lower()
    ar = _ALIASES.get(t) or t
    key = normalize_ar(ar)
    if key in _BY_AR:
        return _BY_AR[key]
    key2 = normalize_ar(re.sub(r"^ال", "", ar))
    for k, g in _BY_AR.items():
        if k == key2 or k == "ال" + key2:
            return g
    return None
