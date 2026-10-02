"""Optional recitation audio for verse cards (mp3quran API, no key). Hafs 'an Asim, Mishary Alafasy by default."""
from core.http import get_json

API = "https://www.mp3quran.net/api/v3/reciters"
DEFAULT_RECITER = 123  # مشاري العفاسي


def surah_audio(surah: int, reciter_id: int = DEFAULT_RECITER) -> dict:
    data = get_json(API, {"language": "ar", "reciter": reciter_id})
    for r in data.get("reciters", []):
        for m in r.get("moshaf", []):
            if "حفص" in m["name"] and str(surah) in m["surah_list"].split(","):
                return {"surah": surah, "reciter": r["name"], "riwaya": m["name"],
                        "url": f"{m['server']}{surah:03d}.mp3", "source": "mp3quran.net"}
    return {"surah": surah, "url": None, "source": "mp3quran.net"}
