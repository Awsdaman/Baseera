"""HadeethEnc full dataset (Arabic + English) -> SQLite. Idempotent; API responses cached in data/cache.

License reminder: content must not be modified and HadeethEnc.com must be credited in the UI.
"""
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.db import connect, replace_passages  # noqa: E402
from core.normalize import normalize_ar, search_form  # noqa: E402
from core.http import get_json  # noqa: E402

H = "https://hadeethenc.com/api/v1"
BATCH = 50


def as_list(v):
    if isinstance(v, list):
        return v
    try:
        return ast.literal_eval(v) if v else []
    except Exception:
        return [v] if v else []


def all_ids() -> list[str]:
    cats = get_json(f"{H}/categories/list/", {"language": "en"})
    ids = set()
    for c in (c for c in cats if not c["parent_id"]):
        page = 1
        while True:
            d = get_json(f"{H}/hadeeths/list/", {"language": "ar", "category_id": c["id"], "per_page": 1000, "page": page})
            ids |= {x["id"] for x in d["data"]}
            if page >= int(d["meta"]["last_page"]):
                break
            page += 1
    return sorted(ids, key=int)


def fetch(ids, lang):
    out = {}
    for i in range(0, len(ids), BATCH):
        chunk = ids[i:i + BATCH]
        for h in get_json(f"{H}/hadeeths/multiple/", {"ids": ",".join(chunk), "language": lang}):
            out[h["id"]] = h
        if (i // BATCH) % 10 == 0:
            print(f"  {lang}: {len(out)}/{len(ids)}", flush=True)
    return out


def main():
    ids = all_ids()
    print("hadith ids:", len(ids))
    ar, en = fetch(ids, "ar"), fetch(ids, "en")
    rows = []
    for i in ids:
        a, e = ar.get(i), en.get(i)
        if not a:
            print("skip (no Arabic)", i)
            continue
        e = e or {}  # some hadiths have no English translation: keep them Arabic-only
        text_ar = a["hadeeth"].strip()
        text_en = (e.get("hadeeth") or "").strip() or None
        search = " ".join([search_form(a["title"]), search_form(text_ar), search_form(a.get("explanation") or ""),
                           normalize_ar(e.get("title") or ""), normalize_ar(text_en or "")])
        rows.append({
            "id": f"hadith:hadeethenc:{i}", "type": "hadith", "source": "hadeethenc",
            "title": a["title"], "text_ar": text_ar, "text_en": text_en, "search_text": search,
            "grade": a["grade"], "reference_url": f"https://hadeethenc.com/ar/browse/hadith/{i}",
            "meta": {"title_en": e.get("title"), "grade_en": e.get("grade"), "attribution_ar": a["attribution"],
                     "attribution_en": e.get("attribution"), "reference": a.get("reference", ""),
                     "explanation_ar": a.get("explanation", ""), "explanation_en": e.get("explanation", ""),
                     "hints_ar": as_list(a.get("hints")), "hints_en": as_list(e.get("hints")),
                     "credit": "HadeethEnc.com"}})
    con = connect()
    print("hadith passages:", replace_passages(con, "hadeethenc", rows))


if __name__ == "__main__":
    main()
