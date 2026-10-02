"""icadb Q&A (encyclopedias 102, 110) and terminology (105) + approved glossary (data.pdf p.7) -> SQLite.

Only approved card versions are ingested (icadb default). Idempotent; responses cached.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.db import connect, replace_passages  # noqa: E402
from core.glossary import GLOSSARY  # noqa: E402
from core.normalize import normalize_ar, search_form  # noqa: E402
from core.http import get_json  # noqa: E402

I = "https://icadb.com"
VERSE_QUOTE = re.compile("﴿.*?﴾")  # ornate parentheses ﴿ ... ﴾ around Quran quotations


def cards(enc_ext_id: int):
    page, out = 1, []
    while True:
        d = get_json(f"{I}/api/encyclopedias/{enc_ext_id}/cards/latest/", {"page": page, "page_size": 200})
        out += d["cards"]
        if page >= d["total_pages"]:
            return out
        page += 1


def fields(card) -> dict:
    return {s["field_name"]: s["text"] for s in sorted(card.get("sentences", []), key=lambda s: s["field_order"])}


def main():
    con = connect()
    qa_rows = []
    for enc, label, audience in ((102, "أسئلة وأجوبة للمسلمين", "muslims"), (110, "أسئلة وأجوبة لغير المسلمين", "non_muslims")):
        for c in cards(enc):
            f = fields(c)
            q = (f.get("السؤال") or f.get("العنوان") or c["name"]).strip()
            a = (f.get("الجواب") or "").strip()
            if not a:
                continue
            title = (f.get("العنوان") or c["name"]).strip()
            text = f"{q}\n\n{a}"
            qa_rows.append({
                "id": f"qa:icadb:{c['external_id']}", "type": "qa", "source": "icadb", "title": title, "text_ar": text,
                "search_text": search_form(f"{title} {text}"),
                "reference_url": f"https://icadb.com/api/encyclopedias/cards/{c['external_id']}/version/{c['latest_version']['version_str']}/sentences/",
                "meta": {"audience": audience, "encyclopedia": label, "card": c["external_id"],
                         "has_verse_quote": bool(VERSE_QUOTE.search(text))}})
    n1 = replace_passages(con, "icadb", qa_rows)

    term_rows = []
    for c in cards(105):
        f = fields(c)
        defin = " ".join(v for k, v in f.items() if k != "المُصْطَلَح" and v)
        term_rows.append({
            "id": f"term:icadb:{c['external_id']}", "type": "term", "source": "icadb-terms", "title": c["name"],
            "text_ar": defin, "search_text": search_form(f"{c['name']} {defin}"),
            "reference_url": "https://terminologyenc.com", "meta": {"card": c["external_id"], "fields": f}})
    n2 = replace_passages(con, "icadb-terms", term_rows)

    g_rows = []
    for i, g in enumerate(GLOSSARY, 1):
        g_rows.append({
            "id": f"term:glossary:{i}", "type": "term", "source": "glossary", "title": g["ar"],
            "text_ar": g["note"], "text_en": g["en"],
            "search_text": search_form(f"{g['ar']} {g['note']}") + " " + normalize_ar(g["en"]),
            "reference_url": "docs/data.pdf#page=7", "meta": {"approved_en": g["en"]}})
    n3 = replace_passages(con, "glossary", g_rows)
    print(f"icadb Q&A: {n1}, terms: {n2}, glossary: {n3}")


if __name__ == "__main__":
    main()
