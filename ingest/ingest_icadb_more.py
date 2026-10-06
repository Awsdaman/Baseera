"""More encyclopedias of the organizers' central database (icadb): words of the Quran, notable people, places, sects/religions, Names of Allah.

    python ingest/ingest_icadb_more.py

Each card becomes one passage (source icadb-<slug>); references fields are dropped from the text. Only approved (latest) card versions are used.
Dictionary-style encyclopedias become type "term" (words, particles); explanatory ones become type "qa". Hadith cards of icadb are NOT ingested:
hadith text and grades come only from HadeethEnc / Dorar (project rule 3). Idempotent.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.db import connect, replace_passages  # noqa: E402
from core.http import get_json  # noqa: E402
from core.normalize import search_form  # noqa: E402

I = "https://icadb.com"
# external id -> (slug, passage type, label, field used as the title)
ENCYCLOPEDIAS = {
    106: ("yaseer", "term", "المعجم اليسير (معاني ألفاظ القرآن)", "الكلمة"),
    107: ("huruf", "term", "حروف المعاني", None),
    108: ("aalam", "qa", "موسوعة الأعلام", "الإسم"),
    103: ("places", "qa", "موسوعة الأماكن", "اسم المكان"),
    109: ("firaq", "qa", "موسوعة الفرق والأديان", "المصطلح"),
    104: ("asma", "qa", "موسوعة الأسماء الحسنى", None),
}
SKIP_FIELD_WORDS = ("المراجع", "المصادر")


def cards(enc_ext_id: int):
    page, out = 1, []
    while True:
        d = get_json(f"{I}/api/encyclopedias/{enc_ext_id}/cards/latest/", {"page": page, "page_size": 200})
        out += d["cards"]
        if page >= d["total_pages"]:
            return out
        page += 1


def main():
    con = connect()
    for ext, (slug, ptype, label, title_field) in ENCYCLOPEDIAS.items():
        rows = []
        for c in cards(ext):
            fields = {s["field_name"]: (s["text"] or "").strip() for s in sorted(c.get("sentences", []), key=lambda s: s["field_order"])}
            title = (fields.get(title_field) if title_field else None) or c["name"]
            if ptype == "term" and slug in ("yaseer", "huruf"):
                # dictionary entries: "word: meaning" only. The field labels ("الكلمة", "الكتاب") would match every question that says "كلمة"
                word = fields.get("الكلمة") or fields.get("الحرف") or c["name"]
                text = f"{word}: {fields.get('المعنى', '')}".strip()
            else:
                body = [f"{k}: {v}" for k, v in fields.items() if v and v != "-" and not any(w in k for w in SKIP_FIELD_WORDS)]
                text = "\n".join(body)
            if len(text) < 20:
                continue
            rows.append({"id": f"{ptype}:icadb-{slug}:{c['external_id']}", "type": ptype, "source": f"icadb-{slug}", "title": title.strip()[:160],
                         "text_ar": text, "search_text": search_form(f"{title} {text}"),
                         "reference_url": f"{I}/api/encyclopedias/cards/{c['external_id']}/version/{c['latest_version']['version_str']}/sentences/",
                         "meta": {"encyclopedia": label, "card": c["external_id"], "kind": "encyclopedia"}})
        n = replace_passages(con, f"icadb-{slug}", rows)
        print(f"{label}: {n} passages", flush=True)


if __name__ == "__main__":
    main()
