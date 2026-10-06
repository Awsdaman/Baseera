"""icadb books (the organizers' central database, https://icadb.com/api/books/) -> passages of type "qa", source "icadb-books".

    python ingest/ingest_icadb_books.py --fetch     # download the Arabic text of every book (cached in data/cache, polite rate)
    python ingest/ingest_icadb_books.py --build     # chunk by chapter, drop duplicates across editions, write to the DB
    python ingest/ingest_icadb_books.py --stats     # how many chunks per book / in total

Only the Arabic text of the latest Arabic version is used; footnotes are skipped (they are references, not explanation). Chunks are
~700 characters, never cross a chapter, and identical text published in several editions is stored once. Idempotent.
"""
import argparse
import hashlib
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.db import connect, replace_passages  # noqa: E402
from core.http import get_json  # noqa: E402
from core.normalize import normalize_ar, search_form  # noqa: E402

I = "https://icadb.com"
TEXT_TYPES = {"نص", "حديث", "آية", "أثر"}   # body text, hadith, verse, athar as they appear inside a book
CHUNK = 700
MIN_CHARS = 60


def list_books() -> list[dict]:
    d = get_json(f"{I}/api/books/list/")
    return d if isinstance(d, list) else d.get("results", d)


def fetch_all(workers: int = 6):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    books = list_books()
    print(len(books), "books", flush=True)

    def one(b):
        d = get_json(f"{I}/api/books/{b['project_id']}/arabic-phrases/")
        return b, d.get("phrases_count", 0)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, b): b for b in books}
        for n, f in enumerate(as_completed(futs), 1):
            b = futs[f]
            try:
                _, cnt = f.result()
                print(f"[{n}/{len(books)}] {b['project_id']:>4} {cnt:>5} phrases  {b['title'][:60]}", flush=True)
            except Exception as e:
                print(f"[{n}/{len(books)}] {b['project_id']} FAILED {str(e)[:80]}", flush=True)


def book_chunks(book: dict) -> list[dict]:
    d = get_json(f"{I}/api/books/{book['project_id']}/arabic-phrases/")
    ver = (d.get("latest_arabic_version") or {}).get("version_name")
    chapters: dict[int, dict] = {}
    for p in d.get("phrases", []):
        if p.get("footnote_id") or p.get("text_type") == "footnote" or p.get("content_type_name") not in TEXT_TYPES:
            continue
        t = (p.get("arabic_text") or "").strip()
        if not t:
            continue
        ch = chapters.setdefault(p["chapter_serial"], {"name": p.get("chapter_name") or "", "parts": []})
        ch["parts"].append(t)
    out = []
    for serial, ch in sorted(chapters.items()):
        cur = ""
        for part in ch["parts"]:
            if cur and len(cur) + len(part) > CHUNK:
                out.append({"chapter": ch["name"], "serial": serial, "text": cur})
                cur = part
            else:
                cur = f"{cur} {part}".strip()
        if cur:
            out.append({"chapter": ch["name"], "serial": serial, "text": cur})
    for n, c in enumerate(out, 1):
        c["n"], c["version"] = n, ver
    return [c for c in out if len(c["text"]) >= MIN_CHARS]


def build_rows(books: list[dict]) -> list[dict]:
    rows, seen = [], set()
    for b in books:
        try:
            chunks = book_chunks(b)
        except Exception as e:
            print("skip", b["project_id"], str(e)[:60])
            continue
        for c in chunks:
            h = hashlib.sha1(normalize_ar(c["text"]).encode("utf-8")).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            btitle = (b.get("title") or "").strip() or (b.get("name") or "").strip()      # 145 of the 345 books have an empty `title` but a `name`
            title = f"{btitle} — {c['chapter'].strip()}" if c["chapter"].strip() and c["chapter"].strip() != btitle else btitle
            rows.append({"id": f"qa:icadb-book:{b['project_id']}:{c['n']}", "type": "qa", "source": "icadb-books", "title": title[:160],
                         "text_ar": c["text"], "search_text": search_form(f"{title} {c['text']}"),
                         "reference_url": f"{I}/api/books/{b['project_id']}/arabic-phrases/",
                         "meta": {"book": btitle, "chapter": c["chapter"], "project_id": b["project_id"], "version": c["version"],
                                  "kind": "book"}})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--stats", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch_all()
    if a.stats or a.build:
        rows = build_rows(list_books())
        print(f"{len(rows)} unique chunks, {sum(len(r['text_ar']) for r in rows) / 1e6:.1f} M characters")
        if a.build:
            con = connect()
            n = replace_passages(con, "icadb-books", rows)
            print("stored", n, "passages (source icadb-books)")


if __name__ == "__main__":
    main()
