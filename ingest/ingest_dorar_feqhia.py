"""Dorar al-Saniyyah Fiqh Encyclopedia (https://dorar.net/feqhia, in the organizers' approved list; robots.txt allows crawling) -> passages.

    python ingest/ingest_dorar_feqhia.py --fetch [--limit N]   # polite crawl (about 2 requests/second), extracted text cached in data/cache/dorar_feqhia
    python ingest/ingest_dorar_feqhia.py --build               # chunk and write to the DB (type "qa", source "dorar-feqhia")

Each encyclopedia article is a question-style heading followed by the positions of the schools with their references: exactly the
"positions with attribution" content level ج needs. Only the page title and the article text are kept (navigation and ads are dropped);
every passage links back to its original page. Idempotent and resumable.
"""
import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import truststore
from bs4 import BeautifulSoup

truststore.inject_into_ssl()
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.db import connect, replace_passages  # noqa: E402
from core.normalize import search_form  # noqa: E402

BASE = "https://dorar.net"
CACHE = ROOT / "data" / "cache" / "dorar_feqhia"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36", "Accept-Language": "ar"}
DELAY = 0.5          # seconds between request starts (all workers together)
CHUNK = 800          # characters per passage (the generator is shown ~900 per passage)
MIN_CHARS = 80
_lock, _last = threading.Lock(), [0.0]


def _get(url: str) -> str:
    for attempt in range(4):
        with _lock:
            slot = max(time.time(), _last[0] + DELAY)
            _last[0] = slot
        time.sleep(max(0.0, slot - time.time()))
        try:
            r = httpx.get(url, headers=UA, timeout=30, follow_redirects=True)
            if r.status_code == 200:
                return r.text
            if r.status_code in (403, 429, 503):
                time.sleep(2 ** (attempt + 1))
                continue
            return ""
        except httpx.HTTPError:
            time.sleep(2 ** (attempt + 1))
    return ""


def article_ids() -> list[int]:
    f = CACHE / "_ids.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    ids = sorted({int(x) for x in re.findall(r'href="/feqhia/(\d+)"', _get(f"{BASE}/feqhia"))})
    f.write_text(json.dumps(ids), encoding="utf-8")
    return ids


def extract(html: str) -> dict | None:
    s = BeautifulSoup(html, "html.parser")
    title = (s.title.get_text(" ", strip=True) if s.title else "").replace("- الموسوعة الفقهية", "").replace("- الدرر السنية", "").strip(" -")
    parts = []
    for art in s.find_all("article"):
        for bad in art.select("script,style,button,.tab-content,nav,.btn"):
            bad.decompose()
        t = re.sub(r"\s+", " ", art.get_text(" ", strip=True)).strip()
        if len(t) >= MIN_CHARS and t not in parts:
            parts.append(t)
    if not parts:
        return None
    return {"title": title, "text": "\n".join(parts)}


def fetch(limit: int | None = None, workers: int = 4):
    CACHE.mkdir(parents=True, exist_ok=True)
    ids = article_ids()[:limit]
    todo = [i for i in ids if not (CACHE / f"{i}.json").exists()]
    print(f"{len(ids)} articles listed, {len(todo)} to fetch", flush=True)

    def one(i):
        html = _get(f"{BASE}/feqhia/{i}")
        data = extract(html) if html else None
        (CACHE / f"{i}.json").write_text(json.dumps(data or {"empty": True}, ensure_ascii=False), encoding="utf-8")
        return i, bool(data)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, i) for i in todo]
        for n, f in enumerate(as_completed(futs), 1):
            if n % 100 == 0:
                print(f"  {n}/{len(todo)}", flush=True)


def chunks(text: str) -> list[str]:
    sents = [x.strip() for x in re.split(r"(?<=[.!؟؛])\s+", text) if x.strip()]
    out, cur = [], ""
    for s in sents:
        while len(s) > CHUNK * 1.5:           # a very long sentence: cut at a space
            cut = s.rfind(" ", 0, CHUNK)
            cut = cut if cut > 200 else CHUNK
            if cur:
                out.append(cur); cur = ""
            out.append(s[:cut]); s = s[cut:].strip()
        if cur and len(cur) + len(s) > CHUNK:
            out.append(cur); cur = s
        else:
            cur = f"{cur} {s}".strip()
    if cur:
        out.append(cur)
    return [c for c in out if len(c) >= MIN_CHARS]


def build():
    rows = []
    for f in sorted(CACHE.glob("[0-9]*.json"), key=lambda p: int(p.stem)):
        d = json.loads(f.read_text(encoding="utf-8"))
        if d.get("empty") or not d.get("text"):
            continue
        for n, c in enumerate(chunks(d["text"]), 1):
            rows.append({"id": f"qa:dorar-feqhia:{f.stem}:{n}", "type": "qa", "source": "dorar-feqhia", "title": d["title"][:160],
                         "text_ar": c, "search_text": search_form(f"{d['title']} {c}"),
                         "reference_url": f"{BASE}/feqhia/{f.stem}",
                         "meta": {"encyclopedia": "الموسوعة الفقهية (الدرر السنية)", "article": int(f.stem), "part": n, "kind": "encyclopedia"}})
    n = replace_passages(connect(), "dorar-feqhia", rows)
    print(f"stored {n} passages from {len({r['meta']['article'] for r in rows})} articles")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    if a.fetch:
        fetch(a.limit)
    if a.build:
        build()
