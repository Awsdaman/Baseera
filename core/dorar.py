"""Dorar hadith search client (live, cached). Grades come ONLY from here / HadeethEnc, never the model.

API: GET https://dorar.net/dorar_api.json?skey=<text>[&page=n] -> {"ahadith": {"result": "<html>"}}
Each hit = <div class="hadith"> text + <div class="hadith-info"> (الراوي, المحدث, المصدر, الصفحة أو الرقم, خلاصة حكم المحدث).
Dorar exposes no hadith ID, so we derive a stable one from text + book + grader + page + narrator + grade (one entry per grading).
"""
import hashlib
import re

from bs4 import BeautifulSoup

from core.http import get_json
from core.normalize import normalize_ar

URL = "https://dorar.net/dorar_api.json"
LABELS = ["الراوي", "المحدث", "المصدر", "الصفحة أو الرقم", "خلاصة حكم المحدث"]
_NUM_PREFIX = re.compile(r"^\s*\d+\s*-\s*")
_ELLIPSIS = re.compile(r"(\s*\.\s*){2,}$")


def _info(div) -> dict:
    text = re.sub(r"\s+", " ", div.get_text(" ")).strip()
    positions = []
    for lab in LABELS:
        m = re.search(re.escape(lab) + r"\s*:", text)
        if m:
            positions.append((m.start(), m.end(), lab))
    positions.sort()
    out = {}
    for i, (_, end, lab) in enumerate(positions):
        nxt = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        out[lab] = text[end:nxt].strip()
    return out


def parse(html: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    hadiths = soup.select("div.hadith")
    infos = soup.select("div.hadith-info")
    out = []
    for h, inf in zip(hadiths, infos):
        text = _ELLIPSIS.sub("", _NUM_PREFIX.sub("", h.get_text(" ").strip())).strip()
        text = re.sub(r"\s+", " ", text)
        info = _info(inf)
        source = info.get("المصدر", "")
        ident = "|".join([normalize_ar(text), normalize_ar(source), normalize_ar(info.get("المحدث", "")),
                          normalize_ar(info.get("الصفحة أو الرقم", "")),
                          normalize_ar(info.get("الراوي", "")), normalize_ar(info.get("خلاصة حكم المحدث", ""))])
        key = hashlib.sha1(ident.encode()).hexdigest()[:12]
        out.append({
            "source": "dorar", "type": "hadith", "id": f"hadith:dorar:{key}", "text": text,
            "grade": info.get("خلاصة حكم المحدث"), "grader": info.get("المحدث"), "narrator": info.get("الراوي"),
            "book": source, "page_or_number": info.get("الصفحة أو الرقم"),
            "reference_url": "https://dorar.net/hadith/search?q=" + text[:60].replace(" ", "+"),
        })
    return out


_down_until = 0.0
BREAKER_SECONDS = 300


def search(query: str, page: int = 1) -> list[dict]:
    """Live Dorar lookup with a circuit breaker: after a failed call Dorar is skipped for 5 minutes (instead of retrying with long waits on
    every hadith), so Verify mode still answers quickly from HadeethEnc when Dorar is blocked or offline. Cached queries never touch the network."""
    global _down_until
    params = {"skey": query}
    if page > 1:
        params["page"] = page
    import time
    if time.time() < _down_until and not _cached(params):
        raise RuntimeError("Dorar skipped (circuit breaker open after a recent failure)")
    try:
        data = get_json(URL, params, delay=1.0, retries=2)
    except Exception:
        _down_until = time.time() + BREAKER_SECONDS
        raise
    html = (data.get("ahadith") or {}).get("result", "")
    return parse(html)


def _cached(params: dict) -> bool:
    import hashlib
    import json

    from core.http import CACHE
    key = hashlib.sha1((URL + json.dumps(params or {}, sort_keys=True)).encode()).hexdigest()
    return (CACHE / f"{key}.json").exists()
