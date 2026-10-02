"""Phase 0: probe every external API, save samples to data/samples/, print a report."""
import json
import re
import time
from pathlib import Path

import ssl

import httpx
import truststore

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "data" / "samples"
SAMPLES.mkdir(parents=True, exist_ok=True)
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"
client = httpx.Client(verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT), headers={"User-Agent": UA, "Accept": "application/json"},
                      follow_redirects=True, timeout=40)
report = []


def probe(name, url, fname=None, params=None, text=False, keep=60000):
    t = time.time()
    try:
        r = client.get(url, params=params)
        body = r.text
        ok = r.status_code == 200
        info = {"name": name, "url": str(r.url), "status": r.status_code,
                "type": r.headers.get("content-type", ""), "bytes": len(r.content),
                "secs": round(time.time() - t, 2),
                "ratelimit": {k: v for k, v in r.headers.items()
                              if "rate" in k.lower() or k.lower() == "retry-after"}}
        if fname:
            (SAMPLES / fname).write_text(body[:keep], encoding="utf-8")
        if ok and not text:
            try:
                info["json"] = r.json()
            except Exception:
                info["json"] = None
    except Exception as e:  # network failure is a result, not a crash
        info = {"name": name, "url": url, "status": "ERR", "error": repr(e)}
    report.append(info)
    flag = "OK " if info["status"] == 200 else "FAIL"
    print(f"[{flag}] {name}: {info['status']} {info.get('type', '')} {info.get('bytes', '')}B")
    return info


def keys(o, depth=1):
    if isinstance(o, dict):
        return {k: (keys(v, depth - 1) if depth > 0 else type(v).__name__) for k, v in list(o.items())[:12]}
    if isinstance(o, list):
        return [keys(o[0], depth)] if o else []
    return type(o).__name__


# ---- HadeethEnc
H = "https://hadeethenc.com/api/v1"
probe("hadeethenc.languages", f"{H}/languages", "hadeethenc_languages.json")
cats = probe("hadeethenc.categories", f"{H}/categories/list/", "hadeethenc_categories_en.json", {"language": "en"})
lst = probe("hadeethenc.list", f"{H}/hadeeths/list/", "hadeethenc_list.json",
            {"language": "en", "category_id": 1, "page": 1, "per_page": 5})
probe("hadeethenc.one.ar", f"{H}/hadeeths/one/", "hadeethenc_one_ar.json", {"language": "ar", "id": 2962})
probe("hadeethenc.one.en", f"{H}/hadeeths/one/", "hadeethenc_one_en.json", {"language": "en", "id": 2962})

# ---- QuranEnc
Q = "https://quranenc.com/api/v1"
tl = probe("quranenc.list", f"{Q}/translations/list", "quranenc_list.json", keep=200000)
probe("quranenc.english_saheeh.1", f"{Q}/translation/sura/english_saheeh/1", "quranenc_english_saheeh_1.json")
probe("quranenc.arabic_moyassar.1", f"{Q}/translation/sura/arabic_moyassar/1", "quranenc_arabic_moyassar_1.json")
probe("quranenc.aya", f"{Q}/translation/aya/english_saheeh/2/255", "quranenc_aya_2_255.json")

# ---- Dorar (hadith search, grades)
probe("dorar.search", "https://dorar.net/dorar_api.json", "dorar_search.json",
      {"skey": "إنما الأعمال بالنيات"})
probe("dorar.search.page2", "https://dorar.net/dorar_api.json", None, {"skey": "إنما الأعمال بالنيات", "page": 2})

# ---- mp3quran
M = "https://www.mp3quran.net/api/v3"
probe("mp3quran.reciters", f"{M}/reciters", "mp3quran_reciters.json", {"language": "ar"}, keep=30000)
probe("mp3quran.suwar", f"{M}/suwar", "mp3quran_suwar.json", {"language": "ar"}, keep=8000)

# ---- icadb (OpenAPI spec first, then endpoints of interest)
spec = probe("icadb.openapi", "https://icadb.com/api/docs/", "icadb_openapi.json", {"format": "openapi"}, keep=500000)
for name, path, p in [
    ("icadb.languages", "/api/languages/list/", None),
    ("icadb.encyclopedias", "/api/encyclopedias/list/", None),
    ("icadb.quran.surahs", "/quran/api/quran/surahs/", None),
    ("icadb.quran.translation-keys", "/quran/api/quran/translation-keys/", None),
    ("icadb.books", "/api/books/list/", None),
    ("icadb.lookup-tables", "/api/lookup-tables/", None),
]:
    probe(name, "https://icadb.com" + path, name.replace(".", "_") + ".json", p, keep=30000)

# ---- Reachability only
probe("qurancomplex.quran-dev", "https://qurancomplex.gov.sa/quran-dev", None, text=True)
probe("dawa.center.file7937", "https://dawa.center/file/7937", None, text=True)
probe("mcp.islamiccontent.org", "https://mcp.islamiccontent.org", None, text=True)

# ---- Summary of shapes
print("\n==== SHAPES ====")
for r in report:
    j = r.get("json")
    if j is not None:
        print(f"\n{r['name']}\n  keys: {json.dumps(keys(j, 1), ensure_ascii=False)[:400]}")
    if r["name"] == "dorar.search":
        txt = client.get("https://dorar.net/dorar_api.json", params={"skey": "إنما الأعمال بالنيات"}).text
        print("\ndorar.search raw head:", txt[:300])
    if r["name"] == "icadb.openapi" and j:
        print("  icadb paths:", len(j.get("paths", {})), "security:", j.get("securityDefinitions"))
print("\n==== SUMMARY ====")
for r in report:
    print(f"{r['status']!s:5} {r['name']:34} {r.get('ratelimit') or ''}")
(SAMPLES / "_probe_report.json").write_text(
    json.dumps([{k: v for k, v in r.items() if k != "json"} for r in report], ensure_ascii=False, indent=1),
    encoding="utf-8")
