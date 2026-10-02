"""Quran (KFGQPC Hafs v30), QuranEnc english_saheeh, and KFGQPC Muyassar tafsir -> SQLite. Idempotent."""
import csv
import io
import json
import re
import sqlite3
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.db import connect, replace_passages  # noqa: E402
from core.normalize import normalize_ar, search_form, strip_aya_end  # noqa: E402

RAW = ROOT / "data" / "raw"
TAG = re.compile(r"<[^>]+>")
FOOTNOTE_REF = re.compile(r"\[\d+\]")


def quran_url(s: int, a: int) -> str:
    return f"https://quranenc.com/en/browse/english_saheeh/{s}#{a}"


def load_quran():
    z = zipfile.ZipFile(RAW / "quran" / "kfgqpc_hafs_v30.zip")
    name = next(n for n in z.namelist() if n.endswith(".json"))
    return json.loads(z.read(name))


def load_translation():
    z = zipfile.ZipFile(RAW / "quran" / "english_saheeh.zip")
    name = next(n for n in z.namelist() if n.endswith(".sqlite"))
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "t.sqlite"
        p.write_bytes(z.read(name))
        con = sqlite3.connect(p)
        rows = con.execute("SELECT sura, aya, translation, footnotes FROM translations").fetchall()
        con.close()
    return {(s, a): (t, f) for s, a, t, f in rows}


def load_tafsir():
    z = zipfile.ZipFile(RAW / "tafsir" / "hafs_tafseerMouaser_v3.zip")
    name = next(n for n in z.namelist() if n.endswith(".csv"))
    rows = csv.DictReader(io.StringIO(z.read(name).decode("utf-8")))
    return {(int(r["sura_no"]), int(r["aya_no"])): r["aya_tafseer"] for r in rows}


def clean_tafsir(t: str) -> str:
    t = TAG.sub("", t)
    t = re.sub(r"^\s*\[\d+\]\s*", "", t)  # leading verse-number marker
    return re.sub(r"\s+", " ", t).strip()


def main():
    con = connect()
    q = load_quran()
    assert len(q) == 6236, f"Quran must have exactly 6236 verses, got {len(q)}"
    assert len({(v["sura_no"], v["aya_no"]) for v in q}) == 6236
    tr = load_translation()
    assert len(tr) == 6236, f"translation has {len(tr)} verses"
    taf = load_tafsir()
    assert len(taf) == 6236, f"tafsir has {len(taf)} verses"

    con.execute("DELETE FROM quran")
    qrows, pq = [], []
    for v in q:
        s, a = v["sura_no"], v["aya_no"]
        uth = strip_aya_end(v["aya_text_unicode"]).strip()
        emla = v["aya_text_emlaey"].strip()
        qrows.append((s, a, uth, emla, normalize_ar(emla), v["page"], v["jozz"],
                      v["sura_name_ar"].strip(), v["sura_name_en"]))
        en, foot = tr[(s, a)]
        en_clean = re.sub(r"^\(\d+\)\s*", "", en or "").strip()
        pq.append({"id": f"quran:{s}:{a}", "type": "quran", "source": "kfgqpc", "title": f"{v['sura_name_ar'].strip()} {a}",
                   "text_ar": uth, "text_en": en_clean,
                   "search_text": " ".join([search_form(emla), normalize_ar(re.sub(r"\[\d+\]", "", en_clean)),
                                            "سوره " + search_form(v["sura_name_ar"]), normalize_ar(v["sura_name_en"])]),
                   "reference_url": quran_url(s, a),
                   "meta": {"surah": s, "ayah": a, "page": v["page"], "juz": v["jozz"],
                            "surah_name_ar": v["sura_name_ar"].strip(), "surah_name_en": v["sura_name_en"],
                            "translation_source": "QuranEnc english_saheeh (Noor International Center)",
                            "footnotes": foot or ""}})
    con.executemany("INSERT INTO quran VALUES(?,?,?,?,?,?,?,?,?)", qrows)
    n1 = replace_passages(con, "kfgqpc", pq)

    pt = []
    for (s, a), t in sorted(taf.items()):
        txt = clean_tafsir(t)
        pt.append({"id": f"tafsir:muyassar:{s}:{a}", "type": "tafsir", "source": "muyassar",
                   "title": f"التفسير الميسر {s}:{a}", "text_ar": txt, "search_text": search_form(txt),
                   "reference_url": f"https://quranenc.com/ar/browse/arabic_moyassar/{s}#{a}",
                   "meta": {"surah": s, "ayah": a, "work": "التفسير الميسر (مجمع الملك فهد)"}})
    n2 = replace_passages(con, "muyassar", pt)
    con.commit()
    assert con.execute("SELECT count(*) FROM quran").fetchone()[0] == 6236
    print(f"quran verses: {len(qrows)}, quran passages: {n1}, tafsir passages: {n2}")


if __name__ == "__main__":
    main()
