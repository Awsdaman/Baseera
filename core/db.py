"""SQLite layer: authoritative text tables plus a unified `passages` table with an FTS5 index."""
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "db" / "baseera.sqlite"

SCHEMA = """
CREATE TABLE IF NOT EXISTS quran(
  surah INTEGER, ayah INTEGER, text_uthmani TEXT NOT NULL, text_emlaey TEXT NOT NULL,
  text_norm TEXT NOT NULL, page INTEGER, juz INTEGER, surah_name_ar TEXT, surah_name_en TEXT,
  PRIMARY KEY(surah, ayah));
CREATE TABLE IF NOT EXISTS passages(
  id TEXT PRIMARY KEY,            -- quran:2:255 | tafsir:muyassar:2:255 | hadith:hadeethenc:2962 | qa:bayyinat:17 ...
  type TEXT NOT NULL,             -- quran | tafsir | hadith | qa | term
  source TEXT NOT NULL,           -- kfgqpc | muyassar | hadeethenc | bayyinat | icadb | glossary
  title TEXT, text_ar TEXT, text_en TEXT,
  search_text TEXT NOT NULL,      -- normalized + light-stemmed text used by FTS
  grade TEXT, reference_url TEXT, meta TEXT);
CREATE INDEX IF NOT EXISTS passages_type ON passages(type);
CREATE VIRTUAL TABLE IF NOT EXISTS passages_fts USING fts5(
  id UNINDEXED, search_text, tokenize='unicode61 remove_diacritics 0');
"""


def connect(path: Path | str = DB_PATH) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def replace_passages(con: sqlite3.Connection, source: str, rows: list[dict]) -> int:
    """Idempotent: delete everything from `source`, then insert `rows`."""
    ids = [r[0] for r in con.execute("SELECT id FROM passages WHERE source=?", (source,))]
    con.executemany("DELETE FROM passages_fts WHERE id=?", [(i,) for i in ids])
    con.execute("DELETE FROM passages WHERE source=?", (source,))
    for r in rows:
        meta = json.dumps(r.get("meta") or {}, ensure_ascii=False)
        con.execute(
            "INSERT INTO passages(id,type,source,title,text_ar,text_en,search_text,grade,reference_url,meta)"
            " VALUES(?,?,?,?,?,?,?,?,?,?)",
            (r["id"], r["type"], source, r.get("title"), r.get("text_ar"), r.get("text_en"),
             r["search_text"], r.get("grade"), r.get("reference_url"), meta))
        con.execute("INSERT INTO passages_fts(id,search_text) VALUES(?,?)", (r["id"], r["search_text"]))
    con.commit()
    return len(rows)
