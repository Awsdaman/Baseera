"""Opt-in problem reports: the ONLY place Baseera ever stores something a user typed.

Privacy design (docs/data.pdf p.5: no personal data unless needed, with a declared policy):
  * nothing is saved unless the user ticks the consent box next to the stated policy (add_report refuses otherwise);
  * only: the question, the app's outcome (status / level / intent / reason), a short excerpt of the answer, the cited source ids,
    the user's chosen reason and optional comment. No IP address, no user agent, no session or account id, no cookies;
  * e-mail addresses, phone numbers and URLs are scrubbed from the free text before saving;
  * reports are deleted automatically after REPORT_RETENTION_DAYS (default 90);
  * they are used only for human review: a reviewer can turn one into a test case, an approved rewrite or a source-gap note.
"""
import os
import re
import sqlite3
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REASONS = ("wrong_answer", "incomplete", "should_have_answered", "wrong_source", "wrong_verify_result", "other")
RETENTION_DAYS = int(os.environ.get("REPORT_RETENTION_DAYS", 90))

POLICY = {
    "ar": "أوافق على حفظ سؤالي وملخّص الإجابة لدى بصيرة لمراجعتهما يدويًّا وتحسين الأداة، وسيُحذفان تلقائيًا بعد 90 يومًا. لا يُحفظ أي شيء آخر عنّي.",
    "en": "I agree that Baseera saves my question and a summary of the answer for manual review to improve the tool; they are deleted automatically after 90 days. Nothing else about me is stored.",
}

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_URL = re.compile(r"https?://\S+|www\.\S+")
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)")

SCHEMA = """
CREATE TABLE IF NOT EXISTS reports(
  id INTEGER PRIMARY KEY AUTOINCREMENT, created TEXT NOT NULL,
  question TEXT NOT NULL, language TEXT, status TEXT, level TEXT, intent TEXT, abstain_reason TEXT,
  answer_excerpt TEXT, source_ids TEXT, reason TEXT NOT NULL, comment TEXT,
  review TEXT NOT NULL DEFAULT 'pending', review_note TEXT);
"""


def db_path() -> Path:
    return Path(os.environ.get("REPORTS_DB") or ROOT / "data" / "db" / "reports.sqlite")


def connect() -> sqlite3.Connection:
    p = db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(p)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def scrub(text: str | None, limit: int = 1000) -> str:
    """Remove e-mail addresses, phone numbers and URLs, then truncate."""
    t = _PHONE.sub("[number]", _URL.sub("[link]", _EMAIL.sub("[email]", text or "")))
    return t.strip()[:limit]


def add_report(question: str, reason: str, comment: str | None, outcome: dict, consent: bool) -> int:
    """Save one opt-in report. Raises ValueError without explicit consent or with an unknown reason."""
    if consent is not True:
        raise ValueError("consent is required: nothing is stored without it")
    if reason not in REASONS:
        raise ValueError(f"reason must be one of {REASONS}")
    q = scrub(question, 2000)
    if not q:
        raise ValueError("empty question")
    con = connect()
    cur = con.execute(
        "INSERT INTO reports(created, question, language, status, level, intent, abstain_reason, answer_excerpt, source_ids, reason, comment)"
        " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
        (time.strftime("%Y-%m-%d %H:%M", time.gmtime()), q, str(outcome.get("language") or "")[:8], str(outcome.get("status") or "")[:30],
         str(outcome.get("level") or "")[:4], str(outcome.get("intent") or "")[:20], str(outcome.get("abstain_reason") or "")[:60],
         scrub(outcome.get("answer_text"), 600), ",".join(str(s)[:80] for s in (outcome.get("source_ids") or [])[:20]),
         reason, scrub(comment, 500)))
    con.commit()
    return cur.lastrowid


def list_reports(review: str | None = "pending") -> list[dict]:
    con = connect()
    rows = con.execute("SELECT * FROM reports" + (" WHERE review=?" if review else "") + " ORDER BY id", (review,) if review else ()).fetchall()
    return [dict(r) for r in rows]


def get_report(rid: int) -> dict | None:
    r = connect().execute("SELECT * FROM reports WHERE id=?", (rid,)).fetchone()
    return dict(r) if r else None


def set_review(rid: int, review: str, note: str = "") -> bool:
    con = connect()
    cur = con.execute("UPDATE reports SET review=?, review_note=? WHERE id=?", (review, note[:300], rid))
    con.commit()
    return cur.rowcount == 1


def purge(days: int | None = None) -> int:
    """Delete reports older than `days` (retention). Returns how many were removed."""
    days = RETENTION_DAYS if days is None else days
    cutoff = time.strftime("%Y-%m-%d %H:%M", time.gmtime(time.time() - days * 86400))
    con = connect()
    cur = con.execute("DELETE FROM reports WHERE created < ?", (cutoff,))
    con.commit()
    return cur.rowcount
