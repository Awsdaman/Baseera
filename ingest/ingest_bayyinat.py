"""Bayyinat 'selected questions about Islam' PDF -> one passage per question. Idempotent.

Passage text = question + short answer (مختصر الإجابة); the full answer text is kept in meta.full_text.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.db import connect, replace_passages  # noqa: E402
from core.normalize import search_form, strip_tashkeel  # noqa: E402
from ingest.pdf_text import open_pdf, page_lines  # noqa: E402

PDF = ROOT / "data" / "raw" / "bayyinat" / "bayyinat_qa.pdf"
URL = "https://dawa.center/file/7937"
HEADER = re.compile(r"^\d*\s*أسئلة منتقاة حول الإسلام")
MARKER = re.compile(r"\(\s*[^)\s]*\d+|\d+\s*-?\s*\)")


HEADERS = {"السؤال", "الجواب", "عبارات مشابهة للسؤال", "مضمون السؤال", "مختصر الإجابة", "مختصر الاجابة"}


def fix_punct(l: str) -> str:
    """Visual-order extraction leaves sentence punctuation at the line start; move it to the end."""
    l = l.strip()
    m = re.match(r"^([.:؛،؟!]+)\s*(.*)$", l)
    return (m.group(2) + m.group(1)) if m else l


def lines_clean(pages) -> list[tuple[int, str]]:
    out = []
    for pno, ls in enumerate(pages, 1):
        for l in ls:
            if HEADER.match(l) or l in ("الصفحة", "المسألة"):
                continue
            out.append((pno, l))
    return out


def find_starts(lines):
    starts = []
    for k, (p, l) in enumerate(lines):
        if l.strip() == "السؤال" and p > 20:
            j = k - 1
            for back in range(1, 7):
                if k - back >= 0 and MARKER.search(lines[k - back][1]) and "..." not in lines[k - back][1]:
                    j = k - back
                    break
            else:
                j = max(0, k - 3)
            starts.append((j, k))
    return starts


def clean_title(raw: str) -> str:
    t = re.sub(r"^(الم|المس|المسأ|المسألة)(?=[\sً-ْ.]|$)", "", raw.strip())
    t = re.sub(r"^[.:\s]+", "", t)
    return re.sub(r"\s+", " ", t).strip()


def section(body: list[str], names: tuple[str, ...]):
    """Return text from the first header whose plain text starts with one of names to the next header."""
    idx = None
    for i, l in enumerate(body):
        plain = strip_tashkeel(l).strip(" :.")
        if any(plain.startswith(n) and len(plain) < 40 for n in names):
            idx = i
            break
    if idx is None:
        return ""
    out = []
    for l in body[idx + 1:]:
        plain = strip_tashkeel(l).strip()
        bare = plain.strip(" :.")
        if (plain.startswith(":") or (plain.endswith(":") and len(plain) < 45) or bare in HEADERS
                or re.fullmatch(r"\d{1,2}", bare)):
            break
        out.append(fix_punct(l))
    return " ".join(out).strip()


def main():
    pages = [page_lines(p) for p in open_pdf(PDF)]
    lines = lines_clean(pages)
    starts = find_starts(lines)
    print("questions found:", len(starts))
    assert 250 <= len(starts) <= 280, "unexpected number of questions"
    rows = []
    for n, (j, k) in enumerate(starts, 1):
        end = starts[n][0] if n < len(starts) else len(lines)
        block = [l for _, l in lines[j:end]]
        sp, ep = lines[j][0] - 1, lines[end - 1][0] - 1  # book page numbers (PDF index - 1)
        title = clean_title(" ".join(l for _, l in lines[j + 1:k] if not MARKER.search(l)))
        qtext = section(block, ("السؤال",))
        short = section(block, ("مختصر الإجابة", "مختصر الاجابة", "الجواب المختصر"))
        if not short:
            ans = section(block, ("الجواب",))
            short = ans[:900]
        full = " ".join(fix_punct(l) for l in block)
        text_ar = f"{qtext}\n\n{short}".strip() if short else qtext
        if not title:
            title = qtext[:120]
        rows.append({
            "id": f"qa:bayyinat:{n}", "type": "qa", "source": "bayyinat", "title": title,
            "text_ar": text_ar, "search_text": search_form(f"{title} {text_ar}"),
            "reference_url": URL,
            "meta": {"number": n, "pages": [sp, ep], "work": "بينات: أسئلة منتقاة حول الإسلام",
                     "full_text": full}})
    con = connect()
    print("bayyinat passages:", replace_passages(con, "bayyinat", rows))
    empty = [r["id"] for r in rows if len(r["text_ar"]) < 60]
    print("short/empty passages:", len(empty), empty[:10])


if __name__ == "__main__":
    main()
