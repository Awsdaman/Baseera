"""Bayyinat PDF text extraction with repair of reversed Arabic ligature clusters.

The PDF font emits lam-alef style ligatures (and the Allah ligature) as zero-width chars followed by one
wide char, in reverse order. We rebuild each line from glyph boxes and reverse those clusters.
"""
import unicodedata

import pymupdf

ZERO = 0.05


def _is_mark(ch: str) -> bool:
    return unicodedata.category(ch) == "Mn"


def fix_chars(chars: list[dict]) -> str:
    out, i = [], 0
    n = len(chars)
    while i < n:
        c = chars[i]
        w = c["bbox"][2] - c["bbox"][0]
        if w < ZERO and not _is_mark(c["c"]) and c["c"].strip():
            # collect zero-width letters, then the wide carrier letter
            j = i
            cluster = []
            while j < n and (chars[j]["bbox"][2] - chars[j]["bbox"][0]) < ZERO and not _is_mark(chars[j]["c"]) and chars[j]["c"].strip():
                cluster.append(chars[j]["c"])
                j += 1
            if j < n and not _is_mark(chars[j]["c"]) and chars[j]["c"].strip():
                cluster.append(chars[j]["c"])
                out.extend(reversed(cluster))
                i = j + 1
                continue
            out.extend(cluster)
            i = j
            continue
        out.append(c["c"])
        i += 1
    return "".join(out)


def page_lines(page) -> list[str]:
    raw = page.get_text("rawdict")
    lines = []
    for b in raw["blocks"]:
        for l in b.get("lines", []):
            chars = [ch for s in l["spans"] for ch in s["chars"]]
            t = fix_chars(chars).strip()
            if t:
                lines.append(t)
    return lines


def open_pdf(path):
    return pymupdf.open(path)
