"""Synthetic labelled data for verify mode (free, deterministic: no LLM). Generates perturbed verses and hadith with EXACT labels,
runs the real matchers, and reports confusion matrices plus a threshold sweep (the 'tune theta on a labelled set' step).

    python evals/synth_verify.py [--verses 250] [--hadith 250] [--seed 7] [--only verses|hadith]

Verse classes   : verified  = exact text or an exact fragment / multi-verse run (any tashkeel, Uthmani or Imla'i spelling)
                  misquoted = one or two words substituted / deleted / inserted / swapped in a long-enough verse
                  fabricated= splice of two different verses, heavily edited text, or ordinary non-Quran Arabic
Hadith (local HadeethEnc only; Dorar is live and not part of this test):
                  match = exact fragment or lightly edited text ; no-match = unrelated Arabic text
Caveat: synthetic edits are easier than real-world misquotes (people mis-remember in more natural ways). Use the numbers to
choose thresholds and to catch regressions, not as a claim about real viral messages.
"""
import argparse
import collections
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rapidfuzz import fuzz  # noqa: E402

from core import verifier_mode as V  # noqa: E402
from core.db import connect  # noqa: E402
from core.normalize import normalize_ar  # noqa: E402

OUT = Path(__file__).with_name("synth_failures.jsonl")


def load_verses():
    rows = connect().execute("SELECT surah, ayah, text_uthmani, text_emlaey FROM quran ORDER BY surah, ayah").fetchall()
    return [dict(r) for r in rows]


def verse_cases(n: int, rnd: random.Random):
    verses = load_verses()
    by_pos = {(v["surah"], v["ayah"]): v for v in verses}
    vocab = [w for v in verses for w in v["text_emlaey"].split()]
    long_v = [v for v in verses if len(v["text_emlaey"].split()) >= 12]
    mid_v = [v for v in verses if 8 <= len(v["text_emlaey"].split())]
    any_v = [v for v in verses if len(v["text_emlaey"].split()) >= 5]
    cases = []

    def add(kind, label, text, v):
        cases.append({"kind": kind, "label": label, "text": text, "ref": f"{v['surah']}:{v['ayah']}"})

    for v in rnd.sample(any_v, n):
        w = v["text_emlaey"].split()
        add("exact_full", "verified", v["text_emlaey"], v)
        add("exact_uthmani", "verified", v["text_uthmani"], v)
        if len(w) >= 6:
            a = rnd.randint(0, len(w) - 4)
            b = rnd.randint(a + 3, min(len(w), a + 12))
            add("fragment", "verified", " ".join(w[a:b]), v)
        nxt = by_pos.get((v["surah"], v["ayah"] + 1))
        if nxt and len(w) + len(nxt["text_emlaey"].split()) <= 28:
            add("two_verses", "verified", v["text_emlaey"] + " " + nxt["text_emlaey"], v)
    for v in rnd.sample(mid_v, n):
        w = v["text_emlaey"].split()
        i = rnd.randrange(len(w))
        add("sub1", "misquoted", " ".join(w[:i] + [rnd.choice(vocab)] + w[i + 1:]), v)
        k = rnd.randint(1, len(w) - 2)  # interior word only: deleting the first/last word leaves a valid fragment
        add("del1", "misquoted", " ".join(w[:k] + w[k + 1:]), v)
        add("ins1", "misquoted", " ".join(w[:i] + [rnd.choice(vocab)] + w[i:]), v)
        j = rnd.randrange(len(w) - 1)
        sw = w[:]
        sw[j], sw[j + 1] = sw[j + 1], sw[j]
        if sw != w:
            add("swap", "misquoted", " ".join(sw), v)
    for v in rnd.sample(long_v, n):
        w = v["text_emlaey"].split()
        i, j = rnd.sample(range(len(w)), 2)
        e = w[:]
        e[i], e[j] = rnd.choice(vocab), rnd.choice(vocab)
        add("sub2", "misquoted", " ".join(e), v)
    for v in rnd.sample(long_v, n):
        o = rnd.choice(long_v)
        if o["surah"] == v["surah"]:
            continue
        w, x = v["text_emlaey"].split(), o["text_emlaey"].split()
        add("splice", "fabricated", " ".join(w[: len(w) // 2] + x[len(x) // 2:]), v)
        half = w[:]
        for k in rnd.sample(range(len(w)), len(w) // 2):
            half[k] = rnd.choice(vocab)
        add("half_replaced", "fabricated", " ".join(half), v)
    return cases


def foreign_arabic(n: int, rnd: random.Random, min_words=8, max_words=14):
    """Ordinary non-Quran Arabic windows (HadeethEnc explanations / icadb answers), excluding any window that fuzzily matches a verse."""
    con = connect()
    rows = [r[0] for r in con.execute("SELECT json_extract(meta, '$.explanation_ar') FROM passages WHERE source='hadeethenc' AND meta LIKE '%explanation_ar%'") if r[0]]
    rows += [r[0] for r in con.execute("SELECT text_ar FROM passages WHERE source='icadb'") if r[0]]
    out = []
    while len(out) < n:
        words = rnd.choice(rows).split()
        if len(words) < min_words + 4:
            continue
        k = rnd.randint(min_words, max_words)
        a = rnd.randint(0, len(words) - k)
        out.append(" ".join(words[a:a + k]))
    return out


def run_verses(n: int, rnd: random.Random):
    cases = verse_cases(n, rnd)
    for t in foreign_arabic(n, rnd):
        cases.append({"kind": "foreign_text", "label": "fabricated", "text": t, "ref": None})
    t0 = time.time()
    for c in cases:
        r = V.check_verse(c["text"])
        c["pred"] = {"verified": "verified", "misquoted": "misquoted", "fabricated": "fabricated"}.get(r["verdict"], r["verdict"])
        c["score"] = r.get("score")
        c["match"] = (r.get("match") or {}).get("ref")
    secs = time.time() - t0
    labels = ["verified", "misquoted", "fabricated"]
    cm = {a: collections.Counter() for a in labels}
    for c in cases:
        cm[c["label"]][c["pred"]] += 1
    print(f"\n=== VERSES: {len(cases)} cases, {secs / len(cases) * 1000:.0f} ms each (threshold VERSE_MISQUOTE_MIN={V.VERSE_MISQUOTE_MIN})")
    print("  confusion (rows = truth, cols = predicted)")
    print("  " + " " * 11 + "".join(f"{p:>11}" for p in labels))
    for a in labels:
        print(f"  {a:>11}" + "".join(f"{cm[a][p]:>11}" for p in labels))
    for a in labels:
        tp = cm[a][a]
        fp = sum(cm[o][a] for o in labels if o != a)
        fn = sum(cm[a][p] for p in labels if p != a)
        pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        print(f"  {a:>10}: precision {pr:.3f} recall {rc:.3f} F1 {2 * pr * rc / max(pr + rc, 1e-9):.3f}")
    by_kind = collections.defaultdict(lambda: [0, 0])
    for c in cases:
        by_kind[c["kind"]][0] += c["pred"] == c["label"]
        by_kind[c["kind"]][1] += 1
    print("  accuracy by edit type: " + ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in by_kind.items()))
    # threshold sweep: a case is called misquoted when its best word-level similarity >= th (exact matches stay verified)
    print("  sweep of the misquote threshold (macro F1 over misquoted/fabricated, exact matches unchanged):")
    best = None
    for th in [x / 100 for x in range(40, 95, 5)]:
        tp = fp = fn = 0
        tpf = fpf = fnf = 0
        for c in cases:
            if c["pred"] == "verified" or c["score"] is None:
                continue
            pred = "misquoted" if c["score"] >= th else "fabricated"
            tp += pred == "misquoted" and c["label"] == "misquoted"
            fp += pred == "misquoted" and c["label"] != "misquoted"
            fn += pred != "misquoted" and c["label"] == "misquoted"
            tpf += pred == "fabricated" and c["label"] == "fabricated"
            fpf += pred == "fabricated" and c["label"] != "fabricated"
            fnf += pred != "fabricated" and c["label"] == "fabricated"
        f1 = lambda a, b, c_: 2 * a / max(2 * a + b + c_, 1)  # noqa: E731
        m = (f1(tp, fp, fn) + f1(tpf, fpf, fnf)) / 2
        best = max(best or (0, th), (m, th))
        print(f"    th={th:.2f}  misquoted F1 {f1(tp, fp, fn):.3f}  fabricated F1 {f1(tpf, fpf, fnf):.3f}  macro {m:.3f}")
    print(f"  best macro-F1 at th={best[1]:.2f} (current {V.VERSE_MISQUOTE_MIN})")
    return [c for c in cases if c["pred"] != c["label"]]


def run_hadith(n: int, rnd: random.Random):
    from core import retrieve as R
    con = connect()
    rows = [dict(r) for r in con.execute("SELECT id, text_ar FROM passages WHERE source='hadeethenc'")]
    rows = [r for r in rows if len(normalize_ar(r["text_ar"]).split()) >= 20]
    vocab = [w for r in rows[:400] for w in normalize_ar(r["text_ar"]).split()]
    cases = []
    for r in rnd.sample(rows, n):
        w = normalize_ar(r["text_ar"]).split()
        a = rnd.randint(0, len(w) - 12)
        frag = w[a:a + rnd.randint(8, 12)]
        cases.append({"kind": "fragment", "label": "match", "text": " ".join(frag), "id": r["id"]})
        e = frag[:]
        for k in rnd.sample(range(len(e)), 1):
            e[k] = rnd.choice(vocab)
        cases.append({"kind": "sub1", "label": "match", "text": " ".join(e), "id": r["id"]})
        e2 = frag[:]
        del e2[rnd.randrange(len(e2))]
        cases.append({"kind": "del1", "label": "match", "text": " ".join(e2), "id": r["id"]})
    for t in foreign_arabic(n, rnd, 8, 12):
        cases.append({"kind": "foreign_text", "label": "no_match", "text": t, "id": None})
    for c in cases:
        cn = normalize_ar(c["text"])
        best = 0
        for pid in R.keyword_search(cn, "hadith", k=15):
            p = R.get_passage(pid)
            if p:
                best = max(best, fuzz.partial_ratio(cn, normalize_ar(p["text"])))
        c["score"] = best
    print(f"\n=== HADITH (local HadeethEnc retrieval + fuzzy match): {len(cases)} cases (current HADITH_MATCH_MIN={V.HADITH_MATCH_MIN})")
    best = None
    for th in range(60, 100, 5):
        tp = sum(c["label"] == "match" and c["score"] >= th for c in cases)
        fn = sum(c["label"] == "match" and c["score"] < th for c in cases)
        fp = sum(c["label"] == "no_match" and c["score"] >= th for c in cases)
        pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        f1 = 2 * pr * rc / max(pr + rc, 1e-9)
        best = max(best or (0, th), (f1, th))
        print(f"    th={th}  precision {pr:.3f} recall {rc:.3f} F1 {f1:.3f}")
    print(f"  best F1 {best[0]:.3f} at th={best[1]} (current {V.HADITH_MATCH_MIN})")
    th = V.HADITH_MATCH_MIN
    return [c for c in cases if (c["score"] >= th) != (c["label"] == "match")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verses", type=int, default=250)
    ap.add_argument("--hadith", type=int, default=250)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--only", choices=["verses", "hadith"])
    a = ap.parse_args()
    fails = []
    if a.only != "hadith":
        fails += [dict(f, section="verse") for f in run_verses(a.verses, random.Random(a.seed))]
    if a.only != "verses":
        fails += [dict(f, section="hadith") for f in run_hadith(a.hadith, random.Random(a.seed + 1))]
    OUT.write_text("\n".join(json.dumps(f, ensure_ascii=False) for f in fails), encoding="utf-8")
    print(f"\n{len(fails)} misclassified examples saved to {OUT.name} for review")


if __name__ == "__main__":
    main()
