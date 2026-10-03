"""Is a trained router needed? Two offline checks (no LLM, no cost).

1. Generalisation of the heuristic router on ~540 real icadb questions it was NOT tuned on (weak labels: audience).
2. Leave-one-out accuracy of an embedding classifier (bge-m3 + kNN) on the labelled golden set, to see what a learned router
   could do with the data we actually have.
"""
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.db import connect  # noqa: E402
from core.router import heuristic_route  # noqa: E402


def icadb_questions():
    out = []
    for r in connect().execute("SELECT title, meta FROM passages WHERE source='icadb'"):
        out.append((r["title"], json.loads(r["meta"] or "{}").get("audience")))
    return out


def check_heuristic():
    qs = icadb_questions()
    by = collections.defaultdict(collections.Counter)
    intents = collections.Counter()
    false_d = []
    for q, aud in qs:
        r = heuristic_route(q)
        by[aud][r["level"]] += 1
        intents[r["intent"]] += 1
        if r["level"] == "د":
            false_d.append(q)
    print(f"{len(qs)} real icadb questions through the heuristic router")
    for aud, c in by.items():
        tot = sum(c.values())
        print(f"  audience={aud:12} n={tot:4}  " + "  ".join(f"{lv}:{100 * c[lv] / tot:4.0f}%" for lv in "أبجد"))
    print(f"  intents: {dict(intents)}")
    print(f"  level د (referral) on general questions: {len(false_d)}/{len(qs)} = {100 * len(false_d) / len(qs):.1f}% (all of these are general-knowledge questions)")
    for q in false_d[:8]:
        print("    د ->", q[:90])


def check_knn():
    from core.embedding import embed_texts
    gold = [json.loads(l) for l in (ROOT / "evals/golden.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    gold = [g for g in gold if g["expected_intent"] != "verify"]
    X = embed_texts([g["input"] for g in gold])
    y = [g["expected_level"] for g in gold]
    correct, per = 0, collections.defaultdict(lambda: [0, 0])
    for i in range(len(gold)):
        sims = sorted(((sum(a * b for a, b in zip(X[i], X[j])), j) for j in range(len(gold)) if j != i), reverse=True)[:3]
        votes = collections.Counter()
        for s, j in sims:
            votes[y[j]] += s
        pred = votes.most_common(1)[0][0]
        correct += pred == y[i]
        per[y[i]][0] += pred == y[i]
        per[y[i]][1] += 1
    print(f"\nembedding kNN (bge-m3, k=3), leave-one-out on {len(gold)} labelled golden questions: {100 * correct / len(gold):.0f}% "
          f"(per level: " + ", ".join(f"{k} {v[0]}/{v[1]}" for k, v in sorted(per.items())) + ")")
    print("  heuristic router on the same questions (tuned on them): " +
          f"{100 * sum(heuristic_route(g['input'])['level'] == g['expected_level'] for g in gold) / len(gold):.0f}%")


if __name__ == "__main__":
    check_heuristic()
    check_knn()
