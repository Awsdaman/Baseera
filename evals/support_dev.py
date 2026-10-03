"""Calibrate the semantic support check on saved eval answers (free: no LLM calls).

    python evals/support_dev.py [results.json] [--scorer bge-m3|bge-reranker|both] [--seed 7]

Positives : (cited stretch of an ACCEPTED answer, the passages it cites).
Hard negs : the same stretch against passages retrieved for the same question but NOT cited/shown anywhere in that answer.
Random negs: the same stretch against passages retrieved for other questions.
Caveat    : an uncited retrieved passage can genuinely support a stretch (label noise); these are NOT true hallucinations, so the
            numbers measure how well a scorer tells 'the cited passage' from 'a topically close but different passage'.
Output    : AUC per negative type, the threshold keeping >=95% of positives, and the weakest positives for manual review.
"""
import argparse
import glob
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import retrieve as R  # noqa: E402
from core import support as S  # noqa: E402
from core.verify import PLACEHOLDER, cite_ids  # noqa: E402

OUT = Path(__file__).with_name("support_dev.jsonl")


def auc(pos: list[float], neg: list[float]) -> float:
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def latest_results() -> str:
    return sorted(x for x in glob.glob(str(ROOT / "evals/reports/results-*.json")) if "rejudged" not in x)[-1]


def build_pairs(results_path: str, seed: int):
    rnd = random.Random(seed)
    data = json.loads(Path(results_path).read_text(encoding="utf-8"))["results"]
    pools = {r["id"]: [i for i in r["trace"]["retrieved_ids"]] for r in data if r.get("trace")}
    pairs = []
    for r in data:
        if r["status"] != "answered" or not r.get("trace") or not r["trace"]["attempts"]:
            continue
        last = r["trace"]["attempts"][-1]
        if not last["ok"] or not last["raw"]:
            continue
        shown = set(cite_ids(last["raw"])) | {f"{m.group(1)}:{m.group(2)}" for m in PLACEHOLDER.finditer(last["raw"])}
        shown_ids = set(cite_ids(last["raw"]))
        for m in PLACEHOLDER.finditer(last["raw"]):
            if m.group(1) == "quran":
                shown_ids.add("quran:" + m.group(2).split("-")[0].strip())
            elif m.group(1) == "hadith":
                shown_ids.add("hadith:" + m.group(2).strip())
        own = [i for i in pools[r["id"]] if i not in shown_ids]
        other = [i for k, v in pools.items() if k != r["id"] for i in v if i not in pools[r["id"]]]
        for st in S.stretches(last["raw"]):
            k = max(1, len(st["cites"]))
            if len(own) < k or len(other) < k:
                continue
            pairs.append({"case": r["id"], "text": st["text"], "cites": st["cites"],
                          "hard": rnd.sample(own, k), "rand": rnd.sample(other, k)})
    return pairs


def best_score(scorer, claim: str, ids: list[str]) -> float:
    items = []
    for i in ids:
        p = R.get_passage(i)
        if p:
            items += [(claim, c) for c in S.passage_chunks(p)]
    return max(scorer.scores(items)) if items else 0.0


def evaluate(name: str, pairs: list[dict]):
    scorer = S.get_scorer(name)
    pos, hard, rnd_ = [], [], []
    for p in pairs:
        p.setdefault("scores", {})[name] = {"pos": (a := best_score(scorer, p["text"], p["cites"])),
                                            "hard": (b := best_score(scorer, p["text"], p["hard"])),
                                            "rand": (c := best_score(scorer, p["text"], p["rand"]))}
        pos.append(a), hard.append(b), rnd_.append(c)
    pos_sorted = sorted(pos)
    th95 = pos_sorted[int(0.05 * len(pos_sorted))]  # 95% of positives are >= this
    th90 = pos_sorted[int(0.10 * len(pos_sorted))]
    print(f"\n=== scorer {name}: {len(pairs)} cited stretches")
    print(f"  AUC cited-vs-hard-negative: {auc(pos, hard):.3f}   cited-vs-random: {auc(pos, rnd_):.3f}")
    print(f"  mean score  cited {sum(pos)/len(pos):.3f} | hard-neg {sum(hard)/len(hard):.3f} | random {sum(rnd_)/len(rnd_):.3f}")
    th99 = pos_sorted[int(0.01 * len(pos_sorted))]
    th98 = pos_sorted[int(0.02 * len(pos_sorted))]
    for label, th in (("keeps 99% of cited", th99), ("keeps 98% of cited", th98), ("keeps 95% of cited", th95), ("keeps 90% of cited", th90)):
        print(f"  theta={th:.3f} ({label}): catches {100*sum(x < th for x in hard)/len(hard):.0f}% of hard negatives, "
              f"{100*sum(x < th for x in rnd_)/len(rnd_):.0f}% of random")
    weakest = sorted(pairs, key=lambda p: p["scores"][name]["pos"])[:6]
    print("  weakest cited stretches (review: unsupported claim, or just a loose paraphrase?):")
    for p in weakest:
        print(f"    {p['scores'][name]['pos']:.3f} {p['case']} cites={p['cites']} | {p['text'][:110]}")
    return th95, th90


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?")
    ap.add_argument("--scorer", default="bge-m3")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    path = a.results or latest_results()
    pairs = build_pairs(path, a.seed)
    print(f"{path}\n{len(pairs)} cited stretches from accepted answers")
    for name in (["bge-m3", "bge-reranker"] if a.scorer == "both" else [a.scorer]):
        evaluate(name, pairs)
    OUT.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in pairs), encoding="utf-8")
    print("saved", OUT)


if __name__ == "__main__":
    main()
