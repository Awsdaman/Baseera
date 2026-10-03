"""Review opt-in problem reports (human in the loop: nothing here changes Baseera by itself).

    python -m tools.review summary                       # counts by reason / outcome + the source-gap list
    python -m tools.review list [--all]                  # pending reports (or all)
    python -m tools.review show ID
    python -m tools.review approve-test ID --level ب [--behavior answer] [--intent ask] [--note "..."]
                                                         # -> new permanent case in evals/golden.jsonl (id r-<ID>)
    python -m tools.review approve-rewrite ID --canonical "ما حكم ...؟" [--claim "..."]
                                                         # -> data/curated/rewrites.json (used before search from now on)
    python -m tools.review gap ID [--topic "..."]        # -> data/curated/source_gaps.json (topics the sources do not cover)
    python -m tools.review dismiss ID [--note "..."]
    python -m tools.review purge [--days N]              # delete reports older than N days (also done at server start)
"""
import argparse
import collections
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import reports, rewrites  # noqa: E402

GOLDEN = ROOT / "evals" / "golden.jsonl"
GAPS = ROOT / "data" / "curated" / "source_gaps.json"
LEVELS = ("أ", "ب", "ج", "د")
BEHAVIORS = ("answer", "answer_or_abstain", "abstain", "abstain_or_clarify", "referral", "term", "verify", "correct_verse")


def _need(rid: int) -> dict:
    r = reports.get_report(rid)
    if not r:
        sys.exit(f"no report {rid}")
    return r


def _fmt(r: dict, full: bool = False) -> str:
    head = f"#{r['id']} [{r['review']}] {r['created']} · {r['reason']} · {r['status']}/{r['level']}/{r['intent']}" + (
        f" · {r['abstain_reason']}" if r.get("abstain_reason") else "")
    if not full:
        return head + "\n    " + r["question"][:110].replace("\n", " ")
    return "\n".join([head, "QUESTION: " + r["question"], "COMMENT:  " + (r["comment"] or "-"), "ANSWER:   " + (r["answer_excerpt"] or "-"),
                      "SOURCES:  " + (r["source_ids"] or "-"), "NOTE:     " + (r["review_note"] or "-")])


def approve_test(rid: int, level: str, behavior: str = "answer", intent: str = "ask", note: str = "", retrieve: list[str] | None = None) -> dict:
    """Turn a report into a permanent regression case. The reviewer supplies the expected level/behaviour (the report only has the question)."""
    r = _need(rid)
    if level not in LEVELS or behavior not in BEHAVIORS:
        sys.exit(f"level must be one of {LEVELS}; behavior one of {BEHAVIORS}")
    case = {"id": f"r-{rid}", "input": r["question"], "expected_level": level, "expected_intent": intent, "expected_behavior": behavior,
            "notes": (note or f"user report #{rid}: {r['reason']} {r['comment'] or ''}").strip(), "source": "user report"}
    if retrieve:
        case["must_retrieve_any"] = retrieve
    rows = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(x["id"] == case["id"] for x in rows):
        sys.exit(f"{case['id']} is already in the golden set")
    with open(GOLDEN, "a", encoding="utf-8", newline=chr(10)) as f:
        f.write(json.dumps(case, ensure_ascii=False) + chr(10))
    reports.set_review(rid, "approved_test", case["id"])
    return case


def approve_rewrite(rid: int, canonical: str, claim: str | None = None) -> dict:
    r = _need(rid)
    entry = rewrites.add(r["question"], canonical, claim, added=time.strftime("%Y-%m-%d"), source=f"report #{rid}")
    reports.set_review(rid, "approved_rewrite", canonical)
    return entry


def gap(rid: int, topic: str | None = None) -> dict:
    r = _need(rid)
    items = json.loads(GAPS.read_text(encoding="utf-8")) if GAPS.exists() else []
    entry = {"topic": topic or r["question"], "example_question": r["question"], "report": rid, "reason": r["reason"],
             "added": time.strftime("%Y-%m-%d"), "what_to_add": "a source that states the answer (ruling / explanation) for this topic"}
    items.append(entry)
    GAPS.parent.mkdir(parents=True, exist_ok=True)
    GAPS.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    reports.set_review(rid, "source_gap", entry["topic"])
    return entry


def summary() -> str:
    allr = reports.list_reports(None)
    out = [f"{len(allr)} reports ({sum(1 for r in allr if r['review'] == 'pending')} pending)"]
    for title, key in (("by reason", "reason"), ("by outcome", "status"), ("by review", "review")):
        c = collections.Counter(r[key] for r in allr)
        out.append(f"  {title}: " + ", ".join(f"{k}={v}" for k, v in c.most_common()))
    gaps = json.loads(GAPS.read_text(encoding="utf-8")) if GAPS.exists() else []
    out.append(f"  source gaps recorded: {len(gaps)}" + "".join(f"\n    - {g['topic'][:90]}" for g in gaps[-10:]))
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("summary")
    p = sub.add_parser("list"); p.add_argument("--all", action="store_true")
    p = sub.add_parser("show"); p.add_argument("id", type=int)
    p = sub.add_parser("approve-test"); p.add_argument("id", type=int); p.add_argument("--level", required=True); p.add_argument("--behavior", default="answer")
    p.add_argument("--intent", default="ask"); p.add_argument("--note", default=""); p.add_argument("--retrieve", nargs="*")
    p = sub.add_parser("approve-rewrite"); p.add_argument("id", type=int); p.add_argument("--canonical", required=True); p.add_argument("--claim")
    p = sub.add_parser("gap"); p.add_argument("id", type=int); p.add_argument("--topic")
    p = sub.add_parser("dismiss"); p.add_argument("id", type=int); p.add_argument("--note", default="")
    p = sub.add_parser("purge"); p.add_argument("--days", type=int)
    a = ap.parse_args(argv)
    if a.cmd == "summary":
        print(summary())
    elif a.cmd == "list":
        rs = reports.list_reports(None if a.all else "pending")
        print("\n".join(_fmt(r) for r in rs) or "no reports")
    elif a.cmd == "show":
        print(_fmt(_need(a.id), full=True))
    elif a.cmd == "approve-test":
        print("added to golden set:", json.dumps(approve_test(a.id, a.level, a.behavior, a.intent, a.note, a.retrieve), ensure_ascii=False))
    elif a.cmd == "approve-rewrite":
        print("approved rewrite:", json.dumps(approve_rewrite(a.id, a.canonical, a.claim), ensure_ascii=False))
    elif a.cmd == "gap":
        print("recorded source gap:", json.dumps(gap(a.id, a.topic), ensure_ascii=False))
    elif a.cmd == "dismiss":
        _need(a.id)
        reports.set_review(a.id, "dismissed", a.note)
        print("dismissed", a.id)
    elif a.cmd == "purge":
        print("deleted", reports.purge(a.days), "reports")


if __name__ == "__main__":
    main()
