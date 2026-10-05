"""Reliability and answer-boundary suite (evals/reliability.jsonl), with the FULL answer to every question kept for reading.

    python evals/reliability.py [--only should_answer,no_source,disputed,paraphrase] [--limit N]

Categories (what a reviewer asked for):
  should_answer  direct questions the approved sources cover        -> must be answered (not refused)
  no_source      no sufficient source / out of scope                -> must abstain, refer, ask for clarification, or answer WITH a limits
                                                                       note (no_ruling / not_direct); a confident answer is a failure
  disputed       scholars differ                                    -> must show a disputed / refer note (or abstain); never one bare ruling
  paraphrase     the same meaning in different wording (groups)     -> outcome class and level must not flip; cited sources should overlap

Writes evals/reports/reliability-<ts>.json and .html (git-ignored: `evals/reports/reliability-*`).
"""
import argparse
import html
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import llm as L  # noqa: E402
from core import pipeline  # noqa: E402

LIMIT_NOTES = ("no_ruling", "not_direct")
DISPUTE_NOTES = ("disputed", "refer")


def notes(resp: dict) -> set[str]:
    return {b.get("note") for b in resp.get("blocks", []) if b.get("kind") == "notice" and b.get("note")}


def klass(resp: dict) -> str:
    """answer-like vs refusal-like, for the consistency check."""
    return "answered" if resp["status"] == "answered" else "withheld"


def judge_case(case: dict, resp: dict) -> tuple[bool, str]:
    st, ns, cat = resp["status"], notes(resp), case["category"]
    if cat == "should_answer":
        return st == "answered", "answered" if st == "answered" else f"refused ({st}/{resp.get('abstain_reason')})"
    if cat == "no_source":
        if st in ("abstained", "referral"):
            return True, f"declined ({resp.get('abstain_reason') or st})"
        if st == "answered" and ns & set(LIMIT_NOTES):
            return True, "answered with a limits note"
        return False, "CONFIDENT answer without a limits note"
    if cat == "disputed":
        if st in ("abstained", "referral"):
            return True, "declined"
        if st == "answered" and ns & (set(DISPUTE_NOTES) | set(LIMIT_NOTES)):
            return True, "positions shown with a disputed/refer/limits note"
        return False, "answered without a disputed/refer note"
    return True, "see consistency"  # paraphrase cases are judged per group


def source_ids(resp: dict) -> set[str]:
    return {s["id"] for s in resp.get("sources", [])}


def group_consistency(rows: list[dict]) -> list[dict]:
    out = []
    groups: dict[str, list[dict]] = {}
    for r in rows:
        if r["category"] == "paraphrase":
            groups.setdefault(r["group"], []).append(r)
    for g, members in groups.items():
        classes = {m["class"] for m in members}
        levels = {m["level"] for m in members}
        sets = [set(m["source_ids"]) for m in members]
        pair = [len(a & b) / max(1, len(a | b)) for i, a in enumerate(sets) for b in sets[i + 1:]]
        out.append({"group": g, "same_outcome_class": len(classes) == 1, "same_level": len(levels) == 1,
                    "source_overlap": round(sum(pair) / len(pair), 2) if pair else None,
                    "classes": sorted(classes), "levels": sorted(map(str, levels)), "ok": len(classes) == 1})
    return out


def run(cases: list[dict]) -> list[dict]:
    rows = []
    for i, c in enumerate(cases, 1):
        t0 = time.time()
        try:
            r = pipeline.ask(c["input"], debug=True)
        except Exception as e:
            r = {"status": "error", "answer_text": str(e)[:300], "blocks": [], "sources": [], "route": {}}
        ok, why = judge_case(c, r)
        row = c | {"status": r["status"], "level": r.get("level"), "abstain_reason": r.get("abstain_reason"), "ok": ok, "why": why,
                   "class": klass(r), "notes": sorted(notes(r)), "source_ids": sorted(source_ids(r)), "answer_text": r.get("answer_text", ""),
                   "blocks": r.get("blocks", []), "secs": round(time.time() - t0, 1),
                   "relevance": ((r.get("debug") or {}).get("attempts") or [{}])[-1].get("relevance")}
        rows.append(row)
        flag = "OK " if ok or c["category"] == "paraphrase" else "BAD"
        print(f"[{flag}] {i:2}/{len(cases)} {c['category']:13} {r['status']:10} {str(r.get('level')):2} {why[:44]:44} {c['input'][:48]}", flush=True)
    return rows


def summarize(rows: list[dict]) -> dict:
    s = {}
    for cat in ("should_answer", "no_source", "disputed"):
        rs = [r for r in rows if r["category"] == cat]
        if rs:
            s[cat] = {"pass": sum(r["ok"] for r in rs), "n": len(rs), "rate": round(100 * sum(r["ok"] for r in rs) / len(rs), 1)}
    gs = group_consistency(rows)
    if gs:
        s["paraphrase_groups"] = {"consistent": sum(g["ok"] for g in gs), "n": len(gs), "same_level": sum(g["same_level"] for g in gs),
                                  "details": gs}
    ans = [r for r in rows if r["status"] == "answered"]
    s["answered"] = len(ans)
    s["answered_with_limits_note"] = sum(1 for r in ans if set(r["notes"]) & set(LIMIT_NOTES))
    s["abstained_or_referred"] = sum(1 for r in rows if r["status"] in ("abstained", "referral"))
    s["errors"] = sum(1 for r in rows if r["status"] == "error")
    return s


def page(rows: list[dict], summary: dict, meta: str) -> str:
    esc = html.escape
    out = [f"""<!doctype html><html lang="ar" dir="rtl"><head><meta charset="utf-8"><title>Baseera reliability</title><style>
body{{font-family:'Segoe UI',Tahoma,sans-serif;background:#f6f4ee;color:#1B2D45;max-width:920px;margin:0 auto;padding:16px;line-height:1.8}}
.q{{background:#fff;border-radius:10px;padding:10px 14px;margin:10px 0;box-shadow:0 1px 3px #0002;border-right:6px solid #999}}
.ok{{border-right-color:#1F7872}}.bad{{border-right-color:#B03A2E}} .meta{{font-size:12px;color:#555;direction:ltr;text-align:left}}
.ans{{white-space:pre-wrap;font-size:15px;margin-top:6px}} pre{{background:#fff;padding:10px;border-radius:8px;direction:ltr;text-align:left;font-size:12px}}
</style></head><body><h1>Baseera: reliability and answer boundaries</h1><p>{esc(meta)}</p>
<pre>{esc(json.dumps({k: v for k, v in summary.items() if k != 'paraphrase_groups'}, ensure_ascii=False, indent=1))}
paraphrase groups: {json.dumps([{k: g[k] for k in ('group', 'same_outcome_class', 'same_level', 'source_overlap')} for g in summary.get('paraphrase_groups', {}).get('details', [])], ensure_ascii=False, indent=1)}</pre>"""]
    for r in rows:
        cls = "ok" if r["ok"] else "bad"
        body = "\n\n".join((b.get("text_ar") or b.get("text") or "") for b in r["blocks"]) or r["answer_text"]
        out.append(f'<div class="q {cls}"><b>{esc(r["input"])}</b><div class="meta">{esc(r["category"])} · {esc(r["status"])} · level {esc(str(r["level"]))} · '
                   f'{esc(r["why"])} · notes={esc(",".join(r["notes"]))} · relevance={esc(str((r.get("relevance") or {}).get("verdict")))} · {r["secs"]}s</div>'
                   f'<div class="ans">{esc(body)}</div></div>')
    return "".join(out) + "</body></html>"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="comma-separated categories")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--file", default=str(ROOT / "evals" / "reliability.jsonl"))
    a = ap.parse_args()
    cases = [json.loads(x) for x in Path(a.file).read_text(encoding="utf-8").splitlines() if x.strip()]
    if a.only:
        keep = set(a.only.split(","))
        cases = [c for c in cases if c["category"] in keep]
    cases = cases[: a.limit]
    print(f"{len(cases)} questions · {L.describe()}", flush=True)
    rows = run(cases)
    summary = summarize(rows)
    meta = f"{time.strftime('%Y-%m-%d %H:%M')} · {json.dumps(L.describe(), ensure_ascii=False)}"
    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = ROOT / "evals" / "reports" / f"reliability-{stamp}"
    out.with_suffix(".json").write_text(json.dumps({"meta": meta, "summary": summary, "results": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    out.with_suffix(".html").write_text(page(rows, summary, meta), encoding="utf-8")
    print("\n== RELIABILITY SUMMARY ==")
    for k, v in summary.items():
        if k != "paraphrase_groups":
            print(f"  {k:26} {v}")
    pg = summary.get("paraphrase_groups")
    if pg:
        print(f"  paraphrase groups consistent {pg['consistent']}/{pg['n']} (same level {pg['same_level']}/{pg['n']})")
        for g in pg["details"]:
            print(f"    {g['group']:16} outcome={g['classes']} levels={g['levels']} source_overlap={g['source_overlap']}")
    print(f"\nreport: {out}.html")


if __name__ == "__main__":
    main()
