"""Run a file of questions (one per line) through Baseera and save EVERY answer in full, readable in a browser.

    python evals/bulk_check.py "data/questions/my questions.txt" [--limit N] [--out evals/reports/bulk-name]

No labels needed: it records what Baseera did with each question (status, level, abstain reason, retry count, sources) and the full answer
text, then writes <out>.json and <out>.html (RTL). Lines ending with ':' are treated as headings and skipped. The outputs are git-ignored
(`evals/reports/bulk-*`): question collections may come from real people.
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


def load_questions(path: str) -> list[str]:
    qs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.endswith(":") and not line.endswith("：") and len(line) > 3:
            qs.append(line)
    return qs


def run(questions: list[str]) -> list[dict]:
    rows = []
    for i, q in enumerate(questions, 1):
        t0 = time.time()
        L.thread_usage_reset()
        try:
            r = pipeline.ask(q, debug=True)
        except Exception as e:  # keep going: one bad case must not lose the rest
            r = {"status": "error", "answer_text": str(e)[:300], "blocks": [], "sources": [], "route": {}, "debug": {"attempts": []}}
        dbg = r.get("debug") or {}
        rows.append({"n": i, "question": q, "status": r["status"], "level": r.get("level"), "intent": r.get("intent"),
                     "route_source": (r.get("route") or {}).get("source"), "abstain_reason": r.get("abstain_reason"),
                     "attempts": len(dbg.get("attempts", [])), "secs": round(time.time() - t0, 1),
                     "answer_text": r.get("answer_text", ""), "blocks": r.get("blocks", []),
                     "sources": [{"n": s["n"], "id": s["id"], "type": s["type"], "title": s.get("title"), "grade": s.get("grade")} for s in r.get("sources", [])],
                     "verification_errors": r.get("verification_errors", []), "empathy": bool(r.get("empathy"))})
        print(f"[{i:2}/{len(questions)}] {r['status']:10} {r.get('level')}  {rows[-1]['secs']:5}s  {q[:70]}", flush=True)
    return rows


COLORS = {"answered": "#1F7872", "abstained": "#B9770E", "referral": "#5B6B8C", "verified": "#1F7872", "error": "#B03A2E"}


def page(rows: list[dict], meta: str) -> str:
    esc = html.escape
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    out = [f"""<!doctype html><html lang="ar" dir="rtl"><head><meta charset="utf-8"><title>Baseera: bulk check</title><style>
body{{font-family:'Segoe UI',Tahoma,sans-serif;background:#f6f4ee;color:#1B2D45;max-width:900px;margin:0 auto;padding:16px;line-height:1.8}}
.q{{background:#fff;border-radius:10px;padding:12px 16px;margin:14px 0;box-shadow:0 1px 3px #0002;border-right:6px solid #999}}
.q h3{{margin:0 0 6px;font-size:17px}} .meta{{font-size:12px;color:#555;direction:ltr;text-align:left}}
.ans{{white-space:pre-wrap;margin-top:8px}} .src{{font-size:12px;color:#444;direction:ltr;text-align:left;margin-top:6px}}
.err{{font-size:12px;color:#B03A2E;direction:ltr;text-align:left}} b.tag{{padding:1px 8px;border-radius:9px;color:#fff;font-size:12px}}
.blk{{background:#eef5f4;border-radius:6px;padding:6px 10px;margin:6px 0;font-size:15px}} .hd{{font-size:12px;color:#1F7872;font-weight:bold}}
</style></head><body><h1>Baseera: bulk check ({len(rows)} questions)</h1><p>{esc(meta)}<br>""" + " · ".join(f"{k}: {v}" for k, v in counts.items()) + "</p>"]
    for r in rows:
        c = COLORS.get(r["status"], "#999")
        out.append(f'<div class="q" style="border-right-color:{c}"><h3>{r["n"]}. {esc(r["question"])}</h3>'
                   f'<div class="meta"><b class="tag" style="background:{c}">{esc(r["status"])}</b> level {esc(str(r["level"]))} · {esc(str(r["intent"]))} · '
                   f'router={esc(str(r["route_source"]))} · attempts={r["attempts"]} · {r["secs"]}s'
                   + (f' · reason={esc(str(r["abstain_reason"]))}' if r["abstain_reason"] else "") + "</div>")
        for b in r["blocks"]:
            k = b["kind"]
            label = {"quran": "Quran", "hadith": "Hadith", "tafsir": "Tafsir", "explanation": "Explanation", "notice": "Notice"}.get(k, k)
            body = b.get("text_ar") or b.get("text") or b.get("message") or ""
            extra = f' ({esc(str(b.get("ref") or b.get("grade") or ""))})' if k in ("quran", "hadith", "tafsir") else ""
            out.append(f'<div class="blk"><span class="hd">{label}{extra}</span><div class="ans">{esc(body)}</div></div>')
        if not r["blocks"]:
            out.append(f'<div class="ans">{esc(r["answer_text"])}</div>')
        if r["sources"]:
            out.append('<div class="src">sources: ' + " | ".join(f'[{s["n"]}] {esc(s["id"])}' for s in r["sources"]) + "</div>")
        if r["verification_errors"]:
            out.append('<div class="err">verifier: ' + esc(" ; ".join(e[:120] for e in r["verification_errors"][:3])) + "</div>")
        out.append("</div>")
    out.append("</body></html>")
    return "".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--out", default=str(ROOT / "evals" / "reports" / f"bulk-{time.strftime('%Y%m%d-%H%M%S')}"))
    a = ap.parse_args()
    qs = load_questions(a.file)[: a.limit]
    print(f"{len(qs)} questions · {L.describe()}", flush=True)
    rows = run(qs)
    meta = f"{time.strftime('%Y-%m-%d %H:%M')} · {json.dumps(L.describe(), ensure_ascii=False)}"
    Path(a.out + ".json").write_text(json.dumps({"meta": meta, "results": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    Path(a.out + ".html").write_text(page(rows, meta), encoding="utf-8")
    print(f"saved {a.out}.html / .json")


if __name__ == "__main__":
    main()
