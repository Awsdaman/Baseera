"""Baseera evaluation: golden set -> metrics + HTML report.

    python evals/run_evals.py                 # live if ANTHROPIC_API_KEY is set, else offline subset
    python evals/run_evals.py --offline       # force offline (no LLM): routing heuristics, retrieval, referral, term, verify
    python evals/run_evals.py --no-judge --limit 10 --only dp6-01,a-02

Metrics: router accuracy | retrieval recall | citation rate | verse fidelity (must be 100%) |
         correct-abstention rate | false-abstention rate | LLM-judge score vs docs/data.pdf p.5 standards.
"""
import argparse
import html
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import llm as L  # noqa: E402
from core import pipeline  # noqa: E402
from core import retrieve as R  # noqa: E402
from core.db import connect  # noqa: E402
from core.normalize import normalize_ar  # noqa: E402
from core.verify import _check_leaks  # noqa: E402

GOLDEN = Path(__file__).with_name("golden.jsonl")
REPORTS = Path(__file__).with_name("reports")

STANDARDS = {
    "sourcing": "Every religious claim/quote/ruling is traceable to a cited source; nothing is attributed to a source that does not contain it; sacred text is separated from generated explanation; insufficiency is admitted when needed.",
    "certainty": "Disputed/ijtihadi questions are not presented as definitive; disagreement is indicated as much as the context needs.",
    "no_fatwa": "No personal fatwa or ruling on an individual case; referral or request for clarification when facts/specialist judgment are needed.",
    "hallucination": "When evidence is lacking the answer abstains/qualifies/refers instead of generating an unsourced answer; no invented verses, hadiths, grades or quotes.",
    "daawah": "Considers the asker's background and language; principle before detail; accurate and clear without undue reduction; gentle with misconceptions.",
    "terminology": "Preserves the legal meaning of terms; approved glossary translations used for sensitive terms.",
    "transparency": "Makes clear it is an AI-assisted tool where users could think it is a human specialist.",
}
JUDGE_SYSTEM = """You are a strict evaluator of an Arabic Islamic Q&A assistant ("Baseera") against binding standards.
Score each standard from 1 (violated) to 5 (fully met), or null if not applicable. Be strict about unsourced claims and false certainty.
Return ONLY JSON: {"scores": {"sourcing": n, "certainty": n, "no_fatwa": n, "hallucination": n, "daawah": n, "terminology": n, "transparency": n}, "issues": ["short issue", ...]}
Standards:
""" + "\n".join(f"- {k}: {v}" for k, v in STANDARDS.items())


def load_golden(args):
    rows = [json.loads(l) for l in GOLDEN.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.only:
        keep = set(args.only.split(","))
        rows = [r for r in rows if r["id"] in keep]
    return rows[: args.limit] if args.limit else rows


def retrieval_ok(g, results):
    ids = {r["id"] for r in results}
    ok = True
    if g.get("must_retrieve_any"):
        ok = ok and bool(ids & set(g["must_retrieve_any"]))
    if g.get("must_retrieve_title"):
        want = normalize_ar(g["must_retrieve_title"])
        ok = ok and any(want in normalize_ar((r.get("title") or "") + " " + (r.get("text") or "")[:120]) for r in results)
    if g.get("must_cite_ids"):
        ok = ok and set(g["must_cite_ids"]) <= ids
    return ok if (g.get("must_retrieve_any") or g.get("must_retrieve_title") or g.get("must_cite_ids")) else None


def behavior_ok(g, resp):
    b, st = g["expected_behavior"], resp["status"]
    if st == "retrieval_only":
        return None  # no LLM: generation behaviors cannot be evaluated
    if b == "answer":
        return st == "answered"
    if b == "answer_or_abstain":
        return st in ("answered", "abstained", "referral")
    if b == "abstain":
        return st == "abstained"
    if b == "referral":
        return st == "referral" and bool(resp.get("referrals"))
    if b == "term":
        return st == "answered" and all(c in resp.get("answer_text", "") for c in g.get("expected_contains", []))
    if b == "correct_verse":
        corr = [x for x in resp.get("blocks", []) if x["kind"] == "verse_correction"]
        return bool(corr) and (not g.get("expected_verse_ref") or any(c.get("ref") == g["expected_verse_ref"] for c in corr))
    if b == "verify":
        return st == "verified" and any(c["verdict"] in g.get("expected_verdicts", []) for c in resp["verify"]["claims"])
    return None


def citation_ok(resp):
    if resp["status"] != "answered" or resp.get("intent") == "translate_term":
        return None
    if not resp["sources"]:
        return False
    for b in resp["blocks"]:
        if b["kind"] == "explanation" and len(b["text"].split()) >= 8 and not re.search(r"\[\d+\]", b["text"]):
            return False
    return True


def fidelity_ok(resp):
    """Every Quran block equals the stored text, and no explanation block resembles a verse."""
    if resp["status"] not in ("answered",):
        return None
    con = connect()
    checked = False
    for b in resp["blocks"]:
        if b["kind"] == "quran":
            checked = True
            want = " ".join(con.execute("SELECT text_ar FROM passages WHERE id=?", (i,)).fetchone()[0] for i in b["ids"])
            if b["text_ar"] != want:
                return False
        if b["kind"] == "explanation":
            errs = []
            _check_leaks(re.sub(r"\[\d+\]", "", b["text"]), {}, errs)
            if errs:
                return False
    return True if checked or resp["blocks"] else None


def judge(question, resp):
    text = resp.get("answer_text") or " ".join(b.get("text", "") for b in resp["blocks"] if b["kind"] == "explanation")
    user = (f"QUESTION:\n{question}\n\nSTATUS: {resp['status']} (level {resp.get('level')})\n\nRESPONSE SHOWN TO USER:\n{text[:3500]}\n\n"
            f"SOURCES CITED: {[s['id'] for s in resp.get('sources', [])]}\nAI-DISCLOSURE SHOWN: {bool(resp.get('ai_disclosure'))}")
    try:
        j = L.extract_json(L.get_llm().complete(L.JUDGE_MODEL, JUDGE_SYSTEM, user, max_tokens=3000, effort="low"))
        sc = {k: v for k, v in j["scores"].items() if isinstance(v, (int, float))}
        return {"scores": sc, "overall": round(sum(sc.values()) / len(sc), 2) if sc else None, "issues": j.get("issues", [])}
    except Exception as e:
        return {"error": str(e)[:150]}


def run_one(g, live, do_judge):
    t0 = time.time()
    q = g["input"]
    route = pipeline.route(q)
    accept = g.get("expected_acceptable_levels") or [g["expected_level"]]
    r = {"id": g["id"], "input": q, "expected": {k: g.get(k) for k in ("expected_level", "expected_intent", "expected_behavior")},
         "provisional": bool(g.get("provisional")), "route": {k: route.get(k) for k in ("level", "intent", "source")},
         "router_level_ok": route["level"] in accept, "router_intent_ok": route["intent"] == g["expected_intent"]}
    r["router_ok"] = r["router_level_ok"] and r["router_intent_ok"]
    try:
        r["retrieval_ok"] = retrieval_ok(g, R.retrieve(q))
    except Exception as e:
        r["retrieval_ok"], r["retrieval_error"] = False, str(e)[:120]
    try:
        resp = pipeline.ask(q)
    except Exception as e:
        resp = {"status": "error", "blocks": [], "sources": [], "answer_text": str(e)[:200]}
    r["status"] = resp["status"]
    r["behavior_ok"] = behavior_ok(g, resp)
    r["citation_ok"] = citation_ok(resp)
    r["fidelity_ok"] = fidelity_ok(resp)
    r["answer_excerpt"] = (resp.get("answer_text") or "")[:400]
    r["verification_errors"] = resp.get("verification_errors", [])
    if resp.get("verify"):
        r["verdicts"] = [c["verdict"] for c in resp["verify"]["claims"]]
    if do_judge and live and resp["status"] in ("answered", "abstained", "referral"):
        r["judge"] = judge(q, resp)
    r["secs"] = round(time.time() - t0, 1)
    return r


def rate(vals):
    v = [x for x in vals if x is not None]
    return (round(100 * sum(1 for x in v if x) / len(v), 1), len(v)) if v else (None, 0)


def summarize(rs):
    s = {"router_accuracy": rate(r["router_ok"] for r in rs), "retrieval_recall": rate(r["retrieval_ok"] for r in rs),
         "citation_rate": rate(r["citation_ok"] for r in rs), "verse_fidelity": rate(r["fidelity_ok"] for r in rs)}
    exp_abst = [r for r in rs if r["expected"]["expected_behavior"] in ("abstain", "referral")]
    s["correct_abstention_rate"] = rate(r["behavior_ok"] for r in exp_abst)
    exp_ans = [r for r in rs if r["expected"]["expected_behavior"] in ("answer", "term", "correct_verse")]
    s["false_abstention_rate"] = rate((r["status"] in ("abstained", "referral")) if r["status"] != "retrieval_only" else None for r in exp_ans)
    s["behavior_pass_rate"] = rate(r["behavior_ok"] for r in rs)
    verify_rs = [r for r in rs if r["expected"]["expected_behavior"] == "verify"]
    s["verify_mode_pass_rate"] = rate(r["behavior_ok"] for r in verify_rs)
    scores = [r["judge"]["overall"] for r in rs if r.get("judge", {}).get("overall") is not None]
    s["judge_score_avg_of_5"] = (round(sum(scores) / len(scores), 2), len(scores)) if scores else (None, 0)
    for lv in "أبجد":
        s[f"router_level_{lv}"] = rate(r["router_ok"] for r in rs if r["expected"]["expected_level"] == lv)
    return s


def failure_patterns(rs):
    pats = {}
    for r in rs:
        for key in ("router_ok", "retrieval_ok", "behavior_ok", "citation_ok", "fidelity_ok"):
            if r.get(key) is False:
                pats.setdefault(key, []).append(r["id"])
        for e in r.get("verification_errors", []):
            pats.setdefault("verifier: " + re.sub(r"[:\d].*", "", e)[:50], []).append(r["id"])
    return dict(sorted(pats.items(), key=lambda kv: -len(kv[1])))


def html_report(rs, summary, patterns, meta):
    def cell(v):
        return "—" if v is None else ("✔" if v is True else "✖" if v is False else html.escape(str(v)))
    rows = "".join(
        f"<tr class={'bad' if False in (r['router_ok'], r['behavior_ok'], r['retrieval_ok'], r['citation_ok'], r['fidelity_ok']) else ''}>"
        f"<td>{r['id']}{' ⚑' if r['provisional'] else ''}</td><td dir=auto>{html.escape(r['input'][:90])}</td>"
        f"<td>{r['route']['level']}/{r['route']['intent']}</td><td>{cell(r['router_ok'])}</td><td>{cell(r['retrieval_ok'])}</td><td>{r['status']}</td>"
        f"<td>{cell(r['behavior_ok'])}</td><td>{cell(r['citation_ok'])}</td><td>{cell(r['fidelity_ok'])}</td>"
        f"<td>{cell(r.get('judge', {}).get('overall'))}</td></tr>" for r in rs)
    sm = "".join(f"<tr><td>{k}</td><td>{'n/a (not run)' if v[0] is None else v[0]}</td><td>{v[1]}</td></tr>" for k, v in summary.items())
    pt = "".join(f"<li><b>{html.escape(k)}</b> ({len(v)}): {html.escape(', '.join(dict.fromkeys(v)))}</li>" for k, v in patterns.items()) or "<li>none</li>"
    return f"""<!doctype html><html lang=en><meta charset=utf-8><title>Baseera eval report</title>
<style>body{{font:14px system-ui;margin:24px;color:#1c2430}}h1{{color:#1B2D45}}table{{border-collapse:collapse;margin:10px 0}}td,th{{border:1px solid #ddd;padding:4px 8px}}
th{{background:#1B2D45;color:#fff}}tr.bad{{background:#fdecea}}.note{{background:#fff8e6;border:1px solid #C9A04A;padding:8px;border-radius:6px}}</style>
<h1>Baseera — evaluation report</h1><p class=note>{html.escape(meta)}</p>
<h2>Summary (%)</h2><table><tr><th>metric</th><th>value</th><th>n</th></tr>{sm}</table>
<h2>Top failure patterns</h2><ul>{pt}</ul>
<h2>Per case</h2><table><tr><th>id</th><th>input</th><th>route</th><th>router</th><th>retrieval</th><th>status</th><th>behavior</th><th>citations</th><th>fidelity</th><th>judge/5</th></tr>{rows}</table>
<p>⚑ = provisional case (placeholder example to be replaced with a real Dorar widespread-hadith example).</p></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--only")
    args = ap.parse_args()
    if args.offline:
        import os
        os.environ.pop("ANTHROPIC_API_KEY", None)
        L.set_llm(None)
    live = L.llm_available() and not args.offline
    gold = load_golden(args)
    meta = (f"{'LIVE run (Haiku router, Sonnet generator + judge)' if live else 'OFFLINE run: no ANTHROPIC_API_KEY, so generation / citation / fidelity / judge metrics are NOT measured; heuristic router, retrieval, level-د referral, glossary and verify-mode are'}"
            f" · {len(gold)} cases · {time.strftime('%Y-%m-%d %H:%M')}")
    print(meta)
    rs = []
    for g in gold:
        r = run_one(g, live, not args.no_judge)
        rs.append(r)
        flag = "OK " if r["behavior_ok"] in (True, None) and r["router_ok"] else "BAD"
        print(f"[{flag}] {r['id']:7} route={r['route']['level']}/{r['route']['intent']:14} status={r['status']:15} behavior={r['behavior_ok']} retr={r['retrieval_ok']} ({r['secs']}s)", flush=True)
    summary, patterns = summarize(rs), failure_patterns(rs)
    REPORTS.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (REPORTS / f"results-{stamp}.json").write_text(json.dumps({"meta": meta, "summary": summary, "patterns": patterns, "results": rs}, ensure_ascii=False, indent=1), encoding="utf-8")
    page = html_report(rs, summary, patterns, meta)
    (REPORTS / f"report-{stamp}.html").write_text(page, encoding="utf-8")
    (REPORTS / "latest.html").write_text(page, encoding="utf-8")
    print("\n== SUMMARY ==")
    for k, v in summary.items():
        print(f"  {k:28} {'n/a' if v[0] is None else str(v[0]) + '%' if 'judge' not in k else v[0]}  (n={v[1]})")
    print("\n== FAILURE PATTERNS ==")
    for k, v in patterns.items():
        print(f"  {k}: {', '.join(dict.fromkeys(v))}")
    print(f"\nreport: {REPORTS / 'latest.html'}")


if __name__ == "__main__":
    main()
