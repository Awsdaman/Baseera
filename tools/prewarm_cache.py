"""Fill the LLM cache for the live demo, then prove every demo question is answered from it.

    python tools/prewarm_cache.py            # pass 1 (writes data/cache/llm) + pass 2 (read-only check)
    python tools/prewarm_cache.py --check    # pass 2 only

Uses the exact live settings from .env.demo (the cache key covers model, prompts and token limits), so run it AFTER the
last code / database / settings change. Questions: the golden + reliability evals, the UI's example and tool questions.
"""
import argparse
import json
import os
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_env_file(path: Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" in line:
            k, v = line.split("=", 1)
            os.environ[k.strip()] = v.strip()


def ui_questions() -> list[str]:
    html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    out = []
    for block in re.findall(r"toolQs:\{(.*?)\},\n\s+ex:", html, re.S):
        out += re.findall(r"'((?:[^'\\]|\\.)*)'", block)
    for line in re.findall(r"\n\s+exText:\{(.*?)\},?\n", html):
        out += [q for q in re.findall(r"\w+:'((?:[^'\\]|\\.)*)'", line)]
    return [q.replace("\\'", "'") for q in out]


def jsonl_inputs(path: Path) -> list[str]:
    return [json.loads(x)["input"] for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def viral() -> str:
    html = (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    m = re.search(r"const VIRAL = '(.*?)';\n", html, re.S)
    return m.group(1).replace("\\n", "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="read-only check only")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--env", default=str(ROOT / ".env.demo"))
    args = ap.parse_args()
    load_env_file(Path(args.env))
    from core import llm as L, pipeline, verifier_mode

    qs = list(dict.fromkeys(q.strip() for q in ui_questions() + jsonl_inputs(ROOT / "evals" / "golden.jsonl")
                            + jsonl_inputs(ROOT / "evals" / "reliability.jsonl") if q.strip()))
    vt = viral().strip()
    print(f"{len(qs)} questions (+ viral verify), model {L.describe()['generate']}, cache {os.environ.get('LLM_CACHE')}")

    def one(q):
        t0 = time.time()
        try:
            r = pipeline.ask(q)
            st = r.get("status")
        except Exception as e:
            st = "EXC " + type(e).__name__
        return q, st, time.time() - t0

    def run(label):
        t0 = time.time()
        with ThreadPoolExecutor(args.workers) as ex:
            res = list(ex.map(one, qs))
        t = time.time()
        try:
            verifier_mode.verify_text(vt)
        except Exception as e:
            print("viral verify failed:", type(e).__name__)
        res.append(("<viral verify>", "verify", time.time() - t))
        ts = [x[2] for x in res]
        print(f"[{label}] {len(res)} done in {time.time() - t0:.0f}s  p50 {statistics.median(ts):.1f}s  max {max(ts):.1f}s  errors {sum(1 for x in res if str(x[1]).startswith('EXC') or x[1] in (None, 'error'))}")
        return res

    if not args.check:
        os.environ["LLM_CACHE"] = "1"
        run("pass 1 (warming)")
    os.environ["LLM_CACHE"] = "readonly"
    res = run("pass 2 (read-only)")
    slow = [x for x in res if x[2] > 3]
    print(f"slow (>3 s, i.e. cache misses): {len(slow)}")
    for q, st, dt in slow[:20]:
        print(f"  {dt:5.1f}s {st}  {q[:70]!r}")
    sys.exit(1 if slow else 0)


if __name__ == "__main__":
    main()
