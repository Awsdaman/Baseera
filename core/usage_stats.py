"""Privacy-safe usage counters for the operator (live demo).

Only aggregate numbers are kept, in memory: how many questions / verifications, how they ended, how long they took, and
how many started in each hour. No question text, no IP address, no user agent, no identifiers. A snapshot of the same
aggregates is appended to logs/stats.jsonl every STATS_SNAPSHOT_S seconds (default 12 hours) and when the app stops.
"""
import collections
import json
import os
import threading
import time
from pathlib import Path

LOG = Path(__file__).resolve().parent.parent / "logs" / "stats.jsonl"
_lock = threading.Lock()


def _fresh() -> dict:
    return {"since": time.time(), "ask": 0, "verify": 0, "errors": 0, "outcomes": collections.Counter(), "secs_sum": 0.0, "secs_max": 0.0,
            "done": 0, "last": None, "hours": collections.OrderedDict()}


_s = _fresh()


def record(kind: str, status: str | None, secs: float, error: bool = False):
    now = time.time()
    hour = time.strftime("%Y-%m-%d %H:00", time.localtime(now))
    with _lock:
        _s[kind if kind in ("ask", "verify") else "ask"] += 1
        _s["done"] += 1
        _s["secs_sum"] += secs
        _s["secs_max"] = max(_s["secs_max"], secs)
        _s["errors"] += 1 if error else 0
        _s["outcomes"]["error" if error else (status or "unknown")] += 1
        _s["last"] = now
        _s["hours"][hour] = _s["hours"].get(hour, 0) + 1
        while len(_s["hours"]) > 96:
            _s["hours"].popitem(last=False)


def snapshot() -> dict:
    with _lock:
        n = _s["done"]
        return {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "since": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(_s["since"])),
                "questions": _s["ask"], "verifications": _s["verify"], "total": _s["ask"] + _s["verify"], "errors": _s["errors"],
                "outcomes": dict(_s["outcomes"]), "avg_secs": round(_s["secs_sum"] / n, 1) if n else 0, "max_secs": round(_s["secs_max"], 1),
                "last_request": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(_s["last"])) if _s["last"] else None,
                "per_hour": dict(_s["hours"])}


def write_snapshot():
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(snapshot(), ensure_ascii=False) + "\n")
    except Exception:
        pass


def snapshot_loop():
    every = int(os.environ.get("STATS_SNAPSHOT_S", 12 * 3600))
    while True:
        time.sleep(every)
        write_snapshot()


def reset():
    global _s
    with _lock:
        _s = _fresh()
