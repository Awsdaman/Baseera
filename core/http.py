"""Shared HTTP client (truststore + browser UA) with an on-disk JSON cache and polite pacing."""
import hashlib
import json
import ssl
import threading
import time
from pathlib import Path

import httpx
import truststore

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"

_client = httpx.Client(verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT),
                       headers={"User-Agent": UA, "Accept": "application/json"},
                       follow_redirects=True, timeout=60)
_last = {}
_lock = threading.Lock()


def get_json(url: str, params: dict | None = None, delay: float = 0.3, retries: int = 4):
    """GET url -> parsed JSON, cached forever on disk by (url, params)."""
    key = hashlib.sha1((url + json.dumps(params or {}, sort_keys=True)).encode()).hexdigest()
    f = CACHE / f"{key}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    host = httpx.URL(url).host
    for attempt in range(retries):
        with _lock:  # reserve the next free slot for this host so parallel callers stay `delay` apart
            slot = max(time.time(), _last.get(host, 0) + delay)
            _last[host] = slot
        wait = slot - time.time()
        if wait > 0:
            time.sleep(wait)
        try:
            r = _client.get(url, params=params)
            if r.status_code == 200:
                data = r.json()
                f.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
                return data
            if r.status_code in (429, 503, 403):
                time.sleep(2 ** attempt * 2)
                continue
            r.raise_for_status()
        except httpx.TransportError:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"GET failed after {retries} tries: {url} {params}")
