"""Download the raw files that can be fetched automatically; report which ones must be placed by hand.

Manual (qurancomplex.gov.sa/quran-dev requires the website):
    data/raw/quran/kfgqpc_hafs_v30.zip        (Hafs v3.0, JSON with real Unicode text)
    data/raw/tafsir/hafs_tafseerMouaser_v3.zip (Tafseer Muyassar, verse level)
"""
import ssl
import sys
from pathlib import Path

import httpx
import truststore

ROOT = Path(__file__).resolve().parent.parent
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"
AUTO = {
    "data/raw/quran/english_saheeh.zip": "https://quranenc.com/downloads/sqlite/english_saheeh.zip",
    "data/raw/bayyinat/bayyinat_qa.pdf": "https://dawa.center/storage/files/AMYj6DfmHlSnZ766Zz0VlBNwmYtdwhAl31XMETlT.pdf",
}
MANUAL = ["data/raw/quran/kfgqpc_hafs_v30.zip", "data/raw/tafsir/hafs_tafseerMouaser_v3.zip"]


def main() -> int:
    client = httpx.Client(verify=truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT), headers={"User-Agent": UA}, follow_redirects=True, timeout=120)
    for rel, url in AUTO.items():
        dest = ROOT / rel
        if dest.exists() and dest.stat().st_size > 1000:
            print(f"have     {rel}")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        with client.stream("GET", url) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_bytes():
                    f.write(chunk)
        print(f"fetched  {rel} ({dest.stat().st_size // 1024} KB)")
    missing = [m for m in MANUAL if not (ROOT / m).exists()]
    for m in missing:
        print(f"MISSING  {m}  -> download manually from https://qurancomplex.gov.sa/quran-dev")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
