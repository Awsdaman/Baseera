"""Run every ingest step in order (idempotent):  python ingest/build_all.py [--no-embed]"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS = ["download_raw.py", "ingest_quran.py", "ingest_hadeethenc.py", "ingest_bayyinat.py", "ingest_icadb.py", "embed.py"]


def main():
    steps = [s for s in STEPS if not (s == "embed.py" and "--no-embed" in sys.argv)]
    for s in steps:
        print(f"\n=== {s} ===", flush=True)
        rc = subprocess.call([sys.executable, str(HERE / s)])
        if rc != 0:
            sys.exit(f"{s} failed (exit {rc})")
    print("\nDone. Start the app with:  uvicorn api.main:app")


if __name__ == "__main__":
    main()
