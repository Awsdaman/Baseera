"""Build a ready-to-run copy for reviewers: dist/Baseera-working-copy.zip

    python tools/make_package.py

Contains the committed files (read from HEAD with `git show`, so uncommitted edits never leak in: no secrets, no private folders), the built
sources database (data/db/baseera.sqlite and data/db/chroma) and the cached Dorar hadith-API answers (so Verify mode works offline). It never
includes .env, API key files, opt-in user reports (reports.sqlite), the challenge PDFs, local question files or the raw crawl caches.
"""
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dist" / "Baseera-working-copy.zip"
DB_FILES = [ROOT / "data" / "db" / "baseera.sqlite"] + sorted((ROOT / "data" / "db" / "chroma").rglob("*"))


def main():
    for must in (ROOT / "data" / "db" / "baseera.sqlite", ROOT / "data" / "db" / "chroma"):
        if not must.exists():
            sys.exit(f"missing {must}: run python ingest/build_all.py first")
    OUT.parent.mkdir(exist_ok=True)
    tracked = subprocess.run(["git", "ls-tree", "-r", "-z", "--name-only", "HEAD"], cwd=ROOT, capture_output=True, check=True).stdout.decode("utf-8").split("\0")
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        n = 0
        for rel in filter(None, tracked):
            blob = subprocess.run(["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True, check=True).stdout   # committed content, not the working tree
            z.writestr(f"Baseera/{rel}", blob)
            n += 1
        for p in DB_FILES:
            if p.is_file():
                z.write(p, f"Baseera/{p.relative_to(ROOT).as_posix()}")
                n += 1
        for p in sorted((ROOT / "data" / "cache").glob("*.json")):   # only Dorar hadith-API answers (Verify mode offline); icadb/crawl caches stay out
            if b'"ahadith"' in p.read_bytes()[:400]:
                z.write(p, f"Baseera/data/cache/{p.name}")
                n += 1
    print(f"{OUT} ({OUT.stat().st_size / 1e6:.0f} MB, {n} files)")


if __name__ == "__main__":
    main()
