"""Build a ready-to-run copy for reviewers: dist/Baseera-working-copy.zip

    python tools/make_package.py

Contains the committed files (git archive of HEAD: no secrets, no private folders) plus the built sources database (data/db/baseera.sqlite and
data/db/chroma). It never includes .env, API key files, opt-in user reports (reports.sqlite), the challenge PDFs or local question files.
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
    tracked = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout.decode("utf-8").split("\0")
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        n = 0
        for rel in filter(None, tracked):
            p = ROOT / rel
            if p.is_file():
                z.write(p, f"Baseera/{rel}")
                n += 1
        for p in DB_FILES:
            if p.is_file():
                z.write(p, f"Baseera/{p.relative_to(ROOT).as_posix()}")
                n += 1
    print(f"{OUT} ({OUT.stat().st_size / 1e6:.0f} MB, {n} files)")


if __name__ == "__main__":
    main()
