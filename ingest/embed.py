"""Embed all passages into Chroma (bge-m3 by default). Idempotent/resumable: skips ids already embedded.

    python ingest/embed.py            # embed everything missing
    EMBED_MODEL=intfloat/multilingual-e5-large python ingest/embed.py
"""
import os
import sys
import time
from pathlib import Path

import truststore

truststore.inject_into_ssl()  # HF downloads on this machine need the Windows cert store

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.db import connect  # noqa: E402
from core.embedding import COLLECTION, chroma_client, embed_texts, passage_embed_text  # noqa: E402

BATCH = 32


def main():
    con = connect()
    col = chroma_client().get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    have = set()
    got = col.get(include=[])
    have.update(got["ids"])
    # tafsir passages are long and numerous; they stay keyword-searchable and always link to their verse.
    skip = set(os.environ.get("EMBED_SKIP_TYPES", "tafsir").split(","))
    skip_sources = set(filter(None, os.environ.get("EMBED_SKIP_SOURCES", "icadb-books").split(",")))  # ~40k book chunks stay keyword-searchable (CPU embedding is ~4/s)
    rows = [r for r in con.execute("SELECT id,type,source,title,text_ar,text_en,meta FROM passages").fetchall()
            if r["type"] not in skip and r["source"] not in skip_sources]
    rows.sort(key=lambda r: len(passage_embed_text(r)))  # similar lengths per batch = less padding
    todo = [r for r in rows if r["id"] not in have]
    print(f"{len(rows)} passages, {len(have)} already embedded, {len(todo)} to do", flush=True)
    t0 = time.time()
    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i + BATCH]
        vecs = embed_texts([passage_embed_text(r) for r in chunk])
        col.add(ids=[r["id"] for r in chunk], embeddings=vecs, metadatas=[{"type": r["type"]} for r in chunk])
        done = i + len(chunk)
        if (i // BATCH) % 20 == 0:
            rate = done / (time.time() - t0)
            print(f"  {done}/{len(todo)}  {rate:.1f}/s  eta {(len(todo) - done) / rate / 60:.1f} min", flush=True)
    print("embedded total:", col.count())


if __name__ == "__main__":
    main()
