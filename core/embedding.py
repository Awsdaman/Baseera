"""Embedding model + Chroma access, shared by ingestion and retrieval."""
import json
import os
import threading
from functools import lru_cache
from pathlib import Path

from core.normalize import normalize_ar

ROOT = Path(__file__).resolve().parent.parent
CHROMA_DIR = ROOT / "data" / "db" / "chroma"
COLLECTION = "baseera"
MODEL_NAME = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
MAX_LEN = int(os.environ.get("EMBED_MAXLEN", 160))
# e5 models need "query: " / "passage: " prefixes; bge-m3 does not.
_E5 = "e5" in MODEL_NAME.lower()


_ENCODE_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def model():
    import truststore
    truststore.inject_into_ssl()
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(MODEL_NAME, device="cpu")
    m.max_seq_length = MAX_LEN
    return m


def embed_texts(texts: list[str], query: bool = False) -> list[list[float]]:
    if _E5:
        texts = [("query: " if query else "passage: ") + t for t in texts]
    with _ENCODE_LOCK:  # one encode at a time: torch inference is fast next to LLM calls, and this avoids CPU thrash
        return model().encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=False).tolist()


@lru_cache(maxsize=1)
def chroma_client():
    import chromadb
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def passage_embed_text(r) -> str:
    """Natural-language text embedded for a passage (diacritics stripped for Arabic; English kept)."""
    ar = r["text_ar"] or ""
    if r["type"] == "quran":
        return f"{normalize_ar(ar)}\n{r['text_en'] or ''}"
    if r["type"] == "hadith":
        return f"{r['title'] or ''}\n{ar[:600]}\n{r['text_en'] or ''}"[:1500]
    if r["type"] in ("qa", "term"):
        return f"{r['title'] or ''}\n{ar}\n{r['text_en'] or ''}"[:1500]
    return ar[:1500]
