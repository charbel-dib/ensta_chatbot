from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INP = PROJECT_ROOT / "03_chunks" / "chunks.jsonl"
OUT_DIR = PROJECT_ROOT / "04_index"
OUT_INDEX = OUT_DIR / "ensta.faiss"
OUT_META = OUT_DIR / "ensta_meta.jsonl"

MODEL_NAME = "intfloat/multilingual-e5-base"
BATCH_SIZE = 128
DOC_PREFIX = "passage: "  # E5 documents


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def normalize_text(t: str) -> str:
    t = (t or "").replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t


def atomic_replace(src: Path, dst: Path, retries: int = 8, sleep_s: float = 0.25) -> None:
    """
    Windows: os.replace peut échouer si un processus lit/scan le fichier à l'instant T.
    On retry quelques fois.
    """
    last_err: Exception | None = None
    for _ in range(retries):
        try:
            os.replace(str(src), str(dst))
            return
        except Exception as e:
            last_err = e
            time.sleep(sleep_s)
    raise RuntimeError(f"atomic_replace failed: {src} -> {dst}: {last_err!r}") from last_err


def main():
    if not INP.exists():
        raise FileNotFoundError(INP)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    OUT_INDEX_TMP = OUT_DIR / (OUT_INDEX.name + ".tmp")
    OUT_META_TMP = OUT_DIR / (OUT_META.name + ".tmp")

    # nettoyage des tmp (restes run précédent)
    if OUT_INDEX_TMP.exists():
        OUT_INDEX_TMP.unlink()
    if OUT_META_TMP.exists():
        OUT_META_TMP.unlink()

    print(f"[INFO] input : {INP}")
    print(f"[INFO] index : {OUT_INDEX}")
    print(f"[INFO] meta  : {OUT_META}")
    print(f"[INFO] model : {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    vectors: List[np.ndarray] = []
    metas: List[Dict[str, Any]] = []

    batch_texts: List[str] = []
    batch_metas: List[Dict[str, Any]] = []
    total = 0

    for obj in read_jsonl(INP):
        text = normalize_text(obj.get("text", "") or "")
        if not text:
            continue

        meta = {
            "id": obj.get("id"),
            "doc_id": obj.get("doc_id"),
            "source_type": obj.get("source_type"),
            "source": obj.get("source"),
            "title": obj.get("title"),
            "header_path": obj.get("header_path"),
            "metadata": obj.get("metadata") or {},
            "text": text,
        }

        batch_texts.append(DOC_PREFIX + text)
        batch_metas.append(meta)

        if len(batch_texts) >= BATCH_SIZE:
            embs = model.encode(
                batch_texts,
                batch_size=BATCH_SIZE,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for e, m in zip(embs, batch_metas):
                vectors.append(np.asarray(e, dtype=np.float32))
                metas.append(m)

            total += len(batch_texts)
            if total % 512 == 0:
                print(f"[OK] embedded={total}")

            batch_texts.clear()
            batch_metas.clear()

    # flush
    if batch_texts:
        embs = model.encode(
            batch_texts,
            batch_size=len(batch_texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        for e, m in zip(embs, batch_metas):
            vectors.append(np.asarray(e, dtype=np.float32))
            metas.append(m)
        total += len(batch_texts)

    if not vectors:
        raise RuntimeError("Aucun embedding produit (chunks vides?).")

    X = np.vstack(vectors).astype(np.float32)
    dim = X.shape[1]

    index = faiss.IndexFlatIP(dim)  # cosine via embeddings normalisés
    index.add(X)

    # 1) écrire en tmp
    faiss.write_index(index, str(OUT_INDEX_TMP))

    with OUT_META_TMP.open("w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())  # important si crash

    # 2) swap atomique (meta puis index ou l’inverse: OK tant que reload.flag est écrit après)
    atomic_replace(OUT_META_TMP, OUT_META)
    atomic_replace(OUT_INDEX_TMP, OUT_INDEX)

    print("\n✅ EMBEDDING DONE")
    print(f"Vectors: {index.ntotal}  dim={dim}")
    print(f"Index  : {OUT_INDEX.resolve()}")
    print(f"Meta   : {OUT_META.resolve()}")


if __name__ == "__main__":
    main()