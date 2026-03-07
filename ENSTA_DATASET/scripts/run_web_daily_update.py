from __future__ import annotations

import os
import sys
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
LOG_DIR = PROJECT_ROOT / "logs"
LOCK_DIR = PROJECT_ROOT / ".locks"
LOCK_FILE = LOCK_DIR / "web_daily.lock"

INDEX_DIR = PROJECT_ROOT / "04_index"
FAISS_PATH = INDEX_DIR / "ensta.faiss"
META_PATH = INDEX_DIR / "ensta_meta.jsonl"
RELOAD_FLAG = INDEX_DIR / "reload.flag"


def run_py(script_name: str):
    p = SCRIPTS / script_name
    if not p.exists():
        raise FileNotFoundError(f"Missing script: {p}")
    subprocess.run([sys.executable, str(p)], check=True, cwd=str(PROJECT_ROOT))


def backup_index():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bdir = INDEX_DIR / "backups" / ts
    bdir.mkdir(parents=True, exist_ok=True)
    if FAISS_PATH.exists():
        shutil.copy2(FAISS_PATH, bdir / FAISS_PATH.name)
    if META_PATH.exists():
        shutil.copy2(META_PATH, bdir / META_PATH.name)
    print(f"[OK] backup -> {bdir}")


def acquire_lock():
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    if LOCK_FILE.exists():
        raise RuntimeError(f"Pipeline already running (lock): {LOCK_FILE}")
    LOCK_FILE.write_text(f"pid={os.getpid()} time={datetime.now().isoformat()}", encoding="utf-8")


def release_lock():
    try:
        if LOCK_FILE.exists():
            LOCK_FILE.unlink()
    except Exception:
        pass


def main():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"web_daily_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    with log_path.open("w", encoding="utf-8") as log:
        class Tee:
            def __init__(self, *streams):
                self.streams = streams
            def write(self, s):
                for st in self.streams:
                    st.write(s)
                    st.flush()
            def flush(self):
                for st in self.streams:
                    st.flush()

        sys.stdout = Tee(sys.__stdout__, log)
        sys.stderr = Tee(sys.__stderr__, log)

        acquire_lock()
        try:
            print(f"[INFO] PROJECT_ROOT={PROJECT_ROOT}")
            print(f"[INFO] Python={sys.executable}")
            print(f"[INFO] Log={log_path}")

            # Web: crawl + clean docling
            run_py("crawl_web_docling.py")
            run_py("doclingp.py")

            # Rebuild complet (web + pdf + xlsx clean déjà présents)
            run_py("merge_corpus.py")     # -> 02_clean/corpus_clean.jsonl
            run_py("chunk_corpus.py")     # -> 03_chunks/chunks.jsonl

            # Backup de l'index actuel avant remplacement
            backup_index()

            # Embeddings + FAISS (doit être atomique: tmp + os.replace)
            run_py("embed_faiss.py")      # -> 04_index/ensta.faiss + 04_index/ensta_meta.jsonl

            # Signal pour le serveur (hot reload)
            RELOAD_FLAG.write_text(datetime.now().isoformat(), encoding="utf-8")
            print("[OK] wrote reload.flag")

            print("\n✅ WEB DAILY UPDATE DONE")
        finally:
            release_lock()


if __name__ == "__main__":
    main()