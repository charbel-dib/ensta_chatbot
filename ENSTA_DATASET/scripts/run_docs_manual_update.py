from __future__ import annotations

import sys
import subprocess
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = PROJECT_ROOT / "scripts"
INDEX_DIR = PROJECT_ROOT / "04_index"

def run_existing(candidates: list[str], step: str):
    for name in candidates:
        p = SCRIPTS / name
        if p.exists():
            print(f"[INFO] {step}: running {name}")
            subprocess.run([sys.executable, str(p)], check=True, cwd=str(PROJECT_ROOT))
            return
    raise FileNotFoundError(f"{step}: none of these scripts exist: {candidates}")

def run_py(name: str, step: str):
    p = SCRIPTS / name
    if not p.exists():
        raise FileNotFoundError(f"{step}: missing {p}")
    subprocess.run([sys.executable, str(p)], check=True, cwd=str(PROJECT_ROOT))

def main():
    # 1) Update PDFs raw (manuel) -> 01_raw/pdfs_docling.jsonl (ou autre)
    run_existing(
        ["ingest_pdfs.py", "pdfs_docling.py", "crawl_pdfs_docling.py"],
        "PDF ingest"
    )

    # 2) Update XLSX raw -> 01_raw/xlsx_courses.jsonl (ou autre)
    run_existing(
        ["ingest_xlsx_courses.py", "xlsx_docling.py", "ingest_xlsx.py"],
        "XLSX ingest"
    )

    # 3) Clean docling raw -> 02_clean/pdfs_clean.jsonl + xlsx_courses_clean.jsonl
    run_py("doclingp.py", "Docling clean")

    # 4) Rebuild full index
    run_py("merge_corpus.py", "Merge corpus")
    run_py("chunk_corpus.py", "Chunk corpus")
    run_py("embed_faiss.py", "Embed + FAISS")

    (INDEX_DIR / "reload.flag").write_text(datetime.now().isoformat(), encoding="utf-8")
    print("\n✅ DOCS MANUAL UPDATE DONE")

if __name__ == "__main__":
    main()