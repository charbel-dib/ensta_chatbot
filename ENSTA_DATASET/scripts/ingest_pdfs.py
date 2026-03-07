from __future__ import annotations

import json
import re
import hashlib
from pathlib import Path
from typing import List, Any

import pdfplumber


# ====== PATHS ======
PROJECT_ROOT = Path(__file__).resolve().parents[1]          # ENSTA_DATASET/
PDF_DIR = PROJECT_ROOT / "01_raw" / "pdfs"
OUT_JSONL = PROJECT_ROOT / "02_clean" / "pdfs_clean.jsonl"

PRINT_EVERY = 20


def stable_id(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return h.hexdigest()[:24]


def table_to_md(table: List[List[Any]]) -> str:
    """Convertit une table (liste de lignes) en Markdown."""
    if not table:
        return ""
    rows = [[("" if c is None else str(c)).strip() for c in r] for r in table]
    w = max(len(r) for r in rows)
    rows = [r + [""] * (w - len(r)) for r in rows]
    header = rows[0]
    sep = ["---"] * w
    out = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for r in rows[1:]:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def clean_text(t: str) -> str:
    t = (t or "").replace("\r\n", "\n").replace("\r", "\n")
    # compresse les blancs
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t


def extract_pdf(path: Path) -> str:
    blocks = []
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            txt = (page.extract_text() or "").strip()
            txt = clean_text(txt)

            if txt:
                blocks.append(f"## Page {i}\n{txt}")

            # tables
            try:
                tables = page.extract_tables() or []
            except Exception:
                tables = []

            for t_idx, t in enumerate(tables, start=1):
                md = table_to_md(t)
                md = clean_text(md)
                if md:
                    blocks.append(f"### Table (Page {i}, #{t_idx})\n{md}")

    return clean_text("\n\n".join(blocks))


def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
                if "id" in o:
                    done.add(o["id"])
            except Exception:
                continue
    return done


def main():
    print(f"[DEBUG] PDF_DIR : {PDF_DIR}")
    print(f"[DEBUG] OUT     : {OUT_JSONL}")

    if not PDF_DIR.exists():
        raise FileNotFoundError(f"Dossier introuvable: {PDF_DIR}")

    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSONL.touch(exist_ok=True)

    done = load_done_ids(OUT_JSONL)
    pdfs = sorted(PDF_DIR.rglob("*.pdf"))
    print(f"[INFO] PDFs trouvés: {len(pdfs)} | déjà traités: {len(done)}")

    kept = 0
    for idx, p in enumerate(pdfs, start=1):
        doc_id = stable_id("pdf", str(p.resolve()))
        if doc_id in done:
            continue

        text = extract_pdf(p)
        if len(text) < 200:  # trop vide
            continue

        obj = {
            "id": doc_id,
            "source_type": "pdf",
            "source": str(p.resolve()),
            "title": p.stem,
            "text": text,
            "metadata": {
                "filename": p.name,
                "relative_path": str(p.relative_to(PDF_DIR)),
            },
        }
        append_jsonl(OUT_JSONL, obj)
        done.add(doc_id)
        kept += 1

        if kept % PRINT_EVERY == 0:
            print(f"[OK] kept={kept} / scanned={idx}  last={p.name}")

    print("\n✅ PDF INGEST DONE")
    print(f"Output: {OUT_JSONL.resolve()}")
    print(f"Kept  : {kept} (nouveaux)")

if __name__ == "__main__":
    main()
