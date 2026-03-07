from __future__ import annotations
import json, re, hashlib
from pathlib import Path
from collections import Counter
from typing import Dict, Any, Iterable, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = PROJECT_ROOT / "02_clean"

IN_FILES = [
    CLEAN_DIR / "web_clean.jsonl",
    CLEAN_DIR / "pdfs_clean.jsonl",
    CLEAN_DIR / "xlsx_courses_clean.jsonl",
]

OUT_FILE = CLEAN_DIR / "corpus_clean.jsonl"

MIN_CHARS = 50

def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
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

def quality_ok(text: str) -> Tuple[bool, str]:
    if len(text) < MIN_CHARS:
        return False, "too_short"
    alpha = sum(ch.isalpha() for ch in text)
    if alpha / max(1, len(text)) < 0.30:
        return False, "low_alpha"
    md_links = text.count("](")
    if md_links > 40 and len(text) < 2500:
        return False, "too_many_links"
    return True, "ok"

def dedup_hash(text: str) -> str:
    norm = re.sub(r"\s+", " ", text.strip().lower())[:8000]
    return hashlib.md5(norm.encode("utf-8", errors="ignore")).hexdigest()

def main():
    for p in IN_FILES:
        if not p.exists():
            raise FileNotFoundError(f"Missing input: {p}")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    if OUT_FILE.exists():
        OUT_FILE.unlink()

    stats = Counter()
    seen = set()
    kept = 0

    for inp in IN_FILES:
        for obj in read_jsonl(inp):
            stats["read"] += 1

            text = normalize_text(obj.get("text","") or "")
            ok, reason = quality_ok(text)
            if not ok:
                stats[f"drop_{reason}"] += 1
                continue

            h = dedup_hash(text)
            if h in seen:
                stats["drop_dup"] += 1
                continue
            seen.add(h)

            out = {
                "id": obj.get("id"),
                "source_type": obj.get("source_type"),
                "source": obj.get("source"),
                "title": obj.get("title") or obj.get("source"),
                "text": text,
                "metadata": obj.get("metadata") or {},
            }

            with OUT_FILE.open("a", encoding="utf-8") as f:
                f.write(json.dumps(out, ensure_ascii=False) + "\n")

            kept += 1
            stats["kept"] += 1
            if kept % 200 == 0:
                print(f"[OK] kept={kept}")

    print("\n✅ MERGE FIXED DONE")
    print(f"Output: {OUT_FILE.resolve()}")
    for k in sorted(stats):
        print(f"{k:18s}: {stats[k]}")

if __name__ == "__main__":
    main()
