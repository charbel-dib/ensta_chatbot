from __future__ import annotations

import json
import re
import hashlib
from pathlib import Path
from typing import Dict, Any, Iterable, List, Tuple, Set
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INP = PROJECT_ROOT / "02_clean" / "corpus_clean.jsonl"
OUT = PROJECT_ROOT / "03_chunks" / "chunks.jsonl"

# ---- Chunking params ----
# Heuristique: ~4 chars/token en français/anglais mixed
TARGET_CHARS = 2800   # ~700 tokens
OVERLAP_CHARS = 450   # ~110 tokens overlap
MIN_CHARS = 350       # filtre chunks trop courts

# Markdown headings
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)\s*$")

# Extra cleanup
WS_RE = re.compile(r"[ \t]+")
MULTI_NL_RE = re.compile(r"\n{3,}")


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def stable_id(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update((p or "").encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return h.hexdigest()[:24]


def normalize_text(t: str) -> str:
    t = (t or "").replace("\r\n", "\n").replace("\r", "\n")
    t = WS_RE.sub(" ", t)
    t = MULTI_NL_RE.sub("\n\n", t)
    return t.strip()


def split_by_headings(text: str) -> List[Tuple[str, str]]:
    """
    Retourne une liste de sections: (header_path, section_text)
    header_path = "H1 > H2 > H3" (selon markdown #)
    """
    lines = text.split("\n")
    stack: List[Tuple[int, str]] = []
    sections: List[Tuple[str, List[str]]] = []

    cur_lines: List[str] = []
    cur_path = "ROOT"

    def stack_to_path() -> str:
        if not stack:
            return "ROOT"
        return " > ".join([h for _, h in stack])

    def flush():
        nonlocal cur_lines, cur_path
        body = "\n".join(cur_lines).strip()
        if body:
            sections.append((cur_path, cur_lines))
        cur_lines = []

    for ln in lines:
        m = HEADING_RE.match(ln)
        if m:
            # nouveau heading => flush body précédent
            flush()
            level = len(m.group(1))
            title = m.group(2).strip()

            # ajuste stack
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            cur_path = stack_to_path()
            continue

        cur_lines.append(ln)

    flush()

    # convert to (path, text)
    out: List[Tuple[str, str]] = []
    for path, ls in sections:
        out.append((path, normalize_text("\n".join(ls))))
    return out


def chunk_section(section_text: str) -> List[str]:
    """
    Découpe une section en chunks par longueur (chars) avec overlap.
    """
    t = section_text.strip()
    if len(t) <= TARGET_CHARS:
        return [t] if len(t) >= MIN_CHARS else []

    chunks = []
    start = 0
    n = len(t)
    while start < n:
        end = min(n, start + TARGET_CHARS)
        # essaie de couper proprement sur une frontière (double newline, newline, point)
        cut = end

        window = t[start:end]
        # priorité: paragraph boundary
        idx = window.rfind("\n\n")
        if idx >= int(TARGET_CHARS * 0.55):
            cut = start + idx
        else:
            idx = window.rfind("\n")
            if idx >= int(TARGET_CHARS * 0.65):
                cut = start + idx
            else:
                idx = window.rfind(". ")
                if idx >= int(TARGET_CHARS * 0.70):
                    cut = start + idx + 1  # include dot

        chunk = t[start:cut].strip()
        if len(chunk) >= MIN_CHARS:
            chunks.append(chunk)

        if cut >= n:
            break

        # overlap
        start = max(0, cut - OVERLAP_CHARS)

    return chunks


def dedup_hash(text: str) -> str:
    norm = re.sub(r"\s+", " ", text.strip().lower())[:6000]
    return hashlib.md5(norm.encode("utf-8", errors="ignore")).hexdigest()


def main():
    if not INP.exists():
        raise FileNotFoundError(INP)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        OUT.unlink()

    stats = Counter()
    seen = set()

    out_rows: List[Dict[str, Any]] = []
    chunk_count = 0

    for doc in read_jsonl(INP):
        stats["docs_read"] += 1

        doc_id = doc.get("id") or stable_id("doc", doc.get("source_type",""), doc.get("source",""))
        source_type = doc.get("source_type")
        source = doc.get("source")
        title = doc.get("title") or source
        md = doc.get("metadata") or {}

        text = normalize_text(doc.get("text", "") or "")
        if not text or len(text) < MIN_CHARS:
            stats["docs_drop_short"] += 1
            continue

        # split by headings (works for markdown-ish web/pdf)
        sections = split_by_headings(text)

        # if no headings, sections will be ROOT chunk
        if not sections:
            sections = [("ROOT", text)]

        for header_path, sec_text in sections:
            for chunk in chunk_section(sec_text):
                h = dedup_hash(chunk)
                if h in seen:
                    stats["chunk_drop_dup"] += 1
                    continue
                seen.add(h)

                chunk_id = stable_id("chunk", doc_id, header_path, str(chunk_count))
                row = {
                    "id": chunk_id,
                    "doc_id": doc_id,
                    "source_type": source_type,
                    "source": source,
                    "title": title,
                    "header_path": header_path,
                    "text": chunk,
                    "metadata": md,
                }
                out_rows.append(row)
                chunk_count += 1
                stats["chunks_kept"] += 1

        if stats["docs_read"] % 200 == 0:
            print(f"[OK] docs={stats['docs_read']} chunks={stats['chunks_kept']}")

    n = write_jsonl(OUT, out_rows)

    print("\n✅ CHUNKING DONE")
    print(f"Input : {INP.resolve()}")
    print(f"Output: {OUT.resolve()}")
    print(f"Chunks written: {n}")
    for k in sorted(stats):
        print(f"{k:18s}: {stats[k]}")


if __name__ == "__main__":
    main()
