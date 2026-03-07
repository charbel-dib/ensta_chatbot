from __future__ import annotations

import json
import re
import hashlib
from pathlib import Path
from urllib.parse import urlparse
from collections import Counter
from typing import Dict, Any, Iterable, Tuple, Optional


# ====== PATHS (chez toi) ======
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "01_raw"
CLEAN_DIR = PROJECT_ROOT / "02_clean"

IN_WEB  = RAW_DIR / "web_docling.jsonl"
IN_PDF  = RAW_DIR / "pdfs_docling.jsonl"
IN_XLSX = RAW_DIR / "xlsx_courses.jsonl"

OUT_WEB  = CLEAN_DIR / "web_clean.jsonl"
OUT_PDF  = CLEAN_DIR / "pdfs_clean.jsonl"
OUT_XLSX = CLEAN_DIR / "xlsx_courses_clean.jsonl"

REPORT = CLEAN_DIR / "clean_report_docling.json"

# ====== RULES ======
MIN_CHARS_WEB = 250
MIN_CHARS_PDF = 250
MIN_CHARS_XLSX = 40

# Filtre web bruit/auth
DROP_WEB_PATH_CONTAINS = ["login", "register", "password", "/cgi", "account", "user"]

SEM_RE = re.compile(r"\bS(\d{1,2})\b", re.IGNORECASE)
SEM_WORD_RE = re.compile(r"semestre\s*(\d{1,2})", re.IGNORECASE)


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


def normalize_text(t: str) -> str:
    t = (t or "").replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t


def dedup_hash(text: str) -> str:
    norm = re.sub(r"\s+", " ", (text or "").strip().lower())[:8000]
    return hashlib.md5(norm.encode("utf-8", errors="ignore")).hexdigest()


def stable_id(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update((p or "").encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return h.hexdigest()[:24]


def alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    a = sum(ch.isalpha() for ch in text)
    return a / max(1, len(text))


def quality_ok(text: str, min_chars: int) -> Tuple[bool, str]:
    if len(text) < min_chars:
        return False, "too_short"
    if alpha_ratio(text) < 0.30:
        return False, "low_alpha"
    # menu pages: trop de liens pour peu de texte
    if text.count("](") > 60 and len(text) < 2500:
        return False, "too_many_links"
    return True, "ok"


def infer_semester_from_path(path_str: str) -> Optional[str]:
    if not path_str:
        return None
    m = SEM_RE.search(path_str)
    if m:
        return f"S{m.group(1)}"
    m2 = SEM_WORD_RE.search(path_str)
    if m2:
        return f"S{m2.group(1)}"
    return None


def clean_web() -> Tuple[int, Dict[str, int]]:
    stats = Counter()
    seen = set()
    out_rows = []

    for obj in read_jsonl(IN_WEB):
        stats["read"] += 1
        src = obj.get("source") or obj.get("url") or ""
        if not src:
            stats["drop_no_source"] += 1
            continue

        p = urlparse(src)
        path_l = (p.path or "").lower()
        if any(x in path_l for x in DROP_WEB_PATH_CONTAINS):
            stats["drop_auth_paths"] += 1
            continue

        text = normalize_text(obj.get("text") or "")
        ok, reason = quality_ok(text, MIN_CHARS_WEB)
        if not ok:
            stats[f"drop_{reason}"] += 1
            continue

        h = dedup_hash(text)
        if h in seen:
            stats["drop_dup"] += 1
            continue
        seen.add(h)

        out = {
            "id": obj.get("id") or stable_id("web", src),
            "source_type": "web",
            "source": src,
            "title": obj.get("title") or src,
            "text": text,
            "metadata": obj.get("metadata") or {},
        }

        md = out["metadata"]
        md.setdefault("host", (p.netloc or "").lower())
        md.setdefault("path", p.path or "/")
        md.setdefault("lang", "en" if (p.path or "").startswith("/en") else "fr")
        out["metadata"] = md

        out_rows.append(out)
        stats["kept"] += 1

    n = write_jsonl(OUT_WEB, out_rows)
    stats["written"] = n
    return n, dict(stats)


def clean_pdf() -> Tuple[int, Dict[str, int]]:
    stats = Counter()
    seen = set()
    out_rows = []

    for obj in read_jsonl(IN_PDF):
        stats["read"] += 1
        src = obj.get("source") or ""
        if not src:
            stats["drop_no_source"] += 1
            continue

        text = normalize_text(obj.get("text") or "")
        ok, reason = quality_ok(text, MIN_CHARS_PDF)
        if not ok:
            stats[f"drop_{reason}"] += 1
            continue

        h = dedup_hash(text)
        if h in seen:
            stats["drop_dup"] += 1
            continue
        seen.add(h)

        p = Path(src)
        out = {
            "id": obj.get("id") or stable_id("pdf", str(p)),
            "source_type": "pdf",
            "source": str(p),
            "title": obj.get("title") or p.stem,
            "text": text,
            "metadata": obj.get("metadata") or {},
        }

        md = out["metadata"]
        md.setdefault("filename", p.name)
        out["metadata"] = md

        out_rows.append(out)
        stats["kept"] += 1

    n = write_jsonl(OUT_PDF, out_rows)
    stats["written"] = n
    return n, dict(stats)


def clean_xlsx() -> Tuple[int, Dict[str, int]]:
    stats = Counter()
    seen = set()
    out_rows = []

    for obj in read_jsonl(IN_XLSX):
        stats["read"] += 1
        src = obj.get("source") or ""
        if not src:
            stats["drop_no_source"] += 1
            continue

        text = normalize_text(obj.get("text") or "")
        if len(text) < MIN_CHARS_XLSX:
            stats["drop_too_short"] += 1
            continue

        h = dedup_hash(text)
        if h in seen:
            stats["drop_dup"] += 1
            continue
        seen.add(h)

        p = Path(src)
        out = {
            "id": obj.get("id") or stable_id("xlsx", str(p)),
            "source_type": "xlsx_course_sheet",
            "source": str(p),
            "title": obj.get("title") or p.stem,
            "text": text,
            "metadata": obj.get("metadata") or {},
        }

        md = out["metadata"]

        # Remplir les null inférables
        if md.get("filename") in (None, ""):
            md["filename"] = p.name
            stats["infer_filename"] += 1

        rel = md.get("relative_path") or str(p)
        if md.get("semester") in (None, ""):
            sem = infer_semester_from_path(rel)
            if sem:
                md["semester"] = sem
                stats["infer_semester"] += 1

        if md.get("formation") in (None, ""):
            up = rel.upper()
            if "FISE" in up:
                md["formation"] = "FISE"
                stats["infer_formation"] += 1
            elif "FIPA" in up:
                md["formation"] = "FIPA"
                stats["infer_formation"] += 1

        # track_code : si vide, on laisse (mapping manuel ensuite)
        if md.get("track_code") is None:
            md["track_code"] = ""

        out["metadata"] = md

        out_rows.append(out)
        stats["kept"] += 1

    n = write_jsonl(OUT_XLSX, out_rows)
    stats["written"] = n
    return n, dict(stats)


def main():
    for p in [IN_WEB, IN_PDF, IN_XLSX]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    n_web, s_web = clean_web()
    n_pdf, s_pdf = clean_pdf()
    n_xlsx, s_xlsx = clean_xlsx()

    report = {"web": s_web, "pdf": s_pdf, "xlsx": s_xlsx}
    with REPORT.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n✅ CLEAN DONE (Docling raw -> 02_clean)")
    print("web  :", n_web, "->", OUT_WEB)
    print("pdf  :", n_pdf, "->", OUT_PDF)
    print("xlsx :", n_xlsx, "->", OUT_XLSX)
    print("report:", REPORT)


if __name__ == "__main__":
    main()
