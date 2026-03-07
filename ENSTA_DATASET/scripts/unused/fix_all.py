from __future__ import annotations

import json
import re
import hashlib
from pathlib import Path
from urllib.parse import urlparse
from collections import Counter
from typing import Dict, Any, Iterable, Tuple, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]  # ENSTA_DATASET/
CLEAN_DIR = PROJECT_ROOT / "02_clean"

INPUTS = {
    "web":  CLEAN_DIR / "web_clean.jsonl",
    "pdf":  CLEAN_DIR / "pdfs_clean.jsonl",
    "xlsx": CLEAN_DIR / "xlsx_courses_clean.jsonl",
}

OUTPUTS = {
    "web":  CLEAN_DIR / "web_clean.fixed.jsonl",
    "pdf":  CLEAN_DIR / "pdfs_clean.fixed.jsonl",
    "xlsx": CLEAN_DIR / "xlsx_courses_clean.fixed.jsonl",
}

REPORT = CLEAN_DIR / "fix_report.json"

# ---------- Regex / heuristiques ----------
NAV_LINE_RE = re.compile(r"^\s*(\*|-)\s*\[[^\]]{1,70}\]\([^)]+\)\s*$")
IMG_LINE_RE = re.compile(r"^\s*!\[[^\]]*\]\([^)]+\)\s*$")
COOKIE_RE = re.compile(r"\b(cookie|cookies|rgpd|privacy|confidentialit|consent)\b", re.I)
SEM_RE = re.compile(r"\bS(\d{1,2})\b", re.IGNORECASE)
SEM_WORD_RE = re.compile(r"semestre\s*(\d{1,2})", re.IGNORECASE)

# Tu peux enrichir si besoin
DROP_WEB_PATH_CONTAINS = ["login", "register", "password", "/cgi", "account", "user"]


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


def fix_mojibake(s: str, stats: Counter, field: str) -> str:
    if not isinstance(s, str) or not s:
        return s
    if "Ã" in s or "Â" in s:
        try:
            fixed = s.encode("latin1").decode("utf-8")
            if fixed != s:
                stats[f"enc_fixed:{field}"] += 1
            return fixed
        except Exception:
            stats[f"enc_failed:{field}"] += 1
            return s
    return s


def normalize_whitespace(text: str) -> str:
    t = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def clean_markdown_like(text: str, stats: Counter) -> str:
    """
    Nettoyage “anti boilerplate” doux (safe) pour web/pdf/xlsx.
    """
    t = normalize_whitespace(text)
    if not t:
        return t

    kept = []
    removed = 0
    for ln in t.split("\n"):
        s = ln.strip()
        if not s:
            kept.append("")
            continue
        if IMG_LINE_RE.match(s):
            removed += 1
            continue
        if NAV_LINE_RE.match(s):
            removed += 1
            continue
        if COOKIE_RE.search(s) and len(s) < 160:
            removed += 1
            continue
        # lignes "trop linkées" = menu
        if s.count("](") >= 2 and len(s) < 160:
            removed += 1
            continue
        kept.append(ln)

    if removed:
        stats["boilerplate_lines_removed"] += removed

    out = "\n".join(kept)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out


def ensure_schema(obj: Dict[str, Any], source_kind: str, stats: Counter) -> Dict[str, Any]:
    """
    Standardise schema minimal et corrige nulls quand possible.
    """
    out = {
        "id": obj.get("id"),
        "source_type": obj.get("source_type") or source_kind,
        "source": obj.get("source"),
        "title": obj.get("title") or obj.get("source"),
        "text": obj.get("text") or "",
        "metadata": obj.get("metadata") or {},
    }

    # Fix encoding fields (top-level)
    out["source"] = fix_mojibake(out["source"] or "", stats, f"{source_kind}.source")
    out["title"]  = fix_mojibake(out["title"] or "",  stats, f"{source_kind}.title")
    out["text"]   = fix_mojibake(out["text"] or "",   stats, f"{source_kind}.text")

    # Fix encoding fields (metadata strings)
    md = out["metadata"]
    for k, v in list(md.items()):
        if isinstance(v, str):
            md[k] = fix_mojibake(v, stats, f"{source_kind}.metadata.{k}")

    out["metadata"] = md

    # Ensure id exists
    if not out["id"]:
        out["id"] = stable_id(out["source_type"], out["source"] or out["title"] or "")
        stats["id_generated"] += 1

    # Clean text
    before_len = len(out["text"])
    out["text"] = clean_markdown_like(out["text"], stats)
    if len(out["text"]) != before_len:
        stats["text_cleaned"] += 1

    # Fill nulls depending on source_type
    if source_kind == "web":
        out = fix_web_nulls(out, stats)
    elif source_kind == "pdf":
        out = fix_pdf_nulls(out, stats)
    elif source_kind == "xlsx":
        out = fix_xlsx_nulls(out, stats)

    return out


def fix_web_nulls(obj: Dict[str, Any], stats: Counter) -> Dict[str, Any]:
    url = obj.get("source") or ""
    p = urlparse(url)
    md = obj.get("metadata") or {}

    # host/path always inferable
    if md.get("host") in (None, ""):
        md["host"] = p.netloc.lower()
        stats["web_infer:host"] += 1
    if md.get("path") in (None, ""):
        md["path"] = p.path or "/"
        stats["web_infer:path"] += 1
    if md.get("lang") in (None, ""):
        # heuristique simple
        md["lang"] = "en" if (p.path or "").startswith("/en") else "fr"
        stats["web_infer:lang"] += 1

    # filter: drop clearly useless paths (security)
    path_l = (p.path or "").lower()
    if any(x in path_l for x in DROP_WEB_PATH_CONTAINS):
        stats["web_drop:auth_paths"] += 1
        return {}  # signal drop

    obj["metadata"] = md
    return obj


def fix_pdf_nulls(obj: Dict[str, Any], stats: Counter) -> Dict[str, Any]:
    md = obj.get("metadata") or {}
    src = obj.get("source") or ""
    p = Path(src)

    if md.get("filename") in (None, "") and p.name:
        md["filename"] = p.name
        stats["pdf_infer:filename"] += 1

    # file stem as title if title empty
    if (not obj.get("title")) and p.stem:
        obj["title"] = p.stem
        stats["pdf_infer:title_from_filename"] += 1

    obj["metadata"] = md
    return obj


def infer_semester_from_path(path_str: str) -> Optional[str]:
    if not path_str:
        return None
    # ex: ...\S4\...
    m = SEM_RE.search(path_str)
    if m:
        return f"S{m.group(1)}"
    # ex: "semestre 2"
    m2 = SEM_WORD_RE.search(path_str)
    if m2:
        return f"S{m2.group(1)}"
    return None


def infer_track_code_from_path(path_str: str) -> Optional[str]:
    """
    Heuristique: si on a ...\<TRACK>\S4\file.xlsx => TRACK = folder before semester.
    Sinon None.
    """
    if not path_str:
        return None
    parts = re.split(r"[\\/]+", path_str)
    # find Sx index
    for i, part in enumerate(parts):
        if re.fullmatch(r"S\d{1,2}", part, flags=re.IGNORECASE):
            if i > 0:
                code = parts[i - 1].strip().upper()
                # ignore obvious non-codes
                if len(code) <= 12 and "FICHES" not in code and "SEMESTRE" not in code:
                    return code
    return None


def fix_xlsx_nulls(obj: Dict[str, Any], stats: Counter) -> Dict[str, Any]:
    md = obj.get("metadata") or {}
    rel = md.get("relative_path") or obj.get("source") or ""

    # formation
    if md.get("formation") in (None, ""):
        up = rel.upper()
        if "FISE" in up:
            md["formation"] = "FISE"
            stats["xlsx_infer:formation"] += 1
        elif "FIPA" in up:
            md["formation"] = "FIPA"
            stats["xlsx_infer:formation"] += 1

    # semester (Sx or "semestre x")
    if md.get("semester") in (None, ""):
        sem = infer_semester_from_path(rel)
        if sem:
            md["semester"] = sem
            stats["xlsx_infer:semester"] += 1

    # track_code
    tc = md.get("track_code")
    if tc is None:
        tc = ""
    if not isinstance(tc, str):
        tc = str(tc)

    if tc.strip() == "":
        inferred = infer_track_code_from_path(rel)
        if inferred:
            md["track_code"] = inferred
            stats["xlsx_infer:track_code"] += 1
        else:
            md["track_code"] = ""
            stats["xlsx_missing:track_code"] += 1
    else:
        md["track_code"] = tc.strip().upper()

    # track_label remains manual (if absent, we keep null but mark)
    if md.get("track_label") in (None, ""):
        stats["xlsx_missing:track_label"] += 1

    # filename
    if md.get("filename") in (None, ""):
        src = obj.get("source") or ""
        p = Path(src)
        if p.name:
            md["filename"] = p.name
            stats["xlsx_infer:filename"] += 1

    obj["metadata"] = md
    return obj


def process_file(kind: str, inp: Path, out: Path) -> Dict[str, int]:
    stats = Counter()
    rows_out = []

    for obj in read_jsonl(inp):
        stats["read"] += 1
        fixed = ensure_schema(obj, kind, stats)
        if not fixed:  # dropped
            stats["dropped"] += 1
            continue
        if fixed.get("text", "") == "":
            stats["drop_empty_text"] += 1
            continue
        rows_out.append(fixed)

    n = write_jsonl(out, rows_out)
    stats["written"] = n
    return dict(stats)


def main():
    # sanity
    for k, p in INPUTS.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing input: {p}")

    report = {}
    for kind in ["web", "pdf", "xlsx"]:
        print(f"\n=== Fixing {kind} ===")
        rep = process_file(kind, INPUTS[kind], OUTPUTS[kind])
        report[kind] = rep
        print(f"written={rep.get('written')}  dropped={rep.get('dropped',0)}  read={rep.get('read')}")

    # save report
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n✅ ALL FIXED")
    print(f"Report: {REPORT.resolve()}")
    for kind, rep in report.items():
        print(f"- {kind}: {rep.get('written')} docs -> {OUTPUTS[kind].name}")


if __name__ == "__main__":
    main()
