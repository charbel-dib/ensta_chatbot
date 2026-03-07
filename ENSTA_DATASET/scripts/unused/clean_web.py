from __future__ import annotations
import json, re, hashlib
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INP = PROJECT_ROOT / "01_raw" / "web_pages.jsonl"
OUT = PROJECT_ROOT / "02_clean" / "web_clean.jsonl"

ALLOWED_HOSTS = {
    "www.ensta.fr",
    "international-admission.ensta.fr",
    # "parcours-talents.ensta.fr",  # (souvent inutile: login/register)
}

DROP_PATH_CONTAINS = [
    "login", "register", "password", "cgi", "user", "account"
]

NAV_LINE_RE = re.compile(r"^\s*(\*|-)\s*\[[^\]]{1,50}\]\([^)]+\)\s*$")
IMG_LINE_RE = re.compile(r"^\s*!\[[^\]]*\]\([^)]+\)\s*$")
COOKIE_RE = re.compile(r"\b(cookie|cookies|rgpd|privacy|confidentialit|consent)\b", re.I)

def read_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def norm_text(t: str) -> str:
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    # supprime lignes nav / images / cookie banners courtes
    kept = []
    for ln in t.split("\n"):
        s = ln.strip()
        if not s:
            kept.append("")
            continue
        if IMG_LINE_RE.match(s):
            continue
        if NAV_LINE_RE.match(s):
            continue
        if COOKIE_RE.search(s) and len(s) < 140:
            continue
        # lignes "trop linkées" = menu
        if s.count("](") >= 2 and len(s) < 140:
            continue
        kept.append(ln)
    out = "\n".join(kept)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    return out

def dedup_hash(t: str) -> str:
    x = re.sub(r"\s+", " ", t.strip().lower())
    return hashlib.md5(x.encode("utf-8", errors="ignore")).hexdigest()

def main():
    if not INP.exists():
        raise FileNotFoundError(INP)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        OUT.unlink()

    seen_hashes = set()
    kept = 0
    dropped = {
        "host": 0,
        "empty": 0,
        "path": 0,
        "dup": 0,
    }

    for obj in read_jsonl(INP):
        url = obj.get("source") or ""
        p = urlparse(url)
        host = p.netloc.lower()

        if host not in ALLOWED_HOSTS:
            dropped["host"] += 1
            continue

        path_l = (p.path or "").lower()
        if any(x in path_l for x in DROP_PATH_CONTAINS):
            dropped["path"] += 1
            continue

        text = (obj.get("text") or "").strip()
        text = norm_text(text)

        if len(text) < 400:
            dropped["empty"] += 1
            continue

        h = dedup_hash(text[:6000])
        if h in seen_hashes:
            dropped["dup"] += 1
            continue
        seen_hashes.add(h)

        out_obj = {
            "id": obj.get("id"),
            "source_type": "web",
            "source": url,
            "title": obj.get("title") or url,
            "text": text,
            "metadata": obj.get("metadata") or {}
        }

        with OUT.open("a", encoding="utf-8") as f:
            f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
        kept += 1

        if kept % 100 == 0:
            print(f"[OK] kept={kept}")

    print("\n✅ WEB CLEAN DONE")
    print(f"Output: {OUT.resolve()}")
    print(f"Kept  : {kept}")
    print("Dropped:", dropped)

if __name__ == "__main__":
    main()
