from __future__ import annotations
import json, re, hashlib
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

# ---- PATHS (adapte si besoin)
PROJECT_ROOT = Path(__file__).resolve().parents[1]   # ENSTA_DATASET/
WEB_JSONL = PROJECT_ROOT / "01_raw" / "web_pages.jsonl"

# ---- Heuristiques bruit
COOKIE_RE = re.compile(r"\b(cookie|cookies|rgpd|privacy|confidentialit|consent)\b", re.I)
LOGIN_RE  = re.compile(r"\b(login|connexion|sign in|auth)\b", re.I)
NAV_LINE_RE = re.compile(r"^\s*(\*|-)\s*\[[^\]]{1,50}\]\([^)]+\)\s*$")  # bullet link
IMG_LINE_RE = re.compile(r"^\s*!\[[^\]]*\]\([^)]+\)\s*$")

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def short_hash(text: str) -> str:
    norm = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.md5(norm.encode("utf-8", errors="ignore")).hexdigest()

def metrics(text: str) -> dict:
    t = text or ""
    lines = t.splitlines()
    n_chars = len(t)
    n_lines = len(lines)
    n_links_md = t.count("](")  # markdown link pattern
    n_img = sum(1 for ln in lines if IMG_LINE_RE.match(ln.strip()))
    nav_lines = sum(1 for ln in lines if NAV_LINE_RE.match(ln.strip()))
    alpha = sum(ch.isalpha() for ch in t)
    alpha_ratio = alpha / max(1, n_chars)
    link_density = n_links_md / max(1, n_lines)

    flags = {
        "cookie": bool(COOKIE_RE.search(t)),
        "login": bool(LOGIN_RE.search(t)),
    }
    return {
        "chars": n_chars,
        "lines": n_lines,
        "md_links": n_links_md,
        "img_lines": n_img,
        "nav_lines": nav_lines,
        "alpha_ratio": alpha_ratio,
        "link_density": link_density,
        **flags
    }

def main():
    if not WEB_JSONL.exists():
        raise FileNotFoundError(f"Introuvable: {WEB_JSONL}")

    total = 0
    by_host = Counter()
    by_path1 = Counter()
    by_path2 = Counter()

    # distributions
    chars_hist = Counter()
    alpha_hist = Counter()
    link_hist = Counter()

    # noisy candidates
    noisy = []
    too_short = []
    high_link = []
    duplicates = defaultdict(list)  # hash -> [url...]

    for obj in read_jsonl(WEB_JSONL):
        total += 1
        url = obj.get("source") or ""
        text = obj.get("text") or ""
        p = urlparse(url)
        host = p.netloc.lower()
        by_host[host] += 1

        parts = [x for x in p.path.split("/") if x]
        if parts:
            by_path1[parts[0]] += 1
        if len(parts) >= 2:
            by_path2[f"{parts[0]}/{parts[1]}"] += 1

        m = metrics(text)

        # hist bins
        chars_hist[(m["chars"]//500)*500] += 1
        alpha_hist[int(m["alpha_ratio"]*10)] += 1
        link_hist[min(20, int(m["link_density"]))] += 1

        # heuristic flags
        if m["chars"] < 400:
            too_short.append((m["chars"], url))
        if m["link_density"] >= 6 or m["md_links"] >= 80:
            high_link.append((m["link_density"], m["md_links"], url))
        if m["nav_lines"] >= 12 and m["chars"] < 2500:
            noisy.append((m["nav_lines"], m["chars"], url))

        # dup
        h = short_hash(text[:5000])  # hash sur début (souvent suffit)
        duplicates[h].append(url)

    dup_groups = [(h, urls) for h, urls in duplicates.items() if len(urls) >= 5]
    dup_groups.sort(key=lambda x: len(x[1]), reverse=True)

    print("\n=== WEB CRAWL AUDIT ===")
    print(f"File: {WEB_JSONL}")
    print(f"Total docs: {total}")

    print("\n-- Hosts (top 20) --")
    for host, c in by_host.most_common(20):
        print(f"{c:6d}  {host}")

    print("\n-- Path level-1 (top 30) --")
    for k, c in by_path1.most_common(30):
        print(f"{c:6d}  /{k}")

    print("\n-- Size distribution (chars, bin=500) top --")
    for b, c in chars_hist.most_common(15):
        print(f"{c:6d}  ~{b} chars")

    print("\n-- Alpha ratio distribution (0..10 => 0.0..1.0) --")
    for b in range(0, 11):
        print(f"{b/10:.1f}..{(b+1)/10:.1f}: {alpha_hist.get(b,0)}")

    print("\n-- Link density (links per line bucketed) --")
    for b in range(0, 21):
        if b in link_hist:
            label = f"{b}" if b < 20 else "20+"
            print(f"{label:>3}: {link_hist[b]}")

    print("\n-- Noisy candidates (nav-heavy) sample 15 --")
    noisy.sort(reverse=True)
    for x in noisy[:15]:
        print(f"nav_lines={x[0]:3d} chars={x[1]:5d}  {x[2]}")

    print("\n-- Too short pages sample 15 --")
    too_short.sort()
    for x in too_short[:15]:
        print(f"chars={x[0]:4d}  {x[1]}")

    print("\n-- High link density sample 15 --")
    high_link.sort(reverse=True)
    for x in high_link[:15]:
        print(f"link_density={x[0]:.1f} md_links={x[1]:3d}  {x[2]}")

    print("\n-- Duplicate clusters (>=5 same-ish pages) top 10 --")
    for h, urls in dup_groups[:10]:
        print(f"{len(urls):4d} pages  hash={h}  example={urls[0]}")

    print("\n✅ Audit finished.")
    print("Next: we’ll decide filters (hosts/paths/noise rules) and build a CLEAN web corpus.")

if __name__ == "__main__":
    main()
