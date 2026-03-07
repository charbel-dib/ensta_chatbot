from __future__ import annotations

import json
import hashlib
import time
from pathlib import Path
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

import requests
from docling.document_converter import DocumentConverter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "01_raw"
OUT_WEB = RAW_DIR / "web_docling.jsonl"

ROOT_URL = "https://www.ensta.fr"
SITEMAP_URL = ROOT_URL.rstrip("/") + "/sitemap.xml"

HEADERS = {"User-Agent": "Mozilla/5.0 (ACA-ENSTA DoclingCrawler/1.0)"}
TIMEOUT = 25
SLEEP_S = 0.15

# bruit / auth
DROP_HOST_CONTAINS = ["bibnum.ensta.fr"]
DROP_PATH_CONTAINS = ["login", "register", "password", "/cgi", "account", "user"]


def stable_id(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update((p or "").encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return h.hexdigest()[:24]


def iter_sitemap_urls(sitemap_url: str):
    r = requests.get(sitemap_url, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}

    # index
    for sm in root.findall("sm:sitemap", ns):
        loc = (sm.findtext("sm:loc", default="", namespaces=ns) or "").strip()
        if loc:
            yield from iter_sitemap_urls(loc)

    # urlset
    for u in root.findall("sm:url", ns):
        loc = (u.findtext("sm:loc", default="", namespaces=ns) or "").strip()
        if loc:
            yield loc


def allowed(url: str) -> bool:
    p = urlparse(url)
    host = (p.netloc or "").lower()
    path_l = (p.path or "").lower()
    if not host.endswith("ensta.fr"):
        return False
    if any(x in host for x in DROP_HOST_CONTAINS):
        return False
    if any(x in path_l for x in DROP_PATH_CONTAINS):
        return False
    return True


def export_text(doc) -> str:
    for fn in ("export_to_markdown", "export_to_text"):
        if hasattr(doc, fn):
            return getattr(doc, fn)()
    return str(doc)


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    urls = [u for u in iter_sitemap_urls(SITEMAP_URL) if allowed(u)]
    print(f"URLs from sitemap (filtered): {len(urls)}")

    converter = DocumentConverter()
    written = 0

    with OUT_WEB.open("w", encoding="utf-8") as f:
        for i, url in enumerate(urls, 1):
            try:
                res = converter.convert(url, headers=HEADERS, raises_on_error=False)
                doc = getattr(res, "document", None)
                if doc is None:
                    continue

                text = export_text(doc)
                if not text or len(text) < 250:
                    continue

                p = urlparse(url)
                obj = {
                    "id": stable_id("web", url),
                    "source_type": "web",
                    "source": url,
                    "title": url,
                    "text": text,
                    "metadata": {
                        "host": p.netloc.lower(),
                        "path": p.path or "/",
                        "lang": "en" if (p.path or "").startswith("/en") else "fr",
                    },
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                written += 1

                if i % 100 == 0:
                    print(f"[{i}/{len(urls)}] written={written}")

            except Exception as e:
                print(f"[WARN] {url} -> {e}")

            time.sleep(SLEEP_S)

    print("DONE:", OUT_WEB)
    print("WRITTEN:", written)


if __name__ == "__main__":
    main()