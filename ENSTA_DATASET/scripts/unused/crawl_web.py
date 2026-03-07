import asyncio, json, re, hashlib
from urllib.parse import urlparse, urldefrag
from pathlib import Path
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter

ROOT_URL = "https://www.ensta.fr"
ALLOWED_SUFFIX = "ensta.fr"
OUT_JSONL = Path("ENSTA_DATASET/01_raw/web_pages.jsonl")

def stable_id(*parts):
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8", errors="ignore")); h.update(b"\0")
    return h.hexdigest()[:24]

def normalize_url(url: str) -> str:
    url, _ = urldefrag(url)
    p = urlparse(url)
    host = p.netloc.lower()
    path = p.path if p.path else "/"
    return f"{p.scheme}://{host}{path}".rstrip("/")

def allowed(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return host == ALLOWED_SUFFIX or host.endswith("." + ALLOWED_SUFFIX)

def clean_markdown(md: str) -> str:
    # nettoyage simple (on renforcera à l’étape cleaning)
    md = md.replace("\r\n", "\n").replace("\r", "\n")
    md = re.sub(r"\n{3,}", "\n\n", md).strip()
    return md

async def main():
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    if OUT_JSONL.exists():
        OUT_JSONL.unlink()

    browser_conf = BrowserConfig(headless=True)
    md_generator = DefaultMarkdownGenerator(
        content_filter=PruningContentFilter(threshold=0.35, threshold_type="fixed")
    )
    run_conf = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        markdown_generator=md_generator,
        only_text=True,
        verbose=False
    )

    # On part du root, Crawl4AI va suivre les liens internes.
    # (On fera une phase "coverage audit" plus tard pour vérifier qu’on n’oublie rien.)
    start_urls = [ROOT_URL]

    docs = 0
    async with AsyncWebCrawler(config=browser_conf) as crawler:
        # deep crawl simple: on itère manuellement via links
        seen = set()
        queue = [ROOT_URL]

        while queue:
            url = queue.pop()
            url = normalize_url(url)
            if url in seen or not allowed(url):
                continue
            seen.add(url)

            res = await crawler.arun(url, config=run_conf)
            if not res.success:
                continue

            md = res.markdown
            if not isinstance(md, str):
                md = getattr(md, "fit_markdown", "") or getattr(md, "raw_markdown", "")
            text = clean_markdown(md or "")
            if not text:
                continue

            meta = res.metadata or {}
            title = meta.get("title") or url

            obj = {
                "id": stable_id("web", url),
                "source_type": "web",
                "source": url,
                "title": title,
                "text": text,
                "metadata": meta
            }
            with OUT_JSONL.open("a", encoding="utf-8") as f:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            docs += 1

            from urllib.parse import urljoin
            from lxml import html as lxml_html

            # ...
            html = getattr(res, "html", None) or ""
            if html:
                try:
                    doc = lxml_html.fromstring(html)
                    hrefs = doc.xpath("//a/@href")
                    for href in hrefs:
                        lk = urljoin(url, href)
                        lk = normalize_url(lk)
                        if allowed(lk) and lk not in seen:
                            queue.append(lk)
                except Exception:
                    pass

            if docs % 50 == 0:
                print(f"[OK] web docs={docs}  seen_urls={len(seen)}")

    print(f"✅ Done. web docs={docs}. Output={OUT_JSONL.resolve()}")

if __name__ == "__main__":
    asyncio.run(main())
