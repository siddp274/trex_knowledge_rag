
import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse, urlunparse
from pydantic import BaseModel
import httpx
from bs4 import BeautifulSoup
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import OpenAI
from config import TREXConfig


logger = logging.getLogger("trex_mcp")

# ── Hardcoded DOM reference ────────────────────
# Agent can read this resource if the transcript tool breaks.
YOUTRANSCRIPTS_DOM = """\
URL: https://www.youtranscripts.com/transcript/{VIDEO_ID}/
Transcript text: .transcript-container p  (call .innerText)
Video title:     h1  (prefixed with "Transcript of")
URL input:       input[placeholder*="YouTube"]
Buttons:         "Generate Transcript", "Copy Transcript", "Download Transcript"
"""

# ── Helpers ──────────────────────────────────────────────────────────
def _extract_video_id(url: str) -> str:
    m = re.search(r"(?:v=|/v/|youtu\.be/|/embed/)([a-zA-Z0-9_-]{11})", url)
    if m:
        return m.group(1)
    if re.fullmatch(r"[a-zA-Z0-9_-]{11}", url):
        return url
    raise ValueError(f"Cannot extract video ID from: {url}")

def _translate_to_english(text: str, config: TREXConfig) -> str:
    """Chunk text and translate each chunk to English via OpenAI."""
    client = OpenAI(base_url=config.openai_indexer_endpoint, api_key=config.openai_indexer_api_key)
    CHUNK_CHARS = 12000 

    chunks = [text[i:i + CHUNK_CHARS] for i in range(0, len(text), CHUNK_CHARS)]
    translated = []

    for i, chunk in enumerate(chunks):
        logger.info("Translating chunk %d/%d (%d chars)", i + 1, len(chunks), len(chunk))
        resp = client.chat.completions.create(
            model=config.indexing_model,
            messages=[
                {"role": "system", "content": (
                    "You are a translator. Translate the following text to English. "
                    "Preserve all meaning, names, numbers, and structure. "
                    "Output ONLY the translation, nothing else. There may be English mixed with other languages so keep in mind while translating."
                )},
                {"role": "user", "content": chunk},
            ],
            temperature=0,
        )
        translated.append(resp.choices[0].message.content.strip())

    return "\n".join(translated)

async def _fetch_transcript_playwright(video_id: str) -> dict:
    """Use Playwright directly (not MCP) to scrape youtranscripts.com."""
    from playwright.async_api import async_playwright

    url = f"https://www.youtranscripts.com/transcript/{video_id}/"
    logger.info("[_fetch_transcript_playwright] Fetching transcript for video_id=%s", video_id)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded")
            await page.wait_for_selector(".transcript-container p", timeout=30000)
            result = await page.evaluate("""() => {
                const p = document.querySelector('.transcript-container p');
                const h1 = document.querySelector('h1');
                const transcript = p ? p.innerText : '';
                const title = (h1?.innerText || '')
                    .replace(/^Transcript of\\s*[""]?/, '')
                    .replace(/[""]$/, '');
                return { transcript, title, char_count: transcript.length, success: transcript.length > 50 };
            }""")
            logger.info("[_fetch_transcript_playwright] title=%r chars=%d success=%s",
                        result.get("title"), result.get("char_count", 0), result.get("success"))
            return result
        finally:
            await browser.close()


BASE_STORAGE = "/Users/siddp278/Desktop/projects/trex/data/scraper_ingestion_storage"
TREES_DIR = Path(BASE_STORAGE) / "trees"
PAGES_DIR = Path(BASE_STORAGE) / "pages"
MANIFESTS_DIR = Path(BASE_STORAGE) / "manifests"

TREES_DIR.mkdir(parents=True, exist_ok=True)
PAGES_DIR.mkdir(parents=True, exist_ok=True)
MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

SKIP_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp",
    ".pdf", ".zip", ".tar", ".gz", ".css", ".js",
    ".ico", ".woff", ".woff2", ".mp4", ".mp3"
}

BAD_SEGMENTS = {
    "login", "signup", "privacy", "terms", "cookie",
    "careers", "pricing", "billing", "about", "contact",
}

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/145.0.0.0 Safari/537.36"
)

class PageInput(BaseModel):
    project_name: str
    url: str

def slugify(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()

def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", "", ""))

def is_doc_link(url: str, base_domain: str) -> bool:
    try:
        parsed = urlparse(url)
        if parsed.netloc and parsed.netloc != base_domain:
            return False
        path = parsed.path.lower()
        if any(path.endswith(ext) for ext in SKIP_EXTENSIONS):
            return False
        if any(seg in path for seg in BAD_SEGMENTS):
            return False
        return True
    except Exception:
        return False

async def fetch_html(url: str) -> Optional[str]:
    logger.info("[fetch_html] Fetching: %s", url)
    try:
        async with httpx.AsyncClient(
            follow_redirects=True, timeout=30, verify=False,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
            logger.info("[fetch_html] OK %s (%d bytes)", url, len(response.text))
            return response.text
    except Exception as e:
        logger.warning("[fetch_html] Failed to fetch %s: %s", url, e)
        return None

def extract_clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "iframe", "footer", "nav"]):
        tag.decompose()

    title = soup.title.get_text(strip=True) if soup.title else ""
    main = (
        soup.find("main") or soup.find("article")
        or soup.find(attrs={"role": "main"}) or soup.body
    )

    paragraphs, code_blocks = [], []
    if main:
        for p in main.find_all("p"):
            text = p.get_text(" ", strip=True)
            if len(text) > 40:
                paragraphs.append(text)
        for code in main.find_all(["pre", "code"]):
            text = code.get_text("\n", strip=True)
            if len(text) > 20:
                code_blocks.append(text)

    logger.debug("[extract_clean_text] title=%r paragraphs=%d code_blocks=%d", title, len(paragraphs), len(code_blocks))
    return "\n\n".join(filter(None, [
        f"# {title}",
        "\n".join(paragraphs),
        "\n\n".join(code_blocks),
    ]))

def extract_links(html: str, current_url: str, base_domain: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    discovered = set()
    for tag in soup.find_all("a", href=True):
        href = tag["href"]
        if href.startswith("#"):
            continue
        absolute = normalize_url(urljoin(current_url, href))
        if is_doc_link(absolute, base_domain):
            discovered.add(absolute)
    links = sorted(discovered)
    logger.info("[extract_links] %s → %d doc links discovered", current_url, len(links))
    return links

async def fetch_accessibility_tree(url: str) -> Dict:
    logger.info("[fetch_accessibility_tree] Launching Playwright for: %s", url)
    server_params = StdioServerParameters(command="npx", args=["-y", "@playwright/mcp@latest"])
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            logger.debug("[fetch_accessibility_tree] Navigating to: %s", url)
            await session.call_tool("browser_navigate", {"url": url})
            snapshot = await session.call_tool("browser_snapshot", {})
            tree_text = ""
            try:
                for item in snapshot.content:
                    if hasattr(item, "text"):
                        tree_text += item.text
            except Exception as e:
                logger.warning("[fetch_accessibility_tree] Snapshot parse error for %s: %s", url, e)
                tree_text = str(snapshot)
            logger.info("[fetch_accessibility_tree] Snapshot captured for %s (%d chars)", url, len(tree_text))
            return {"url": url, "timestamp": datetime.utcnow().isoformat(), "tree": tree_text}


def save_page(project: str, url: str, text: str) -> str:
    path = PAGES_DIR / project / f"{slugify(url)}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    logger.info("[save_page] Saved page text → %s", path)
    return str(path)

def save_tree(project: str, url: str, tree: Dict) -> str:
    path = TREES_DIR / project / f"{slugify(url)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tree, indent=2))
    logger.info("[save_tree] Saved accessibility tree → %s", path)
    return str(path)

def load_manifest(project: str) -> Dict:
    path = MANIFESTS_DIR / f"{project}.json"
    if not path.exists():
        logger.info("[load_manifest] No manifest found for '%s', returning empty", project)
        return {"project": project, "visited": [], "saved_pages": []}
    logger.debug("[load_manifest] Loaded manifest for '%s'", project)
    return json.loads(path.read_text())

def save_manifest(project: str, manifest: Dict):
    path = MANIFESTS_DIR / f"{project}.json"
    path.write_text(json.dumps(manifest, indent=2))
    logger.debug("[save_manifest] Manifest saved for '%s' (%d pages visited)", project, len(manifest.get("visited", [])))

