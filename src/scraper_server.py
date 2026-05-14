#!/usr/bin/env python3
"""
docs_scraper_mcp.py

Generic MCP documentation scraper server.

Features:
- Accessibility tree extraction via Playwright MCP
- Documentation link discovery
- Recursive crawling
- TXT persistence
- Accessibility tree persistence
- MCP resources for saved docs
- BFS traversal
- Domain-restricted crawling

Requirements:
pip install fastmcp beautifulsoup4 httpx pydantic
npm install -g @playwright/mcp

Run:
python docs_scraper_mcp.py
"""

import asyncio
import hashlib
import json
import re
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse, urlunparse

import httpx
from bs4 import BeautifulSoup
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field, field_validator

# =============================================================================
# MCP SERVER
# =============================================================================

mcp = FastMCP("docs_scraper_mcp")

# =============================================================================
# STORAGE
# =============================================================================

BASE_STORAGE = Path("./ingestion/storage")
TREES_DIR = BASE_STORAGE / "trees"
PAGES_DIR = BASE_STORAGE / "pages"
MANIFESTS_DIR = BASE_STORAGE / "manifests"

TREES_DIR.mkdir(parents=True, exist_ok=True)
PAGES_DIR.mkdir(parents=True, exist_ok=True)
MANIFESTS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# CONSTANTS
# =============================================================================

SKIP_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp",
    ".pdf", ".zip", ".tar", ".gz", ".css", ".js",
    ".ico", ".woff", ".woff2", ".mp4", ".mp3"
}

BAD_SEGMENTS = {
    "login",
    "signup",
    "privacy",
    "terms",
    "cookie",
    "careers",
    "pricing",
    "billing",
    "about",
    "contact",
}

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/145.0.0.0 Safari/537.36"
)

# =============================================================================
# HELPERS
# =============================================================================

def slugify(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()

def normalize_url(url: str) -> str:
    parsed = urlparse(url)

    normalized = urlunparse((
        parsed.scheme,
        parsed.netloc,
        parsed.path.rstrip("/"),
        "",
        "",
        ""
    ))

    return normalized

def is_same_domain(url: str, base_domain: str) -> bool:
    parsed = urlparse(url)
    return parsed.netloc == base_domain

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
    try:
        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=30,
            verify=False,
            headers={"User-Agent": USER_AGENT},
        ) as client:

            response = await client.get(url)
            response.raise_for_status()

            return response.text

    except Exception:
        return None

def extract_clean_text(html: str) -> Dict:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup([
        "script",
        "style",
        "noscript",
        "iframe",
        "footer",
        "nav"
    ]):
        tag.decompose()

    title = soup.title.get_text(strip=True) if soup.title else ""

    main = (
        soup.find("main")
        or soup.find("article")
        or soup.find(attrs={"role": "main"})
        or soup.body
    )

    headings = []
    paragraphs = []
    code_blocks = []

    if main:
        for h in main.find_all(["h1", "h2", "h3", "h4"]):
            text = h.get_text(" ", strip=True)
            if text:
                headings.append({
                    "level": h.name,
                    "text": text
                })

        for p in main.find_all("p"):
            text = p.get_text(" ", strip=True)

            if len(text) > 40:
                paragraphs.append(text)

        for code in main.find_all(["pre", "code"]):
            text = code.get_text("\n", strip=True)

            if len(text) > 20:
                code_blocks.append(text)

    combined_text = "\n\n".join([
        f"# {title}",
        "\n".join([p for p in paragraphs]),
        "\n\n".join(code_blocks)
    ])

    return {
        "title": title,
        "headings": headings,
        "paragraphs": paragraphs,
        "code_blocks": code_blocks,
        "text": combined_text,
    }

def extract_links_from_html(
    html: str,
    current_url: str,
    base_domain: str
) -> List[str]:

    soup = BeautifulSoup(html, "html.parser")

    discovered = set()

    for tag in soup.find_all("a", href=True):

        href = tag["href"]

        if href.startswith("#"):
            continue

        absolute = urljoin(current_url, href)
        absolute = normalize_url(absolute)

        if is_doc_link(absolute, base_domain):
            discovered.add(absolute)

    return sorted(discovered)

async def fetch_accessibility_tree(url: str) -> Dict:
    """
    Uses Playwright MCP browser_snapshot.
    """

    server_params = StdioServerParameters(
        command="npx",
        args=["-y", "@playwright/mcp@latest"]
    )

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            await session.initialize()

            await session.call_tool(
                "browser_navigate",
                {"url": url}
            )

            snapshot = await session.call_tool(
                "browser_snapshot",
                {}
            )

            tree_text = ""

            try:
                for item in snapshot.content:
                    if hasattr(item, "text"):
                        tree_text += item.text

            except Exception:
                tree_text = str(snapshot)

            return {
                "url": url,
                "timestamp": datetime.utcnow().isoformat(),
                "tree": tree_text,
            }

# =============================================================================
# PERSISTENCE
# =============================================================================

def save_tree(project: str, url: str, tree: Dict) -> str:
    project_dir = TREES_DIR / project
    project_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{slugify(url)}.json"

    path = project_dir / filename

    with open(path, "w") as f:
        json.dump(tree, f, indent=2)

    return str(path)

def save_page(project: str, url: str, text: str) -> str:
    project_dir = PAGES_DIR / project
    project_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{slugify(url)}.txt"

    path = project_dir / filename

    with open(path, "w", encoding="utf-8") as f:
        f.write(text)

    return str(path)

def load_manifest(project: str) -> Dict:
    path = MANIFESTS_DIR / f"{project}.json"

    if not path.exists():
        return {
            "project": project,
            "visited": [],
            "saved_pages": [],
        }

    with open(path) as f:
        return json.load(f)

def save_manifest(project: str, manifest: Dict):
    path = MANIFESTS_DIR / f"{project}.json"

    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)

# =============================================================================
# INPUT MODELS
# =============================================================================

class InitProjectInput(BaseModel):
    project_name: str
    root_url: str

class CrawlPageInput(BaseModel):
    project_name: str
    url: str

class DiscoverLinksInput(BaseModel):
    url: str

class RecursiveCrawlInput(BaseModel):
    project_name: str
    root_url: str
    max_depth: int = Field(default=2, ge=1, le=10)
    max_pages: int = Field(default=20, ge=1, le=500)

# =============================================================================
# MCP TOOLS
# =============================================================================

@mcp.tool()
async def initialize_project(params: InitProjectInput) -> str:
    """
    Initialize a new documentation scraping project.
    """

    manifest = {
        "project": params.project_name,
        "root_url": normalize_url(params.root_url),
        "created_at": datetime.utcnow().isoformat(),
        "visited": [],
        "saved_pages": [],
    }

    save_manifest(params.project_name, manifest)

    return json.dumps({
        "success": True,
        "project": params.project_name,
        "root_url": params.root_url,
    }, indent=2)

@mcp.tool()
async def get_accessibility_tree(params: CrawlPageInput) -> str:
    """
    Fetch and save accessibility tree.
    """

    tree = await fetch_accessibility_tree(params.url)

    tree_path = save_tree(
        params.project_name,
        params.url,
        tree
    )

    return json.dumps({
        "success": True,
        "url": params.url,
        "tree_path": tree_path,
    }, indent=2)

@mcp.tool()
async def discover_links(params: DiscoverLinksInput) -> str:
    """
    Discover documentation links from a page.
    """

    html = await fetch_html(params.url)

    if not html:
        return json.dumps({
            "success": False,
            "error": "Could not fetch page"
        })

    base_domain = urlparse(params.url).netloc

    links = extract_links_from_html(
        html,
        params.url,
        base_domain
    )

    return json.dumps({
        "success": True,
        "url": params.url,
        "total_links": len(links),
        "links": links,
    }, indent=2)

@mcp.tool()
async def crawl_page(params: CrawlPageInput) -> str:
    """
    Crawl and save a documentation page.
    """

    html = await fetch_html(params.url)

    if not html:
        return json.dumps({
            "success": False,
            "error": f"Could not fetch {params.url}"
        })

    parsed = extract_clean_text(html)

    page_path = save_page(
        params.project_name,
        params.url,
        parsed["text"]
    )

    tree = await fetch_accessibility_tree(params.url)

    tree_path = save_tree(
        params.project_name,
        params.url,
        tree
    )

    manifest = load_manifest(params.project_name)

    manifest["visited"].append(params.url)

    manifest["saved_pages"].append({
        "url": params.url,
        "page_path": page_path,
        "tree_path": tree_path,
    })

    save_manifest(params.project_name, manifest)

    return json.dumps({
        "success": True,
        "url": params.url,
        "page_path": page_path,
        "tree_path": tree_path,
    }, indent=2)

@mcp.tool()
async def recursive_crawl(params: RecursiveCrawlInput) -> str:
    """
    BFS recursive crawl for documentation sites.
    """

    visited: Set[str] = set()

    queue = deque([
        (normalize_url(params.root_url), 0)
    ])

    base_domain = urlparse(params.root_url).netloc

    crawled = []

    while queue and len(visited) < params.max_pages:

        current_url, depth = queue.popleft()

        if current_url in visited:
            continue

        if depth > params.max_depth:
            continue

        visited.add(current_url)

        print(f"[CRAWLING] {current_url}")

        try:
            html = await fetch_html(current_url)

            if not html:
                continue

            parsed = extract_clean_text(html)

            page_path = save_page(
                params.project_name,
                current_url,
                parsed["text"]
            )

            tree = await fetch_accessibility_tree(current_url)

            tree_path = save_tree(
                params.project_name,
                current_url,
                tree
            )

            crawled.append({
                "url": current_url,
                "page_path": page_path,
                "tree_path": tree_path,
                "depth": depth,
            })

            links = extract_links_from_html(
                html,
                current_url,
                base_domain
            )

            for link in links:

                if link not in visited:
                    queue.append((link, depth + 1))

            await asyncio.sleep(1.5)

        except Exception as e:
            print(f"[ERROR] {current_url} -> {e}")

    manifest = load_manifest(params.project_name)

    manifest["visited"] = list(visited)
    manifest["saved_pages"] = crawled

    save_manifest(params.project_name, manifest)

    return json.dumps({
        "success": True,
        "total_crawled": len(crawled),
        "pages": crawled,
    }, indent=2)

# =============================================================================
# MCP RESOURCES
# =============================================================================

@mcp.resource("docs://manifest/{project}")
def get_manifest(project: str) -> str:

    manifest_path = MANIFESTS_DIR / f"{project}.json"

    if not manifest_path.exists():
        return "Manifest not found"

    return manifest_path.read_text()

@mcp.resource("docs://page/{project}/{page_hash}")
def get_page(project: str, page_hash: str) -> str:

    page_path = PAGES_DIR / project / f"{page_hash}.txt"

    if not page_path.exists():
        return "Page not found"

    return page_path.read_text()

@mcp.resource("docs://tree/{project}/{page_hash}")
def get_tree(project: str, page_hash: str) -> str:

    tree_path = TREES_DIR / project / f"{page_hash}.json"

    if not tree_path.exists():
        return "Tree not found"

    return tree_path.read_text()

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    mcp.run()