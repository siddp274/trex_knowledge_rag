"""
Documentation scraper MCP server.

Workflow:
1. Call `scrape_landing_page(project_name, url)` — fetches the landing page,
   saves its text and accessibility tree, and returns all discovered sub-section links.
2. Call `scrape_page(project_name, url)` for each sub-section link — fetches,
   saves text and accessibility tree for that page.

Resources:
- docs://manifest/{project}         — list of all visited URLs and saved file paths
- docs://page/{project}/{page_hash} — plain text for a saved page
- docs://tree/{project}/{page_hash} — accessibility tree JSON for a saved page

Storage layout:
  ./ingestion/storage/
    pages/{project}/{md5_of_url}.txt
    trees/{project}/{md5_of_url}.json
    manifests/{project}.json

Requirements:
  pip install fastmcp beautifulsoup4 httpx pydantic
  npm install -g @playwright/mcp
"""
import logging
import os
import sys
from urllib.parse import urljoin, urlparse, urlunparse
import json
from datetime import datetime
from mcp.server.fastmcp import FastMCP

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from src.utils.mcp_helper import (fetch_html, extract_clean_text, normalize_url, 
                                  is_doc_link, slugify, PageInput, save_page, save_tree, 
                                  load_manifest, save_manifest, fetch_accessibility_tree,
                                  extract_links, MANIFESTS_DIR, PAGES_DIR, TREES_DIR)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s — %(message)s")
logger = logging.getLogger("trex_mcp")

mcp = FastMCP("docs_scraper_mcp")

# =============================================================================
# MCP TOOLS
# =============================================================================

@mcp.tool()
async def scrape_landing_page(params: PageInput) -> str:
    """
    Scrape the landing/main documentation page.

    - Fetches HTML and saves clean text to disk
    - Fetches and saves the Playwright accessibility tree
    - Discovers and returns all sub-section links on the page
    - Initialises the project manifest

    Call this first with the root documentation URL, then call
    scrape_page() for each link returned in `sub_section_links`.
    """
    url = normalize_url(params.url)
    base_domain = urlparse(url).netloc

    html = await fetch_html(url)
    if not html:
        return json.dumps({"success": False, "error": f"Could not fetch {url}"})

    text = extract_clean_text(html)
    page_path = save_page(params.project_name, url, text)

    tree = await fetch_accessibility_tree(url)
    tree_path = save_tree(params.project_name, url, tree)

    links = extract_links(html, url, base_domain)

    manifest = {
        "project": params.project_name,
        "root_url": url,
        "created_at": datetime.utcnow().isoformat(),
        "visited": [url],
        "saved_pages": [{"url": url, "page_path": page_path, "tree_path": tree_path}],
    }
    save_manifest(params.project_name, manifest)

    logger.info(f"[scrape_landing_page] {url} → {len(links)} sub-section links found")

    return json.dumps({
        "success": True,
        "url": url,
        "page_path": page_path,
        "tree_path": tree_path,
        "sub_section_links": links,
        "total_links": len(links),
    }, indent=2)


@mcp.tool()
async def scrape_page(params: PageInput) -> str:
    """
    Scrape a single sub-section documentation page.

    - Fetches HTML and saves clean text to disk
    - Fetches and saves the Playwright accessibility tree
    - Updates the project manifest

    Call this for each link returned by scrape_landing_page().
    """
    url = normalize_url(params.url)

    html = await fetch_html(url)
    if not html:
        return json.dumps({"success": False, "error": f"Could not fetch {url}"})

    text = extract_clean_text(html)
    page_path = save_page(params.project_name, url, text)

    tree = await fetch_accessibility_tree(url)
    tree_path = save_tree(params.project_name, url, tree)

    manifest = load_manifest(params.project_name)
    if url not in manifest["visited"]:
        manifest["visited"].append(url)
        manifest["saved_pages"].append({"url": url, "page_path": page_path, "tree_path": tree_path})
        save_manifest(params.project_name, manifest)
        logger.info("[scrape_page] %s → saved", url)
    else:
        logger.info("[scrape_page] %s already visited, skipping manifest update", url)

    return json.dumps({
        "success": True,
        "url": url,
        "page_path": page_path,
        "tree_path": tree_path,
    }, indent=2)


@mcp.resource("docs://manifest/{project}")
def get_manifest(project: str) -> str:
    path = MANIFESTS_DIR / f"{project}.json"
    return path.read_text() if path.exists() else "Manifest not found"

@mcp.resource("docs://page/{project}/{page_hash}")
def get_page(project: str, page_hash: str) -> str:
    path = PAGES_DIR / project / f"{page_hash}.txt"
    return path.read_text() if path.exists() else "Page not found"

@mcp.resource("docs://tree/{project}/{page_hash}")
def get_tree(project: str, page_hash: str) -> str:
    path = TREES_DIR / project / f"{page_hash}.json"
    return path.read_text() if path.exists() else "Tree not found"


if __name__ == "__main__":
    mcp.run()
