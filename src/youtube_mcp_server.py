"""
YouTube Transcript — MCP Server

Tools:
  youtube_get_transcript — Fetch a YouTube transcript from youtranscripts.com
                           and optionally save it as a .txt file.

Resources:
  info://youtranscripts_dom — Hardcoded DOM reference for youtranscripts.com.
                              Read before calling youtube_get_transcript.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from mcp.server.fastmcp import FastMCP

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import TREXConfig
from src.utils.mcp_helper import (
    _extract_video_id,
    _translate_to_english,
    _fetch_transcript_playwright,
    YOUTRANSCRIPTS_DOM,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s — %(message)s")
logger = logging.getLogger("youtube_mcp")

TRANSCRIPT_DIR = "/Users/siddp278/Desktop/projects/trex/data/transcripts"

mcp = FastMCP("youtube_mcp")


@mcp.resource("info://youtranscripts_dom")
def youtranscripts_dom() -> str:
    """Hardcoded DOM reference for youtranscripts.com. Read before calling youtube_get_transcript."""
    return YOUTRANSCRIPTS_DOM


@mcp.tool()
async def youtube_get_transcript(
    youtube_url: str,
    save_dir: str = TRANSCRIPT_DIR,
    download: bool = True,
    not_english_language: bool = False,
) -> str:
    """
    Fetch a YouTube transcript from youtranscripts.com and optionally save as .txt.

    Args:
        youtube_url:          YouTube video URL or bare video ID.
        save_dir:             Directory to save the transcript file.
        download:             If True, save the transcript as a .txt file.
        not_english_language: If True, translate the transcript to English before saving.

    Returns:
        JSON with video_id, title, and file_path (if downloaded).
    """
    try:
        video_id = _extract_video_id(youtube_url)
        logger.info("[youtube_get_transcript] video_id=%s download=%s translate=%s", video_id, download, not_english_language)

        extraction = await _fetch_transcript_playwright(video_id)

        if not extraction or not extraction.get("success"):
            logger.warning("[youtube_get_transcript] Failed to extract transcript for video_id=%s", video_id)
            return json.dumps({"error": "Could not extract transcript", "video_id": video_id})

        transcript = extraction["transcript"]
        title = extraction.get("title", video_id)
        logger.info("[youtube_get_transcript] Extracted transcript: title=%r chars=%d", title, len(transcript))

        config = TREXConfig()
        if not_english_language:
            logger.info("[youtube_get_transcript] Translating to English")
            transcript = _translate_to_english(transcript, config)

        file_path = None
        if download:
            os.makedirs(save_dir, exist_ok=True)
            safe = re.sub(r'[^\w\s-]', '', title)[:80].strip().replace(' ', '_')
            fname = f"transcript_{video_id}_{safe}.txt" if safe else f"transcript_{video_id}.txt"
            file_path = os.path.join(save_dir, fname)
            Path(file_path).write_text(transcript, encoding="utf-8")
            logger.info("[youtube_get_transcript] Saved transcript → %s", file_path)

        result = {"video_id": video_id, "title": title}
        if file_path:
            result["file_path"] = file_path
        return json.dumps(result, indent=2)

    except Exception as e:
        logger.exception("[youtube_get_transcript] failed")
        return json.dumps({"error": str(e)})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--http", action="store_true")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    if args.http:
        mcp.run(transport="streamable_http", port=args.port)
    else:
        mcp.run()
