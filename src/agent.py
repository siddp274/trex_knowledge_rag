"""
TREX Knowledge Graph — LangGraph Agent
========================================
A conversational agent that uses the TREX MCP server tools to answer
questions grounded in the indexed knowledge base.

Uses:
  - langgraph `create_react_agent` for the ReAct loop
  - `MemorySaver` for in-memory conversation history (persists per thread)
  - `langchain_mcp_adapters` to bridge MCP tools into LangChain

Usage:
    python agent.py                          # interactive CLI
    python agent.py --mcp-http http://...    # connect to remote MCP server
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import uuid
from typing import Optional
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
)
logger = logging.getLogger("trex_agent")

load_dotenv()

SYSTEM_PROMPT = """\
You are an expert research assistant with access to a GraphRAG Entity Knowledge Graph — \
a hybrid TREX vector + graph index built from different sources of truth documents.

You have the following tools:

**Querying the knowledge base:**
• **trex_retrieve** — Retrieve raw passages without generating an answer. \
  Use when you need to examine source material yourself or compare passages.
• **trex_status** — Check the knowledge base status (collection size, graph \
  layer active, etc.).
  entity_neo4j_status — Check the Neo4j graph status.

**Web & transcript tools and resources:**
• **info://youtranscripts_dom** - a reference resource about youtranscripts.com, a freference to call the tool trex_get_youtube_transcript \
• **trex_get_youtube_transcript** — Fetch a YouTube video's transcript via \
  youtranscripts.com and save it as a .txt file. Pass a YouTube URL or video ID. \
  Returns the saved file path.

**Indexing pipeline:**
• **trex_run_pipeline** — Run the full TREX indexing pipeline on source files \
  (ingest → embed → optional graph extraction → RAPTOR tree → Qdrant index). \
  Use this after saving transcripts to build a searchable knowledge base from them.

Typical workflow for adding new video content (if user later wants to add it to the knowledge base (info vector db + entity knowledge db):
1. Use trex_status and entity_neo4j_status to check current knowledge base status and graph status.
2. Use trex_get_youtube_transcript to fetch and save transcript(s). Before starting always ask the user if the video is in English or other launguage.
3. Use trex_run_pipeline with the saved file path(s) as sources. Ask the user if they want to create the entity graph or not. And ask if user
wants to providde their own collection name for trex or use the default one.
4. Use trex_retrieve to answer questions about the content.

Guidelines:
1. Always ground your answers in the retrieved knowledge. If the knowledge \
   base doesn't contain relevant information, say so clearly.
2. When the user asks for comparison or analysis, use trex_retrieve to pull \
   passages and reason over them yourself.
3. Cite sources when possible (document title, section).
4. If the user's question is ambiguous, ask for clarification before querying.
5. Maintain conversation context — refer back to previous answers when relevant.
6. You can't run more than 5 concurrent tool calls, so use them judiciously. Always prefer retrieval over 
running the full pipeline, unless the user explicitly wants to add new content.
"""

async def build_agent(
    trex_mcp_server_script: str = "trex_mcp_server.py",
    scraper_mcp_server_script: str = "scraper_server.py",
    mcp_http_url: Optional[str] = None,
    model_name: str = "gpt-4o",
    server_config_override: Optional[dict] = None,
):
    """
    Build and return the LangGraph agent with MCP tools loaded.

    Args:
        trex_mcp_server_script: Path to the TREX MCP server script (stdio transport).
        scraper_mcp_server_script: Path to the scraper MCP server script (stdio transport).
        mcp_http_url: URL of a running MCP HTTP server (overrides script).
        model_name: OpenAI model for the agent.
        server_config_override: Full server config dict for MultiServerMCPClient.
            When provided, mcp_server_script and mcp_http_url are ignored.
            Used by api.py to wire both the trex and scraper MCP servers.

    Returns:
        (agent, mcp_client, memory)
    """
    if server_config_override:
        server_config = server_config_override
    elif mcp_http_url:
        server_config = {
            "trex": {
                "url": mcp_http_url,
                "transport": "http",
            }
        }
    else:
        server_config = {
            "trex": {
                "command": os.getenv("MCP_SCRIPT"),
                "args": [trex_mcp_server_script],
                "transport": "stdio",
            },
            "scraper": {
                "command": os.getenv("MCP_SCRIPT"),
                "args": [scraper_mcp_server_script],
                "transport": "stdio",
            }
        }

    mcp_client = MultiServerMCPClient(server_config)
    tools = await mcp_client.get_tools()

    logger.info("Loaded %d MCP tools: %s", len(tools), [t.name for t in tools])

    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_ENDPOINT"),
    )
    memory = MemorySaver()

    agent = create_agent(
        model=llm,
        tools=tools,
        checkpointer=memory,
        system_prompt=SYSTEM_PROMPT,
    )

    return agent, mcp_client, memory


async def chat_loop(
    trex_mcp_server_script: str = "trex_mcp_server.py",
    scraper_mcp_server_script: str = "scraper_server.py",
    mcp_http_url: Optional[str] = None,
    model_name: str = "gpt-4o",
):
    """Run an interactive terminal chat with the agent."""

    agent, mcp_client, memory = await build_agent(
        trex_mcp_server_script=trex_mcp_server_script,
        scraper_mcp_server_script=scraper_mcp_server_script,
        mcp_http_url=mcp_http_url,
        model_name=model_name,
    )

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║   TREX Knowledge Graph Agent                        ║")
    print("║   Type your question, or 'quit' / 'exit' to leave.  ║")
    print("║   Conversation memory persists within this session.  ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    try:
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue
            if user_input.lower() in {"quit", "exit", "q"}:
                print("Goodbye!")
                break

            print("\nAgent: ", end="", flush=True)
            async for event in agent.astream_events(
                {"messages": [{"role": "user", "content": user_input}]},
                config=config,
                version="v2",
            ):
                kind = event.get("event")
                if kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        print(chunk.content, end="", flush=True)
                elif kind == "on_tool_start":
                    tool_name = event.get("name", "unknown")
                    print(f"\n  [calling {tool_name}…]", flush=True)
                elif kind == "on_tool_end":
                    tool_name = event.get("name", "unknown")
                    print(f"  [{tool_name} done]", flush=True)

            print("\n")

    finally:
        if hasattr(mcp_client, "close"):
            await mcp_client.close()
        logger.info("MCP client closed.")


def main():
    parser = argparse.ArgumentParser(description="TREX Knowledge Graph Agent")
    parser.add_argument(
        "--mcp-script",
        default="src/trex_mcp_server.py",
        help="Path to the MCP server script (stdio transport)",
    )
    parser.add_argument(
        "--mcp-http",
        default=None,
        help="URL of a running MCP HTTP server (overrides --mcp-script)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="OpenAI model for the agent (default: gpt-4o)",
    )
    args = parser.parse_args()

    asyncio.run(
        chat_loop(
            mcp_server_script=args.mcp_script,
            mcp_http_url=args.mcp_http,
            model_name=args.model,
        )
    )

if __name__ == "__main__":
    main()
