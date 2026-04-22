"""
FastAPI server that wraps the TREX LangGraph agent.
Streams responses via SSE (Server-Sent Events).

Usage:
    uvicorn server:app --reload --port 8080
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import json
import logging
import os
import sys
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s — %(message)s")
logger = logging.getLogger("trex_server")

# ── System prompt (same as agent.py) ────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert research assistant with access to a GraphRAG Entity Knowledge Graph — \
a hybrid TREX vector + graph index built from different sources of truth documents.

You have the following tools:

**Querying the knowledge base:**
• **trex_retrieve** — Retrieve raw passages without generating an answer. \
  Use when you need to examine source material yourself or compare passages.
• **trex_status** — Check the knowledge base status (collection size, graph \
  layer active, etc.).
• **entity_neo4j_status** — Check the Neo4j graph status.

**Web & transcript tools and resources:**
• **info://youtranscripts_dom** - a reference resource about youtranscripts.com
• **trex_get_youtube_transcript** — Fetch a YouTube video's transcript via \
  youtranscripts.com and save it as a .txt file.

**Indexing pipeline:**
• **trex_run_pipeline** — Run the full TREX indexing pipeline on source files \
  (ingest → embed → optional graph extraction → RAPTOR tree → Qdrant index).

Typical workflow for adding new video content:
1. Use trex_status and entity_neo4j_status to check current knowledge base status.
2. Use trex_get_youtube_transcript to fetch and save transcript(s). Ask if the video is in English first.
3. Use trex_run_pipeline with the saved file path(s) as sources.
4. Use trex_retrieve to answer questions about the content.

Guidelines:
1. Always ground your answers in the retrieved knowledge.
2. Use trex_retrieve for comparison or analysis.
3. Cite sources when possible.
4. If the user's question is ambiguous, ask for clarification.
5. Maintain conversation context.
6. Max 5 concurrent tool calls.
"""

# ── Globals (initialized on startup) ────────────────────────────────

agent = None
mcp_client = None
memory = None

# ── FastAPI app ─────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent, mcp_client, memory

    mcp_script = os.getenv("MCP_SCRIPT", "src/trex_mcp_server.py")
    model_name = os.getenv("AGENT_MODEL", "gpt-4o")

    server_config = {
        "trex": {
            "command": "/Users/siddp278/Desktop/projects/trex_knowledge_rag/trex_knowledge_rag/venv/bin/python",
            "args": [mcp_script],
            "transport": "stdio",
        }
    }

    mcp_client = MultiServerMCPClient(server_config)
    tools = await mcp_client.get_tools()
    logger.info("Loaded %d MCP tools: %s", len(tools), [t.name for t in tools])

    llm = ChatOpenAI(
        model=model_name,
        temperature=0,
        api_key=os.getenv("OPENAI_QUERY_API_KEY"),
        base_url=os.getenv("OPENAI_ENDPOINT"),
    )

    memory = MemorySaver()

    agent = create_agent(
        model=llm,
        tools=tools,
        checkpointer=memory,
        system_prompt=SYSTEM_PROMPT,
    )

    logger.info("Agent ready.")

    yield {}

    if mcp_client and hasattr(mcp_client, "close"):
        await mcp_client.close()


app = FastAPI(title="TREX Agent API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Thread-Id"]
)


async def _stream_agent(message: str, thread_id: str):
    """Yield SSE events from the agent's stream."""
    config = {"configurable": {"thread_id": thread_id}}

    async for event in agent.astream_events(
        {"messages": [{"role": "user", "content": message}]},
        config=config,
        version="v2",
    ):
        kind = event.get("event")

        if kind == "on_chat_model_stream":
            chunk = event.get("data", {}).get("chunk")
            if chunk and hasattr(chunk, "content") and chunk.content:
                yield f"data: {json.dumps({'type': 'token', 'content': chunk.content})}\n\n"

        elif kind == "on_tool_start":
            name = event.get("name", "unknown")
            yield f"data: {json.dumps({'type': 'tool_start', 'tool': name})}\n\n"

        elif kind == "on_tool_end":
            name = event.get("name", "unknown")
            yield f"data: {json.dumps({'type': 'tool_end', 'tool': name})}\n\n"

    yield f"data: {json.dumps({'type': 'done'})}\n\n"


@app.post("/chat")
async def chat(req: ChatRequest):
    thread_id = req.thread_id or str(uuid.uuid4())

    return StreamingResponse(
        _stream_agent(req.message, thread_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Thread-Id": thread_id,
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok", "agent_ready": agent is not None}