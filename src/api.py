"""
FastAPI server that wraps the TREX LangGraph agent.
Streams responses via SSE (Server-Sent Events).
Serves frontend as static files.
Authenticates users via Microsoft Azure Entra ID (OAuth 2.0 PKCE).

Environment variables required:
    AZURE_CLIENT_ID       — App registration client ID
    AZURE_TENANT_ID       — Your Azure AD tenant ID
    AZURE_REDIRECT_URI    — e.g. http://localhost:8080/auth/callback
    MCP_SCRIPT            — Path to trex_mcp_server.py
    OPENAI_API_KEY        — OpenAI API key
    OPENAI_ENDPOINT       — OpenAI base URL
    AGENT_MODEL           — Model name (default: gpt-4o)

Usage:
    uvicorn src.api:app --reload --port 8080
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import hashlib
import hmac
import json
import logging
import os
import secrets
import sys
import time
import uuid
from base64 import urlsafe_b64encode, urlsafe_b64decode
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
print(f"Current dir: {CURRENT_DIR} and project root: {PROJECT_ROOT}")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.helper import RESPONSE_403, _error_page
from src.prompt import SYSTEM_PROMPT

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s — %(message)s")
logger = logging.getLogger("trex_backend_server")

# ── Azure / Auth config ─────────────────────────────────────────────

AZURE_CLIENT_ID   = os.getenv("AZURE_CLIENT_ID", "")
AZURE_CLIENT_SECRET = os.getenv("AZURE_CLIENT_SECRET")
AZURE_TENANT_ID   = os.getenv("AZURE_TENANT_ID", "")
AZURE_REDIRECT_URI = os.getenv("AZURE_REDIRECT_URI", "http://localhost:8080/auth/callback")
SESSION_SECRET    = os.getenv("SESSION_SECRET", secrets.token_hex(32))

AZURE_AUTH_URL    = f"https://login.microsoftonline.com/{AZURE_TENANT_ID}/oauth2/v2.0/authorize"
AZURE_TOKEN_URL   = f"https://login.microsoftonline.com/{AZURE_TENANT_ID}/oauth2/v2.0/token"
AZURE_JWKS_URL    = f"https://login.microsoftonline.com/{AZURE_TENANT_ID}/discovery/v2.0/keys"
AZURE_SCOPE       = "openid profile email User.Read"

# ── Session helpers (signed cookie, no DB needed) ───────────────────

SESSION_COOKIE = "trex_session"
SESSION_TTL    = 8 * 3600  # 8 hours


# ── Allowlist ── add any personal or work emails here ──────────────
ALLOWED_EMAILS = {
    "siddp274@gmail.com",
    "siddp278@gmail.com",
}


def _sign(payload: str) -> str:
    """HMAC-sign a string, return payload.signature."""
    sig = hmac.new(SESSION_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return f"{payload}.{sig}"


def _verify(token: str) -> Optional[str]:
    """Verify signed token; return payload or None."""
    try:
        payload, sig = token.rsplit(".", 1)
        expected = hmac.new(SESSION_SECRET.encode(), payload.encode(), hashlib.sha256).hexdigest()
        if hmac.compare_digest(sig, expected):
            return payload
    except Exception:
        pass
    return None


def create_session(user_email: str, user_name: str) -> str:
    data = json.dumps({"email": user_email, "name": user_name, "exp": int(time.time()) + SESSION_TTL})
    encoded = urlsafe_b64encode(data.encode()).decode()
    return _sign(encoded)


def parse_session(cookie: str) -> Optional[dict]:
    payload = _verify(cookie)
    if not payload:
        return None
    try:
        data = json.loads(urlsafe_b64decode(payload + "==").decode())
        if data.get("exp", 0) < time.time():
            return None
        return data
    except Exception:
        return None


def get_current_user(request: Request) -> Optional[dict]:
    cookie = request.cookies.get(SESSION_COOKIE)
    if not cookie:
        return None
    return parse_session(cookie)


def require_auth(request: Request) -> dict:
    user = get_current_user(request)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


# ── PKCE helpers ────────────────────────────────────────────────────

# In-memory store for OAuth state → code_verifier (short-lived, per-login)
_oauth_states: dict[str, tuple[str, float]] = {}

def _pkce_pair() -> tuple[str, str]:
    verifier = urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode()
    digest = hashlib.sha256(verifier.encode()).digest()
    challenge = urlsafe_b64encode(digest).rstrip(b"=").decode()
    return verifier, challenge


# ── Globals ──────────────────────────────────────────────────────────

agent = None
mcp_client = None
memory = None

# ── FastAPI app ──────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    thread_id: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent, mcp_client, memory

    model_name  = os.getenv("AGENT_MODEL", "gpt-4o")

    server_config = {
        "trex": {
            "command": os.getenv("PYTHON_SCRIPT"),
            "args": [os.getenv("TREX_SERVER_SCRIPT")],
            "transport": "stdio",
        },
        "youtube": {
            "command": os.getenv("PYTHON_SCRIPT"),
            "args": [os.getenv("YOUTUBE_SERVER_SCRIPT")],
            "transport": "stdio",
        },
        "scraper": {
            "command": os.getenv("PYTHON_SCRIPT"),
            "args": [os.getenv("SCRAPER_SERVER_SCRIPT")],
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
    agent  = create_agent(model=llm, tools=tools, checkpointer=memory, system_prompt=SYSTEM_PROMPT)
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
    expose_headers=["X-Thread-Id"],
)


# ── Static files (frontend) ──────────────────────────────────────────
# Serve everything in ./static at /static, but the root "/" is handled
# by a custom route below so we can inject auth state.

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ── Auth routes ──────────────────────────────────────────────────────

@app.get("/auth/login")
async def auth_login(request: Request):
    """Redirect to Azure Entra ID login page."""
    state = secrets.token_urlsafe(24)
    verifier, challenge = _pkce_pair()

    # Store verifier keyed by state; expire after 10 min
    _oauth_states[state] = (verifier, time.time() + 600)

    params = {
        "client_id":             AZURE_CLIENT_ID,
        "response_type":         "code",
        "redirect_uri":          AZURE_REDIRECT_URI,
        "response_mode":         "query",
        "scope":                 AZURE_SCOPE,
        "state":                 state,
        "code_challenge":        challenge,
        "code_challenge_method": "S256",
    }
    return RedirectResponse(url=f"{AZURE_AUTH_URL}?{urlencode(params)}")


@app.get("/auth/callback")
async def auth_callback(request: Request, code: str, state: str):
    """Handle Azure redirect; exchange code for tokens; set session cookie."""
    entry = _oauth_states.pop(state, None)
    if not entry or entry[1] < time.time():
        raise HTTPException(400, "Invalid or expired OAuth state")

    verifier, _ = entry

    async with httpx.AsyncClient() as client:
        token_resp = await client.post(AZURE_TOKEN_URL, data={
            "client_id":     AZURE_CLIENT_ID,
            "client_secret": AZURE_CLIENT_SECRET,
            "grant_type":    "authorization_code",
            "code":          code,
            "redirect_uri":  AZURE_REDIRECT_URI,
            "code_verifier": verifier,
        })

    if token_resp.status_code != 200:
        logger.error("Token exchange failed: %s", token_resp.text)
        return HTMLResponse(_error_page("Authentication Failed", "Could not complete sign-in with Microsoft. Please try again."), status_code=400)

    tokens = token_resp.json()

    # Decode the id_token (no signature verify needed — came directly from Azure)
    id_token  = tokens.get("id_token", "")
    parts     = id_token.split(".")
    if len(parts) < 2:
        raise HTTPException(400, "Malformed id_token")

    padding   = 4 - len(parts[1]) % 4
    claims    = json.loads(urlsafe_b64decode(parts[1] + "=" * padding))
    user_email = claims.get("preferred_username") or claims.get("email") or claims.get("upn", "")
    user_name  = claims.get("name", user_email)

    if not user_email:
        return HTMLResponse(_error_page("Invalid Token", "Could not extract your account email from Microsoft's response. Try signing in again."), status_code=400)
    if user_email.lower() not in ALLOWED_EMAILS:
        logger.warning("Blocked login attempt from: %s", user_email)
        return HTMLResponse(RESPONSE_403, status_code=403)

    session_token = create_session(user_email, user_name)

    response = RedirectResponse(url="/", status_code=302)
    response.set_cookie(
        SESSION_COOKIE,
        session_token,
        httponly=True,
        samesite="lax",
        max_age=SESSION_TTL,
        secure=False,   # Set to True when behind HTTPS (e.g. zrok)
    )
    logger.info("User logged in: %s", user_email)
    return response


@app.get("/auth/logout")
async def auth_logout():
    response = RedirectResponse(url="/", status_code=302)
    response.delete_cookie(SESSION_COOKIE)
    azure_logout = (
        f"https://login.microsoftonline.com/{AZURE_TENANT_ID}/oauth2/v2.0/logout"
        f"?post_logout_redirect_uri=http://localhost:8080/"
    )
    return RedirectResponse(url=azure_logout, status_code=302)


@app.get("/auth/me")
async def auth_me(request: Request):
    """Return current user info (used by frontend JS)."""
    user = get_current_user(request)
    if not user:
        return JSONResponse({"authenticated": False}, status_code=401)
    return JSONResponse({"authenticated": True, "email": user["email"], "name": user["name"]})


# ── Root — serve the frontend HTML ──────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve index.html from ./static/."""
    index = STATIC_DIR / "index.html"
    if not index.exists():
        return HTMLResponse("<h1>Frontend not found — copy index.html to ./static/</h1>", status_code=404)
    return HTMLResponse(index.read_text())


# ── Chat (protected) ─────────────────────────────────────────────────

async def _stream_agent(message: str, thread_id: str):
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
            yield f"data: {json.dumps({'type': 'tool_start', 'tool': event.get('name', 'unknown')})}\n\n"

        elif kind == "on_tool_end":
            yield f"data: {json.dumps({'type': 'tool_end', 'tool': event.get('name', 'unknown')})}\n\n"

    yield f"data: {json.dumps({'type': 'done'})}\n\n"


@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    require_auth(request)          # 401 if not logged in
    thread_id = req.thread_id or str(uuid.uuid4())

    return StreamingResponse(
        _stream_agent(req.message, thread_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Thread-Id":   thread_id,
        },
    )


@app.get("/health")
async def health():
    return {"status": "ok", "agent_ready": agent is not None}
