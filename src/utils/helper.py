# ── System prompt ────────────────────────────────────────────────────

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

RESPONSE_403 = """
        <!DOCTYPE html>
        <html>
        <head><title>Access Denied</title>
        <style>
          body { font-family: Inter, sans-serif; background: #050510; color: #f0f0f5;
                 display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; }
          .box { text-align: center; padding: 48px; background: rgba(15,15,30,0.85);
                 border: 1px solid rgba(255,255,255,0.06); border-radius: 24px; max-width: 400px; }
          h1 { color: #ef4444; font-size: 20px; margin-bottom: 12px; }
          p  { color: #8888a8; font-size: 14px; line-height: 1.6; }
          a  { color: #6366f1; text-decoration: none; font-size: 13px; }
        </style>
        </head>
        <body><div class="box">
          <h1>Access Denied</h1>
          <p>Your Microsoft account is not authorised to use TREX Agent.<br><br>
          Contact the administrator to request access.</p>
          <br><a href="/">← Back to login</a>
        </div></body>
        </html>
    """


def _error_page(title: str, message: str) -> str:
    return f"""<!DOCTYPE html>
<html>
<head><title>{title}</title>
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{
    font-family: Inter, -apple-system, sans-serif;
    background: #050510; color: #f0f0f5;
    display: flex; align-items: center; justify-content: center;
    height: 100vh;
  }}
  .bg-glow {{ position:fixed; inset:0; pointer-events:none; overflow:hidden; }}
  .orb {{
    position:absolute; border-radius:50%; filter:blur(120px); opacity:0.12;
    animation: float 20s ease-in-out infinite;
  }}
  .orb:nth-child(1) {{ width:500px;height:500px;background:#6366f1;top:-150px;left:-100px; }}
  .orb:nth-child(2) {{ width:400px;height:400px;background:#ef4444;bottom:-100px;right:-80px;animation-delay:-8s; }}
  @keyframes float {{
    0%,100% {{ transform:translate(0,0); }}
    50% {{ transform:translate(20px,-30px); }}
  }}
  .card {{
    position: relative; z-index: 1;
    text-align: center; padding: 48px 40px;
    background: rgba(15,15,30,0.85);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 24px;
    backdrop-filter: blur(40px);
    box-shadow: 0 25px 70px -15px rgba(0,0,0,0.6), 0 0 40px rgba(239,68,68,0.1);
    max-width: 420px; width: 90%;
    animation: cardIn 0.4s cubic-bezier(0.16,1,0.3,1);
  }}
  @keyframes cardIn {{
    from {{ opacity:0; transform:translateY(16px) scale(0.97); }}
    to   {{ opacity:1; transform:translateY(0) scale(1); }}
  }}
  .icon {{
    width:56px; height:56px; border-radius:16px; margin: 0 auto 20px;
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.2);
    display:flex; align-items:center; justify-content:center; font-size:24px;
  }}
  h1 {{ font-size:20px; font-weight:700; color:#ef4444; margin-bottom:12px; }}
  p  {{ font-size:14px; color:#8888a8; line-height:1.65; max-width:300px; margin:0 auto; }}
  .actions {{ display:flex; gap:10px; justify-content:center; margin-top:28px; }}
  .btn {{
    padding: 10px 20px; border-radius:12px; font-size:13px;
    font-family:inherit; cursor:pointer; text-decoration:none;
    transition: all 0.2s ease;
  }}
  .btn-primary {{
    background: #6366f1; color:#fff; border:none;
    box-shadow: 0 4px 15px rgba(99,102,241,0.3);
  }}
  .btn-primary:hover {{ background:#4f46e5; transform:translateY(-1px); }}
  .btn-ghost {{
    background: rgba(255,255,255,0.04); color:#8888a8;
    border: 1px solid rgba(255,255,255,0.06);
  }}
  .btn-ghost:hover {{ background:rgba(255,255,255,0.08); color:#f0f0f5; }}
  .code {{ font-size:11px; color:#55556a; margin-top:20px; font-family:monospace; }}
</style>
</head>
<body>
  <div class="bg-glow"><div class="orb"></div><div class="orb"></div></div>
  <div class="card">
    <div class="icon">&#x26A0;&#xFE0F;</div>
    <h1>{title}</h1>
    <p>{message}</p>
    <div class="actions">
      <a href="/auth/login" class="btn btn-primary">Try again</a>
      <a href="/" class="btn btn-ghost">Home</a>
    </div>
    <div class="code">HTTP 401 — Unauthorised</div>
  </div>
</body>
</html>"""