"""
============================================
Autozap Agent Service - FastAPI Server
============================================
Ponto de entrada HTTP para o agente Python.
Recebe chamadas da Edge Function process-message.
"""

import os
import time
import logging
import traceback

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─── Logging ───
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("autozap-agent")

# ─── App ───
app = FastAPI(
    title="Autozap Agent Service",
    description="Agente Python com LangChain para processamento inteligente de mensagens",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Models ───
class ProcessRequest(BaseModel):
    lead_id: str
    workspace_id: str
    message: str
    agent_config: dict = {}
    instance_id: str | None = None
    ai_api_key: str | None = None
    ai_model: str | None = None


class ProcessResponse(BaseModel):
    response: str | None
    status: str
    tools_used: list[str] = []
    latency_ms: float = 0


# ─── Agent (lazy loaded) ───
_agents: dict[str, object] = {}
_agent_class = None


def _get_agent_class():
    """Lazy import of AutozapAgent to avoid crashing on startup."""
    global _agent_class
    if _agent_class is None:
        logger.info("Loading AutozapAgent class (first request)...")
        from agent.graph import AutozapAgent
        _agent_class = AutozapAgent
        logger.info("AutozapAgent loaded successfully!")
    return _agent_class


def get_agent(ai_api_key: str | None = None, ai_model: str | None = None):
    """Get or create an agent instance. Caches by API key + model combo."""
    AgentClass = _get_agent_class()
    key = f"{ai_api_key or 'env'}:{ai_model or 'default'}"
    if key not in _agents:
        _agents[key] = AgentClass(ai_api_key=ai_api_key, ai_model=ai_model)
    return _agents[key]


# ─── Routes ───
@app.get("/health")
async def health():
    return {"status": "ok", "service": "autozap-agent", "version": "1.0.0"}


@app.post("/process", response_model=ProcessResponse)
async def process_message(req: ProcessRequest, request: Request):
    """Processa uma mensagem e retorna a resposta do agente."""

    # Autenticação simples
    agent_secret = os.environ.get("AGENT_SECRET", "")
    if agent_secret:
        auth_header = request.headers.get("authorization", "")
        token = auth_header.replace("Bearer ", "")
        if token != agent_secret:
            raise HTTPException(status_code=401, detail="Unauthorized")

    start_time = time.time()

    try:
        agent_instance = get_agent(ai_api_key=req.ai_api_key, ai_model=req.ai_model)

        result = await agent_instance.process(
            lead_id=req.lead_id,
            workspace_id=req.workspace_id,
            message=req.message,
            agent_config=req.agent_config,
            instance_id=req.instance_id,
        )

        latency = (time.time() - start_time) * 1000

        logger.info(
            f"[process] lead={req.lead_id[:8]}... "
            f"tools={result.get('tools_used', [])} "
            f"latency={latency:.0f}ms"
        )

        return ProcessResponse(
            response=result.get("response"),
            status=result.get("status", "success"),
            tools_used=result.get("tools_used", []),
            latency_ms=round(latency, 1),
        )

    except Exception as e:
        logger.error(f"[process] Error: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def startup_event():
    logger.info("="*50)
    logger.info("Autozap Agent Service starting...")
    logger.info(f"PORT={os.environ.get('PORT', 'not set')}")
    logger.info(f"SUPABASE_URL={'set' if os.environ.get('SUPABASE_URL') else 'NOT SET'}")
    logger.info(f"SUPABASE_SERVICE_ROLE_KEY={'set' if os.environ.get('SUPABASE_SERVICE_ROLE_KEY') else 'NOT SET'}")
    logger.info(f"LANGCHAIN_TRACING_V2={os.environ.get('LANGCHAIN_TRACING_V2', 'not set')}")
    logger.info("Server ready! Agent will load on first request.")
    logger.info("="*50)


# ─── Run ───
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
