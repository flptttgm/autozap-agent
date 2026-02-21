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

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# LangSmith tracing
if os.environ.get("LANGCHAIN_TRACING_V2") == "true":
    os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

from agent.graph import AutozapAgent

# ─── Logging ───
logging.basicConfig(level=logging.INFO)
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

# ─── Agent Instance (singleton) ───
agent: AutozapAgent | None = None

def get_agent() -> AutozapAgent:
    global agent
    if agent is None:
        agent = AutozapAgent()
    return agent


# ─── Models ───
class ProcessRequest(BaseModel):
    lead_id: str
    workspace_id: str
    message: str
    agent_config: dict = {}
    instance_id: str | None = None


class ProcessResponse(BaseModel):
    response: str | None
    status: str
    tools_used: list[str] = []
    latency_ms: float = 0


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
        agent_instance = get_agent()

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
        logger.error(f"[process] Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# ─── Run ───
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
