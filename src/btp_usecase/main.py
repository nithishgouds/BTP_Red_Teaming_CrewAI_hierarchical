from __future__ import annotations

from fastapi import FastAPI

from .api.routes.agent import router as agent_router
from .api.routes.health import router as health_router
from .api.routes.upload import router as upload_router

app = FastAPI(title="BTP Privacy Red Teaming Prototype", version="0.1.0")

app.include_router(health_router)
app.include_router(upload_router)
app.include_router(agent_router)


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "BTP Privacy Red Teaming Prototype"}
