"""
FastAPI app entry point.

Run locally with:
    uvicorn backend.main:app --reload --port 8000

Hits to the agent's webhooks (from ElevenLabs) come into this server.
Routes are split by concern:
  - /knowledge/*  → static airline policy lookups
  - /flights/*    → search the flight catalog
  - /bookings/*   → create, retrieve, cancel, reschedule, add addons

Design notes:
  - lifespan handler initializes the DB schema on startup (no manual setup)
  - permissive CORS for local dev (ElevenLabs webhook calls cross-origin
    once we expose via a tunnel; tightening CORS is a production task)
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.database import init_db
from backend.routes import bookings, flights, knowledge


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: ensure the schema exists. Shutdown: nothing to do."""
    init_db()
    yield


app = FastAPI(
    title="Aegis Airlines API",
    description="Backend for the ElevenLabs airline customer service agent.",
    version="0.1.0",
    lifespan=lifespan,
)

# Permissive CORS for local development. In production this would be locked
# down to the exact ElevenLabs / agent origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the route modules.
app.include_router(knowledge.router)
app.include_router(flights.router)
app.include_router(bookings.router)


@app.get("/health", tags=["meta"])
async def health() -> dict:
    """Liveness probe. The orchestrator hits this before starting a run."""
    return {"status": "ok"}