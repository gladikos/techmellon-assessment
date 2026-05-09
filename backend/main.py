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

import logging
import queue
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# from backend.database import init_db - We do not need to initialize the 
# database at startup, as the seed function will handle that.
from backend.seed import seed
from backend.routes import bookings, flights, knowledge
from backend.routes.dashboard import router as dashboard_router


# --- Log fan-out infrastructure ----------------------------------------- #

_log_subscribers: list[queue.Queue] = []
_subscribers_lock = threading.Lock()


def subscribe_to_logs() -> queue.Queue:
    q: queue.Queue = queue.Queue(maxsize=200)
    with _subscribers_lock:
        _log_subscribers.append(q)
    return q


def unsubscribe_from_logs(q: queue.Queue) -> None:
    with _subscribers_lock:
        if q in _log_subscribers:
            _log_subscribers.remove(q)


class WebsocketBroadcastHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        item = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": msg,
        }
        with _subscribers_lock:
            subscribers = list(_log_subscribers)
        for sub_q in subscribers:
            try:
                sub_q.put_nowait(item)
            except queue.Full:
                try:
                    sub_q.get_nowait()
                    sub_q.put_nowait(item)
                except Exception:
                    pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: wipe and re-seed the database for a clean state. Shutdown: nothing to do."""
    seed()
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
app.include_router(dashboard_router)

# Attach the broadcast handler to the uvicorn access logger.
_broadcast_handler = WebsocketBroadcastHandler()
_broadcast_handler.setLevel(logging.INFO)
logging.getLogger("uvicorn.access").addHandler(_broadcast_handler)
print("WebsocketBroadcastHandler attached to uvicorn.access")


@app.get("/health", tags=["meta"])
async def health() -> dict:
    """Liveness probe. The orchestrator hits this before starting a run."""
    return {"status": "ok"}