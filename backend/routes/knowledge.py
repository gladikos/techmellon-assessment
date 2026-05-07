"""
Knowledge base routes.

Loads policies from data/knowledge_base.json once at startup (the file is
small and never changes during a run, so caching in memory is fine). The
agent's tool calls one endpoint:

  GET /knowledge/{topic}

Returns the requested topic with its summary, rules, and details.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException

from backend.models import KnowledgeTopicResponse

router = APIRouter(prefix="/knowledge", tags=["knowledge"])

KB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "knowledge_base.json"


@lru_cache(maxsize=1)
def _load_kb() -> dict:
    """Load the KB once and cache it. Excludes the _meta key from queryable topics."""
    with open(KB_PATH, encoding="utf-8") as f:
        return json.load(f)


@router.get("/topics", response_model=list[str])
async def list_topics() -> list[str]:
    """List all available KB topic names (excluding metadata)."""
    kb = _load_kb()
    return [k for k in kb.keys() if not k.startswith("_")]


@router.get("/{topic}", response_model=KnowledgeTopicResponse)
async def get_topic(topic: str) -> KnowledgeTopicResponse:
    """Fetch a single KB topic by name."""
    kb = _load_kb()

    # Defense: don't let callers retrieve metadata or random keys.
    if topic.startswith("_") or topic not in kb:
        available = [k for k in kb.keys() if not k.startswith("_")]
        raise HTTPException(
            status_code=404,
            detail=f"Unknown topic '{topic}'. Available: {', '.join(available)}",
        )

    entry = kb[topic]
    return KnowledgeTopicResponse(
        topic=topic,
        summary=entry["summary"],
        rules=entry.get("rules", {}),
        details=entry["details"],
    )