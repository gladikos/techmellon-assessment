"""
Flight search routes.

The agent's tool calls one endpoint:

  GET /flights/search
       ?destination=CDG       (required)
       &date=2026-05-12       (optional, ISO date)
       &max_price=150         (optional, EUR)
       &fare_class=economy    (optional)

Returns matching flights ordered by departure time then price (cheapest first).
Limits to 20 results to keep agent tool responses tight.

Design notes:
  - All filters are optional except destination (otherwise we'd return the
    whole catalog, which is noisy for the agent).
  - We exclude flights with seats_available = 0 by default — the agent
    should not be offered to "find" sold-out flights.
"""

from __future__ import annotations

from typing import Optional
import math

from fastapi import APIRouter, HTTPException, Query

from backend.database import get_conn
from backend.models import FareClass, FlightResponse

router = APIRouter(prefix="/flights", tags=["flights"])


@router.get("/search", response_model=list[FlightResponse])
async def search_flights(
    destination: str = Query(..., min_length=3, max_length=3, description="IATA code, e.g. CDG"),
    date: Optional[str] = Query(None, description="ISO date YYYY-MM-DD"),
    max_price: Optional[str] = Query(None, description="Maximum price in EUR"),
    fare_class: Optional[str] = Query(None, description="Fare class"),
    limit: int = Query(20, ge=1, le=100),
) -> list[FlightResponse]:
    """Search the flight catalog with optional filters.

    Note: optional params accept empty strings as 'missing' to be tolerant
    of upstream callers (e.g. ElevenLabs webhook tools) that send unset
    optional params as '' rather than omitting them entirely.
    """
    # Normalize: treat empty strings as missing.
    date = date or None
    fare_class = fare_class or None
    max_price_value: Optional[float] = None
    if max_price not in (None, "", "NaN", "null", "undefined"):
        try:
            max_price_value = float(max_price)
            if math.isnan(max_price_value) or math.isinf(max_price_value):
                max_price_value = None  # treat NaN/Inf as "no filter"
            elif max_price_value < 0:
                raise ValueError("max_price must be >= 0")
        except ValueError:
            raise HTTPException(400, f"Invalid max_price: {max_price!r}")

    # Validate fare_class against allowed set (was a Literal type before).
    allowed_fare_classes = {
        "economy_light", "economy", "economy_flex", "business", "business_flex"
    }
    if fare_class and fare_class not in allowed_fare_classes:
        raise HTTPException(
            400,
            f"Invalid fare_class: {fare_class!r}. "
            f"Allowed: {', '.join(sorted(allowed_fare_classes))}",
        )

    sql = """
        SELECT id, flight_number, origin, destination,
               departure_time, arrival_time,
               fare_class, price_eur, seats_available
        FROM flights
        WHERE destination = ?
          AND seats_available > 0
    """
    params: list = [destination.upper()]

    if date:
        sql += " AND departure_time LIKE ?"
        params.append(f"{date}%")

    if max_price_value is not None:
        sql += " AND price_eur <= ?"
        params.append(max_price_value)

    if fare_class:
        sql += " AND fare_class = ?"
        params.append(fare_class)

    sql += " ORDER BY departure_time ASC, price_eur ASC LIMIT ?"
    params.append(limit)

    with get_conn() as conn:
        rows = conn.execute(sql, params).fetchall()

    return [FlightResponse(**dict(row)) for row in rows]