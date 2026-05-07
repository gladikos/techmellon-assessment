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

from fastapi import APIRouter, Query

from backend.database import get_conn
from backend.models import FareClass, FlightResponse

router = APIRouter(prefix="/flights", tags=["flights"])


@router.get("/search", response_model=list[FlightResponse])
async def search_flights(
    destination: str = Query(..., min_length=3, max_length=3, description="IATA code, e.g. CDG"),
    date: Optional[str] = Query(None, description="ISO date YYYY-MM-DD"),
    max_price: Optional[float] = Query(None, ge=0),
    fare_class: Optional[FareClass] = Query(None),
    limit: int = Query(20, ge=1, le=100),
) -> list[FlightResponse]:
    """Search the flight catalog with optional filters."""
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
        # Match any flight whose departure_time starts with the requested date.
        # Stored as ISO 8601 strings, so prefix-match works correctly.
        sql += " AND departure_time LIKE ?"
        params.append(f"{date}%")

    if max_price is not None:
        sql += " AND price_eur <= ?"
        params.append(max_price)

    if fare_class:
        sql += " AND fare_class = ?"
        params.append(fare_class)

    sql += " ORDER BY departure_time ASC, price_eur ASC LIMIT ?"
    params.append(limit)

    with get_conn() as conn:
        rows = conn.execute(sql, params).fetchall()

    return [FlightResponse(**dict(row)) for row in rows]