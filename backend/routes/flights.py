"""
Flight search routes.

The agent's tool calls one endpoint:

  GET /flights/search
       ?destination=CDG       (optional, IATA code)
       &date=2026-05-12       (optional, ISO date)
       &max_price=150         (optional, EUR)
       &fare_class=economy    (optional)
       &sort=price            (optional, 'time' or 'price'; default 'time')

At least one filter is recommended — omitting all returns the full catalog.
Returns up to 20 results by default; sort by departure time (default) or price.

Design notes:
  - destination is now optional so callers can search by date or price alone.
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
    destination: Optional[str] = Query(None, min_length=3, max_length=3, description="IATA code, e.g. CDG (optional — omit to search across all destinations)"),
    date: Optional[str] = Query(None, description="ISO date YYYY-MM-DD (start date, inclusive)"),
    date_to: Optional[str] = Query(None, description="ISO date YYYY-MM-DD (end date, inclusive); used with date for range queries"),
    max_price: Optional[str] = Query(None, description="Maximum price in EUR"),
    fare_class: Optional[str] = Query(None, description="Fare class"),
    sort: str = Query("time", description="Sort order: 'time' (default) or 'price'"),
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
        WHERE seats_available > 0
    """
    params: list = []

    if destination:
        sql += " AND destination = ?"
        params.append(destination.upper())

    date_to = date_to or None
    if date and date_to:
        sql += " AND DATE(departure_time) BETWEEN ? AND ?"
        params.extend([date, date_to])
    elif date:
        sql += " AND departure_time LIKE ?"
        params.append(f"{date}%")
    elif date_to:
        sql += " AND DATE(departure_time) <= ?"
        params.append(date_to)

    if max_price_value is not None:
        sql += " AND price_eur <= ?"
        params.append(max_price_value)

    if fare_class:
        sql += " AND fare_class = ?"
        params.append(fare_class)

    if sort == "price":
        sql += " ORDER BY price_eur ASC, departure_time ASC LIMIT ?"
    else:
        sql += " ORDER BY departure_time ASC, price_eur ASC LIMIT ?"
    params.append(limit)

    with get_conn() as conn:
        rows = conn.execute(sql, params).fetchall()

    return [FlightResponse(**dict(row)) for row in rows]