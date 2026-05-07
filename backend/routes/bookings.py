"""
Booking routes — the heart of the backend.

Endpoints:
  POST   /bookings                     create a booking on a flight
  GET    /bookings/{reference}         retrieve a booking + its addons
  POST   /bookings/{reference}/cancel  cancel and return the seat
  POST   /bookings/{reference}/reschedule  move to a different flight
  POST   /bookings/{reference}/addons  attach baggage / pet / etc.

Critical design choices:
  - References are short and human-pronounceable: "AGX-7K2P9". The agent
    needs to be able to read them back to a customer.
  - Double-booking prevented at SQL level: the booking transaction
    decrements seats_available with a conditional UPDATE. If 0 rows
    update, no seat was free and the booking fails atomically. The
    flights.seats_available column also has CHECK (>= 0) as a backstop.
  - Cancellation returns the seat to inventory (seats_available += 1)
    inside the same transaction as marking the booking cancelled.
  - Rescheduling is two atomic ops: free the old seat, claim the new one.
    Both happen in a single transaction; if claiming the new seat fails
    (e.g. the new flight is full), the rollback restores the old seat.
"""

from __future__ import annotations

import secrets
import string
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from backend.database import get_conn, transaction
from backend.models import (
    BookingAddonRequest,
    BookingAddonResponse,
    BookingCreateRequest,
    BookingRescheduleRequest,
    BookingResponse,
)

router = APIRouter(prefix="/bookings", tags=["bookings"])


# --- Helpers ------------------------------------------------------------- #

def _now() -> str:
    """Current UTC time as an ISO 8601 string. Used for created_at / cancelled_at."""
    return datetime.now(timezone.utc).isoformat()


def _generate_reference() -> str:
    """
    Produce a booking reference like 'AGX-7K2P9'.

    Format: 'AGX-' prefix + 5 chars from an unambiguous alphabet
    (no I/O/0/1 to avoid confusion when read aloud over a phone line).
    Collisions are vanishingly unlikely at 33^5 ≈ 39M combinations,
    but the bookings.reference UNIQUE constraint guarantees correctness
    even in the rare event of a collision (we'd retry).
    """
    alphabet = "".join(c for c in string.ascii_uppercase + string.digits if c not in "IO01")
    return "AGX-" + "".join(secrets.choice(alphabet) for _ in range(5))


def _hydrate_booking(conn, booking_row) -> BookingResponse:
    """
    Build a BookingResponse from a bookings row, joining the flight info
    and any addons.
    """
    flight = conn.execute(
        "SELECT flight_number, origin, destination, departure_time "
        "FROM flights WHERE id = ?",
        (booking_row["flight_id"],),
    ).fetchone()

    addons = conn.execute(
        "SELECT id, addon_type, description, fee_eur, created_at "
        "FROM booking_addons WHERE booking_id = ? ORDER BY created_at",
        (booking_row["id"],),
    ).fetchall()

    return BookingResponse(
        reference=booking_row["reference"],
        flight_id=booking_row["flight_id"],
        flight_number=flight["flight_number"],
        origin=flight["origin"],
        destination=flight["destination"],
        departure_time=flight["departure_time"],
        passenger_name=booking_row["passenger_name"],
        passenger_email=booking_row["passenger_email"],
        seat=booking_row["seat"],
        seat_preference=booking_row["seat_preference"],
        fare_class=booking_row["fare_class"],
        price_paid_eur=booking_row["price_paid_eur"],
        status=booking_row["status"],
        created_at=booking_row["created_at"],
        cancelled_at=booking_row["cancelled_at"],
        addons=[BookingAddonResponse(**dict(a)) for a in addons],
    )


# --- Routes -------------------------------------------------------------- #

@router.post("", response_model=BookingResponse, status_code=201)
async def create_booking(req: BookingCreateRequest) -> BookingResponse:
    """Book a seat on a flight. Atomic, race-free seat decrement."""
    with transaction() as conn:
        # Conditional UPDATE: only decrement if seats_available > 0.
        # If the flight is full or doesn't exist, rowcount will be 0.
        cursor = conn.execute(
            """
            UPDATE flights
               SET seats_available = seats_available - 1
             WHERE id = ?
               AND seats_available > 0
            """,
            (req.flight_id,),
        )
        if cursor.rowcount == 0:
            # Distinguish "flight doesn't exist" from "flight full" for clarity.
            exists = conn.execute(
                "SELECT 1 FROM flights WHERE id = ?", (req.flight_id,)
            ).fetchone()
            if not exists:
                raise HTTPException(404, f"Flight {req.flight_id} not found")
            raise HTTPException(409, f"Flight {req.flight_id} is fully booked")

        # Snapshot the flight's price and fare class at booking time.
        flight = conn.execute(
            "SELECT fare_class, price_eur FROM flights WHERE id = ?",
            (req.flight_id,),
        ).fetchone()

        reference = _generate_reference()
        conn.execute(
            """
            INSERT INTO bookings (
                reference, flight_id, passenger_name, passenger_email,
                seat_preference, fare_class, price_paid_eur,
                status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'confirmed', ?)
            """,
            (
                reference, req.flight_id, req.passenger_name, req.passenger_email,
                req.seat_preference, flight["fare_class"], flight["price_eur"],
                _now(),
            ),
        )

        booking = conn.execute(
            "SELECT * FROM bookings WHERE reference = ?", (reference,)
        ).fetchone()

        return _hydrate_booking(conn, booking)


@router.get("/{reference}", response_model=BookingResponse)
async def get_booking(reference: str) -> BookingResponse:
    """Retrieve a booking and its addons by reference."""
    with get_conn() as conn:
        booking = conn.execute(
            "SELECT * FROM bookings WHERE reference = ?", (reference,)
        ).fetchone()
        if not booking:
            raise HTTPException(404, f"Booking '{reference}' not found")
        return _hydrate_booking(conn, booking)


@router.post("/{reference}/cancel", response_model=BookingResponse)
async def cancel_booking(reference: str) -> BookingResponse:
    """Cancel a booking and return the seat to flight inventory."""
    with transaction() as conn:
        booking = conn.execute(
            "SELECT * FROM bookings WHERE reference = ?", (reference,)
        ).fetchone()
        if not booking:
            raise HTTPException(404, f"Booking '{reference}' not found")
        if booking["status"] == "cancelled":
            raise HTTPException(409, f"Booking '{reference}' is already cancelled")

        # Mark cancelled.
        conn.execute(
            "UPDATE bookings SET status = 'cancelled', cancelled_at = ? "
            "WHERE reference = ?",
            (_now(), reference),
        )
        # Return the seat.
        conn.execute(
            "UPDATE flights SET seats_available = seats_available + 1 "
            "WHERE id = ?",
            (booking["flight_id"],),
        )

        updated = conn.execute(
            "SELECT * FROM bookings WHERE reference = ?", (reference,)
        ).fetchone()
        return _hydrate_booking(conn, updated)


@router.post("/{reference}/reschedule", response_model=BookingResponse)
async def reschedule_booking(
    reference: str, req: BookingRescheduleRequest
) -> BookingResponse:
    """Move a booking to a different flight, atomically."""
    with transaction() as conn:
        booking = conn.execute(
            "SELECT * FROM bookings WHERE reference = ?", (reference,)
        ).fetchone()
        if not booking:
            raise HTTPException(404, f"Booking '{reference}' not found")
        if booking["status"] == "cancelled":
            raise HTTPException(409, f"Cannot reschedule a cancelled booking")
        if booking["flight_id"] == req.new_flight_id:
            raise HTTPException(400, "New flight is the same as the current one")

        # Try to claim a seat on the new flight (same atomic pattern as create).
        cursor = conn.execute(
            "UPDATE flights SET seats_available = seats_available - 1 "
            "WHERE id = ? AND seats_available > 0",
            (req.new_flight_id,),
        )
        if cursor.rowcount == 0:
            exists = conn.execute(
                "SELECT 1 FROM flights WHERE id = ?", (req.new_flight_id,)
            ).fetchone()
            if not exists:
                raise HTTPException(404, f"Flight {req.new_flight_id} not found")
            raise HTTPException(409, f"Flight {req.new_flight_id} is fully booked")

        # Free the old seat.
        conn.execute(
            "UPDATE flights SET seats_available = seats_available + 1 "
            "WHERE id = ?",
            (booking["flight_id"],),
        )

        # Snapshot new flight's price/class.
        new_flight = conn.execute(
            "SELECT fare_class, price_eur FROM flights WHERE id = ?",
            (req.new_flight_id,),
        ).fetchone()

        conn.execute(
            """
            UPDATE bookings
               SET flight_id = ?, fare_class = ?, price_paid_eur = ?,
                   status = 'rescheduled'
             WHERE reference = ?
            """,
            (req.new_flight_id, new_flight["fare_class"],
             new_flight["price_eur"], reference),
        )

        updated = conn.execute(
            "SELECT * FROM bookings WHERE reference = ?", (reference,)
        ).fetchone()
        return _hydrate_booking(conn, updated)


@router.post("/{reference}/addons", response_model=BookingResponse)
async def add_addon(
    reference: str, req: BookingAddonRequest
) -> BookingResponse:
    """Attach an addon (baggage, pet, sports equipment, assistance, etc.)."""
    with transaction() as conn:
        booking = conn.execute(
            "SELECT * FROM bookings WHERE reference = ?", (reference,)
        ).fetchone()
        if not booking:
            raise HTTPException(404, f"Booking '{reference}' not found")
        if booking["status"] == "cancelled":
            raise HTTPException(409, "Cannot add addons to a cancelled booking")

        conn.execute(
            """
            INSERT INTO booking_addons (
                booking_id, addon_type, description, fee_eur, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (booking["id"], req.addon_type, req.description, req.fee_eur, _now()),
        )

        return _hydrate_booking(conn, booking)