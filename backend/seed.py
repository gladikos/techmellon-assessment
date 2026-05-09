"""
Seed the flights table with one week of fictional Aegis Airlines flights.

Design notes:
- All flights depart from Athens (ATH) and run for the next 7 days
  starting tomorrow (relative to current time). This keeps the data
  "fresh" no matter when the seed is run.
- Each route has 1-2 daily flights, each with 3 fare classes
  (economy_light, economy, business). This gives the agent enough
  variety to handle "cheapest", "next available", and seat-class
  scenarios meaningfully.
- A fixed random seed keeps the data deterministic across runs so
  the refinement loop sees consistent test signal.
- Idempotent: drops existing flights before inserting. Bookings are
  preserved (their FK constraint will fail if a flight they reference
  is deleted — which is the right behavior; bookings outliving their
  flight would be a bug).
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

from backend.database import get_conn, init_db, transaction

# Reproducibility — same data every run.
random.seed(42)

# Aegis Airlines route network. Origin is fixed; we vary destinations.
ORIGIN = "ATH"

# (destination_iata, city_name, base_price_eur, flight_duration_hours)
# Base price is the economy tier; other classes are computed from it.
ROUTES = [
    ("LHR", "London",    140, 3.75),
    ("CDG", "Paris",     120, 3.25),
    ("FCO", "Rome",       85, 1.75),
    ("BCN", "Barcelona", 110, 3.00),
    ("IST", "Istanbul",   75, 1.50),
]

# Flight number ranges per route (so AG1xx is London, AG2xx is Paris, etc.)
FLIGHT_NUMBER_PREFIXES = {
    "LHR": "AG1",
    "CDG": "AG2",
    "FCO": "AG3",
    "BCN": "AG4",
    "IST": "AG5",
}

# Fare classes and their price multipliers / capacities.
# Tuple: (fare_class, price_multiplier, seat_count)
# economy_light: cheapest, no checked bag, non-refundable
# economy:       standard, one checked bag, partial refund
# economy_flex:  premium economy, fully refundable, free changes
# business:      lie-flat, two checked bags, partial refund
# business_flex: business + fully refundable, priority changes
FARE_CLASSES = [
    ("economy_light", 0.65, 30),
    ("economy",       1.00, 80),
    ("economy_flex",  1.40, 25),
    ("business",      2.80, 12),
    ("business_flex", 3.60, 8),
]

# Departure time slots (24h format). Most routes get 2 daily flights.
DEPARTURE_SLOTS = ["08:00", "17:30"]


def _generate_flights() -> list[dict]:
    """Build the list of flight rows to insert, deterministically."""
    flights: list[dict] = []
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    # Generate 7 days starting from tomorrow (so flights are always in the future).
    for day_offset in range(1, 8):
        date = today + timedelta(days=day_offset)

        for dest_code, _city, base_price, duration_hours in ROUTES:
            prefix = FLIGHT_NUMBER_PREFIXES[dest_code]

            for slot_idx, departure_str in enumerate(DEPARTURE_SLOTS):
                hour, minute = map(int, departure_str.split(":"))
                departure = date.replace(hour=hour, minute=minute)
                arrival = departure + timedelta(hours=duration_hours)

                # Flight number: prefix + (00 + slot index) so morning is AG100, evening is AG101
                flight_number = f"{prefix}{slot_idx:02d}"

                # Each (flight_number, departure) gets all three fare classes as separate rows.
                for fare_class, multiplier, capacity in FARE_CLASSES:
                    # Tiny price jitter (±5%) makes "cheapest" non-trivial across days
                    jitter = random.uniform(0.95, 1.05)
                    price = round(base_price * multiplier * jitter, 2)

                    flights.append({
                        "flight_number":   flight_number,
                        "origin":          ORIGIN,
                        "destination":     dest_code,
                        "departure_time":  departure.isoformat(),
                        "arrival_time":    arrival.isoformat(),
                        "fare_class":      fare_class,
                        "price_eur":       price,
                        "seats_total":     capacity,
                        "seats_available": capacity,
                    })

    return flights


def seed() -> None:
    """Wipe and re-populate the flights table."""
    init_db()  # ensure schema exists

    flights = _generate_flights()

    with transaction() as conn:
        # Wipe existing flight data. Bookings reference flights via FK, so we
        # must delete bookings first if any exist (in dev/seed contexts only).
        conn.execute("DELETE FROM booking_addons")
        conn.execute("DELETE FROM bookings")
        conn.execute("DELETE FROM flights")

        conn.executemany(
            """
            INSERT INTO flights (
                flight_number, origin, destination,
                departure_time, arrival_time,
                fare_class, price_eur,
                seats_total, seats_available
            ) VALUES (
                :flight_number, :origin, :destination,
                :departure_time, :arrival_time,
                :fare_class, :price_eur,
                :seats_total, :seats_available
            )
            """,
            flights,
        )

        demo_bookings = [
            {
                "reference": "AGX-DEMO1",
                "passenger_name": "Maria Papadopoulou",
                "passenger_email": "maria.p@example.com",
                "fare_class": "economy",
                "flight_lookup": {"flight_number": "AG100", "fare_class": "economy"},
            },
            {
                "reference": "AGX-DEMO2",
                "passenger_name": "Andreas Christou",
                "passenger_email": "andreas.c@example.com",
                "fare_class": "economy",
                "flight_lookup": {"flight_number": "AG200", "fare_class": "economy"},
            },
            {
                "reference": "AGX-DEMO3",
                "passenger_name": "Eleni Markou",
                "passenger_email": "eleni.m@example.com",
                "fare_class": "economy",
                "flight_lookup": {"flight_number": "AG300", "fare_class": "economy"},
            },
            {
                "reference": "AGX-DEMO4",
                "passenger_name": "Nikos Stavros",
                "passenger_email": "nikos.s@example.com",
                "fare_class": "economy",
                "flight_lookup": {"flight_number": "AG400", "fare_class": "economy"},
            },
        ]

        now_iso = datetime.now(timezone.utc).isoformat()

        for booking in demo_bookings:
            reference = booking["reference"]
            flight_number = booking["flight_lookup"]["flight_number"]
            fare_class_lookup = booking["flight_lookup"]["fare_class"]

            row = conn.execute(
                """
                SELECT id, price_eur FROM flights
                WHERE flight_number = ? AND fare_class = ? AND departure_time > ?
                ORDER BY departure_time
                LIMIT 1
                """,
                (flight_number, fare_class_lookup, now_iso),
            ).fetchone()

            if row is None:
                raise RuntimeError(
                    f"Could not find a future flight for {reference}: "
                    f"{flight_number}/{fare_class_lookup}"
                )

            flight_id = row["id"]
            price_paid = row["price_eur"]

            conn.execute(
                """
                INSERT INTO bookings (
                    reference, flight_id, passenger_name, passenger_email,
                    fare_class, price_paid_eur, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'confirmed', ?)
                """,
                (
                    reference,
                    flight_id,
                    booking["passenger_name"],
                    booking["passenger_email"],
                    booking["fare_class"],
                    price_paid,
                    now_iso,
                ),
            )

            conn.execute(
                "UPDATE flights SET seats_available = seats_available - 1 WHERE id = ?",
                (flight_id,),
            )

        print(f"Seeded {len(demo_bookings)} demo bookings: AGX-DEMO1..AGX-DEMO4")

    print(f"Seeded {len(flights)} flight rows across "
          f"{len(ROUTES)} destinations × 7 days × {len(FARE_CLASSES)} fare classes.")


if __name__ == "__main__":
    seed()