"""
Database layer for the airline backend.

Responsibilities:
- Open SQLite connections with foreign keys enabled
- Define and create the schema on first run
- Provide a context-manager helper for safe transactions

Design notes:
- Uses raw sqlite3 (not SQLAlchemy) because we have 3 tables and ~5 query
  shapes; an ORM would be ceremony without payoff at this size.
- Connection-per-request pattern: a fresh sqlite3.Connection is created for
  each operation. SQLite is fast at opening connections, and this avoids
  thread-safety pitfalls when FastAPI runs multiple workers.
- PRAGMA foreign_keys = ON is enforced on every connection because SQLite
  silently ignores foreign keys without it.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

# Path to the SQLite file. Lives in /data so it can be .gitignored cleanly.
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "airline.db"


# --- Schema --------------------------------------------------------------- #

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS flights (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    flight_number   TEXT    NOT NULL,
    origin          TEXT    NOT NULL,
    destination     TEXT    NOT NULL,
    departure_time  TEXT    NOT NULL,
    arrival_time    TEXT    NOT NULL,
    fare_class      TEXT    NOT NULL CHECK (fare_class IN (
        'economy_light', 'economy', 'economy_flex', 'business', 'business_flex'
    )),
    price_eur       REAL    NOT NULL CHECK (price_eur >= 0),
    seats_total     INTEGER NOT NULL CHECK (seats_total > 0),
    seats_available INTEGER NOT NULL CHECK (seats_available >= 0),
    UNIQUE (flight_number, fare_class, departure_time)
);

CREATE INDEX IF NOT EXISTS idx_flights_search
    ON flights (destination, departure_time, price_eur);

CREATE TABLE IF NOT EXISTS bookings (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    reference        TEXT    NOT NULL UNIQUE,
    flight_id        INTEGER NOT NULL,
    passenger_name   TEXT    NOT NULL,
    passenger_email  TEXT    NOT NULL,
    seat             TEXT,
    seat_preference  TEXT,
    fare_class       TEXT    NOT NULL,
    price_paid_eur   REAL    NOT NULL,
    status           TEXT    NOT NULL DEFAULT 'confirmed' CHECK (status IN (
        'confirmed', 'cancelled', 'rescheduled'
    )),
    created_at       TEXT    NOT NULL,
    cancelled_at     TEXT,
    FOREIGN KEY (flight_id) REFERENCES flights(id)
);

CREATE INDEX IF NOT EXISTS idx_bookings_reference
    ON bookings (reference);

CREATE TABLE IF NOT EXISTS booking_addons (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    booking_id  INTEGER NOT NULL,
    addon_type  TEXT    NOT NULL CHECK (addon_type IN (
        'baggage', 'pet', 'sports_equipment', 'musical_instrument',
        'special_assistance', 'pram', 'wheelchair'
    )),
    description TEXT    NOT NULL,
    fee_eur     REAL    NOT NULL DEFAULT 0,
    created_at  TEXT    NOT NULL,
    FOREIGN KEY (booking_id) REFERENCES bookings(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_addons_booking
    ON booking_addons (booking_id);
"""


# --- Connection helpers --------------------------------------------------- #

def _connect() -> sqlite3.Connection:
    """
    Open a SQLite connection with sane defaults:
    - foreign keys enforced (off by default, famous footgun)
    - row factory set so query results behave like dicts
    - isolation_level=None to use explicit BEGIN/COMMIT control
      (we wrap writes in the `transaction` context manager below)
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def get_conn() -> Iterator[sqlite3.Connection]:
    """
    Context manager for read operations.

    Usage:
        with get_conn() as conn:
            rows = conn.execute("SELECT ...").fetchall()
    """
    conn = _connect()
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def transaction() -> Iterator[sqlite3.Connection]:
    """
    Context manager for write operations. Wraps the block in BEGIN/COMMIT
    and rolls back on any exception. Use this for anything that mutates data.

    Usage:
        with transaction() as conn:
            conn.execute("INSERT INTO bookings ...")
            conn.execute("UPDATE flights SET seats_available = ...")
    """
    conn = _connect()
    try:
        conn.execute("BEGIN")
        yield conn
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


# --- Schema initialization ------------------------------------------------ #

def init_db() -> None:
    """
    Create all tables and indexes. Safe to call multiple times — every
    statement uses IF NOT EXISTS, so calling this on a populated DB is a
    no-op.
    """
    with get_conn() as conn:
        conn.executescript(SCHEMA_SQL)


if __name__ == "__main__":
    # Allow running `python -m backend.database` to initialize an empty DB.
    init_db()
    print(f"Initialized DB at {DB_PATH}")