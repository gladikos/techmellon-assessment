"""
Pydantic request/response models for the FastAPI backend.

Each model represents either:
  - the JSON body of a request (e.g. BookingCreateRequest), OR
  - the JSON shape of a response (e.g. BookingResponse).

Keeping these in one file (rather than colocated with their routes) makes
it easy for the prompt fixer and code fixer to see the full data contract
at a glance.

Validation we lean on Pydantic for:
  - required fields (no defaults → required)
  - email format (EmailStr)
  - enum constraints (Literal types) for fare_class, seat_preference, addon_type
  - date format (str, validated by individual route logic when needed)
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, EmailStr, Field

# --- Shared / Enum-like literals ----------------------------------------- #

FareClass = Literal[
    "economy_light", "economy", "economy_flex", "business", "business_flex"
]
SeatPreference = Literal["window", "aisle", "middle", "extra_legroom", "exit_row"]
BookingStatus = Literal["confirmed", "cancelled", "rescheduled"]
AddonType = Literal[
    "baggage", "pet", "sports_equipment", "musical_instrument",
    "special_assistance", "pram", "wheelchair"
]


# --- Flight / search ----------------------------------------------------- #

class FlightResponse(BaseModel):
    """Public shape of a flight as the agent / customer sees it."""
    id: int
    flight_number: str
    origin: str
    destination: str
    departure_time: str
    arrival_time: str
    fare_class: FareClass
    price_eur: float
    seats_available: int


# --- Booking creation ---------------------------------------------------- #

class BookingCreateRequest(BaseModel):
    flight_id: int = Field(..., description="ID of the flight to book")
    passenger_name: str = Field(..., min_length=1, max_length=120)
    passenger_email: EmailStr
    seat_preference: Optional[SeatPreference] = None


class BookingAddonResponse(BaseModel):
    id: int
    addon_type: AddonType
    description: str
    fee_eur: float
    created_at: str


class BookingResponse(BaseModel):
    """Full booking details, returned on create / get / mutate."""
    reference: str
    flight_id: int
    flight_number: str
    origin: str
    destination: str
    departure_time: str
    passenger_name: str
    passenger_email: str
    seat: Optional[str] = None
    seat_preference: Optional[str] = None
    fare_class: FareClass
    price_paid_eur: float
    status: BookingStatus
    created_at: str
    cancelled_at: Optional[str] = None
    addons: list[BookingAddonResponse] = []


# --- Booking mutations --------------------------------------------------- #

class BookingRescheduleRequest(BaseModel):
    new_flight_id: int = Field(..., description="ID of the new flight")


class BookingAddonRequest(BaseModel):
    addon_type: AddonType
    description: str = Field(..., min_length=1, max_length=200)
    fee_eur: float = Field(..., ge=0, description="Fee in EUR (>=0)")


# --- Knowledge base ------------------------------------------------------ #

class KnowledgeTopicResponse(BaseModel):
    """A single KB topic. The 'rules' dict is freeform per topic."""
    topic: str
    summary: str
    rules: dict
    details: str


# --- Errors -------------------------------------------------------------- #

class ErrorResponse(BaseModel):
    """Standard error envelope. FastAPI uses this when a route raises HTTPException."""
    detail: str