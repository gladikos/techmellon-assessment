"""
Tool definitions for the Aegis Airlines agent on ElevenLabs.

Each entry describes one of our backend endpoints in the format
ElevenLabs' agent platform expects. The orchestrator reads this list and
pushes it to ElevenLabs whenever it (re)configures the agent.

Design rules:
- One tool per backend endpoint. Granular tools make the agent's choices
  easier to debug when it picks the wrong action.
- Names are snake_case verbs.
- Descriptions matter most: the LLM picks tools by description.
- Path params use {curly_braces} in the URL.
- All tools include the ngrok-skip-browser-warning header so ElevenLabs
  bypasses the free-tier splash page.

Schema notes (learned from API validation):
- request_headers: a flat dict {name: value}, NOT a list of header objects.
- path_params_schema, query_params_schema, request_body_schema: all use
  the same shape — {"properties": [ {param}, {param}, ... ]}.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://localhost:8000")


COMMON_HEADERS = {
    "ngrok-skip-browser-warning": "true",
    "Content-Type": "application/json",
}


def _param(id: str, type: str, description: str, required: bool = True) -> dict:
    """Helper: build a single parameter spec in the format ElevenLabs expects."""
    return {
        "id": id,
        "type": type,
        "description": description,
        "dynamic_variable": "",
        "constant_value": "",
        "required": required,
        "value_type": "llm_prompt",
    }


def _build_tools() -> list[dict]:
    return [
        # --- Knowledge base ---------------------------------------------- #
        {
            "type": "webhook",
            "name": "get_policy",
            "description": (
                "Look up a specific airline policy. Use this whenever the customer "
                "asks about pets, baggage, refunds, cancellations, rescheduling, "
                "check-in, flight status, special assistance, special items, or "
                "seating policy. Pass the topic name. Returns summary, structured "
                "rules, and a full natural-language description."
            ),
            "api_schema": {
                "url": f"{BACKEND_BASE_URL}/knowledge/{{topic}}",
                "method": "GET",
                "request_headers": COMMON_HEADERS,
                "path_params_schema": {
                    "properties": [
                        _param("topic", "string",
                               "One of: pets, baggage, special_items, special_assistance, "
                               "check_in, flight_status, cancellation, rescheduling, seating."),
                    ],
                },
            },
        },

        # --- Flight search ----------------------------------------------- #
        {
            "type": "webhook",
            "name": "search_flights",
            "description": (
                "Search the Aegis Airlines flight catalog. Use this when the customer "
                "wants to find or book a flight, asks about prices, availability, "
                "or wants the cheapest/next option. Required: destination (3-letter "
                "IATA code, e.g. CDG, LHR, FCO, BCN, IST). Optional: date "
                "(YYYY-MM-DD), max_price (EUR), fare_class. Returns up to 20 flights."
            ),
            "api_schema": {
                "url": f"{BACKEND_BASE_URL}/flights/search",
                "method": "GET",
                "request_headers": COMMON_HEADERS,
                "query_params_schema": {
                    "properties": [
                        _param("destination", "string",
                               "3-letter IATA airport code (uppercase).",
                               required=True),
                        _param("date", "string",
                               "ISO date YYYY-MM-DD (optional).",
                               required=False),
                        _param("max_price", "number",
                               "Maximum price in EUR (optional).",
                               required=False),
                        _param("fare_class", "string",
                               "One of: economy_light, economy, economy_flex, "
                               "business, business_flex (optional).",
                               required=False),
                    ],
                },
            },
        },

        # --- Booking creation -------------------------------------------- #
        {
            "type": "webhook",
            "name": "book_flight",
            "description": (
                "Create a booking on a specific flight. Call this ONLY after the "
                "customer has confirmed which flight they want and provided their "
                "name and email. Returns a booking reference like AGX-7K2P9 — read "
                "it back to the customer. Returns 409 if the flight is fully booked; "
                "in that case suggest alternatives via search_flights."
            ),
            "api_schema": {
                "url": f"{BACKEND_BASE_URL}/bookings",
                "method": "POST",
                "request_headers": COMMON_HEADERS,
                "request_body_schema": {
                    "type": "object",
                    "properties": [
                        _param("flight_id", "integer",
                               "ID of the flight to book (from search_flights results)."),
                        _param("passenger_name", "string",
                               "Full passenger name as on travel document."),
                        _param("passenger_email", "string",
                               "Passenger email for booking confirmation."),
                        _param("seat_preference", "string",
                               "Optional: window, aisle, middle, extra_legroom, or exit_row.",
                               required=False),
                    ],
                },
            },
        },

        # --- Booking retrieval ------------------------------------------- #
        {
            "type": "webhook",
            "name": "get_booking",
            "description": (
                "Retrieve an existing booking by its reference (e.g. AGX-7K2P9). "
                "Use this whenever the customer wants details about an existing "
                "booking, or before performing any action that requires booking "
                "context (cancel, reschedule, add baggage, etc)."
            ),
            "api_schema": {
                "url": f"{BACKEND_BASE_URL}/bookings/{{reference}}",
                "method": "GET",
                "request_headers": COMMON_HEADERS,
                "path_params_schema": {
                    "properties": [
                        _param("reference", "string",
                               "The booking reference, format AGX-XXXXX."),
                    ],
                },
            },
        },

        # --- Booking cancellation ---------------------------------------- #
        {
            "type": "webhook",
            "name": "cancel_booking",
            "description": (
                "Cancel an existing booking and return the seat to inventory. "
                "Confirm the customer's intent before calling. After cancellation, "
                "explain refund timing per the cancellation policy (use get_policy "
                "for exact figures)."
            ),
            "api_schema": {
                "url": f"{BACKEND_BASE_URL}/bookings/{{reference}}/cancel",
                "method": "POST",
                "request_headers": COMMON_HEADERS,
                "path_params_schema": {
                    "properties": [
                        _param("reference", "string",
                               "The booking reference, format AGX-XXXXX."),
                    ],
                },
            },
        },

        # --- Booking reschedule ------------------------------------------ #
        {
            "type": "webhook",
            "name": "reschedule_booking",
            "description": (
                "Move a booking to a different flight. Use search_flights first "
                "to find a candidate, confirm with the customer, then call this. "
                "Pass the booking reference and the new flight_id."
            ),
            "api_schema": {
                "url": f"{BACKEND_BASE_URL}/bookings/{{reference}}/reschedule",
                "method": "POST",
                "request_headers": COMMON_HEADERS,
                "path_params_schema": {
                    "properties": [
                        _param("reference", "string",
                               "The booking reference, format AGX-XXXXX."),
                    ],
                },
                "request_body_schema": {
                    "type": "object",
                    "properties": [
                        _param("new_flight_id", "integer",
                               "ID of the new flight (from search_flights results)."),
                    ],
                },
            },
        },

        # --- Booking addons ---------------------------------------------- #
        {
            "type": "webhook",
            "name": "add_booking_addon",
            "description": (
                "Attach an addon to an existing booking: extra baggage, pet, sports "
                "equipment, musical instrument, special assistance, pram, or "
                "wheelchair. Use get_policy to confirm fees and rules before "
                "calling. Pass the addon_type, a short description, and the fee."
            ),
            "api_schema": {
                "url": f"{BACKEND_BASE_URL}/bookings/{{reference}}/addons",
                "method": "POST",
                "request_headers": COMMON_HEADERS,
                "path_params_schema": {
                    "properties": [
                        _param("reference", "string",
                               "The booking reference, format AGX-XXXXX."),
                    ],
                },
                "request_body_schema": {
                    "type": "object",
                    "properties": [
                        _param("addon_type", "string",
                               "One of: baggage, pet, sports_equipment, "
                               "musical_instrument, special_assistance, pram, wheelchair."),
                        _param("description", "string",
                               "Short human-readable description, e.g. 'Extra hold bag, 23kg'."),
                        _param("fee_eur", "number",
                               "Fee in EUR. Look up via get_policy if unsure."),
                    ],
                },
            },
        },
    ]


TOOLS = _build_tools()


if __name__ == "__main__":
    print(f"Defined {len(TOOLS)} tools:")
    for t in TOOLS:
        print(f"  - {t['name']}: {t['api_schema']['method']} {t['api_schema']['url']}")