"""
The 10 customer scenarios from the assessment, plus a few fields that
help downstream code:
  - id: stable string used for logging
  - title: short human-readable name
  - persona: the prompt Claude Haiku uses to play the customer
  - first_message: what the customer says to open the call
  - expected_tools: tools the agent SHOULD call to handle this scenario
                    (used by the evaluator as a hint, not a hard requirement)
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    persona: str
    first_message: str
    expected_tools: tuple[str, ...]


SCENARIOS: list[Scenario] = [
    Scenario(
        id="book_next_to_destination",
        title="Book the next available flight to a destination",
        persona=(
            "You need to fly from Athens to Paris (CDG) as soon as possible. "
            "When the agent gives you the next available flight, push back and "
            "ask whether there is anything earlier. Only accept the booking "
            "after the agent has confirmed it really is the soonest option. "
            "Your name is {{NAME}} and your email is {{EMAIL}}. "
            "Be polite. Once the agent confirms a booking and gives you a "
            "reference, thank them and end the call."
        ),
        first_message="Hi, I need to book the next available flight to Paris.",
        expected_tools=("search_flights", "book_flight"),
    ),
    Scenario(
        id="cheapest_within_week",
        title="Find and book the cheapest tickets within next week",
        persona=(
            "You are calling Aegis Airlines to book the cheapest flight you can "
            "find anytime within the next 7 days. You don't have a specific "
            "destination in mind — you're looking for the cheapest deal "
            "regardless of where it goes, as long as it's within the next "
            "week. If the agent asks where you want to fly, tell them you're "
            "open to any destination they offer. "
            "IMPORTANT: This is a phone call. If the agent reads out a long "
            "list of every flight to every city, react like a real customer "
            "would on the phone — say something like 'wait, that's a lot to "
            "take in, can you just tell me some of the cheapest ones?' Push "
            "back if the agent over-explains or dumps too much information "
            "in one turn. "
            "Your name is {{NAME}}, email {{EMAIL}}. Be "
            "polite. Once the agent confirms a booking and gives you a "
            "reference, thank them and end the call."
        ),
        first_message="Hello, what's the cheapest flight you have available in the next week?",
        expected_tools=("search_flights", "book_flight"),
    ),
    Scenario(
        id="pet_policy_inquiry",
        title="Ask about pet travel policy",
        persona=(
            "You are calling to ask about Aegis Airlines' pet policy. You have a "
            "small dog (5kg) and want to know if you can bring it in the cabin, "
            "what it would cost, and whether you need to book it in advance. "
            "You are NOT booking anything yet — just gathering information. "
            "Once the agent has answered all your questions clearly, thank them "
            "and end the call."
        ),
        first_message="Hi, I have a small dog and I'm wondering if I can take her with me on a flight.",
        expected_tools=("get_policy",),
    ),
    Scenario(
        id="reschedule_booking",
        title="Reschedule an existing booking",
        persona=(
            "You have an existing Aegis Airlines booking and you want to move it "
            "to a different day. You don't have your booking reference handy at "
            "first — if the agent asks, say 'I'm not sure, can you find it some "
            "other way?' to test how the agent handles missing information. "
            "If pushed, eventually 'remember' your reference is AGX-DEMO1 and "
            "say it. (The agent should find this reference and should "
            "explain that.) Be patient and polite. End the call after the agent "
            "has clearly explained next steps."
        ),
        first_message="Hi, I'd like to reschedule my flight to a different day.",
        expected_tools=("get_booking",),
    ),
    Scenario(
        id="baggage_inquiry",
        title="Ask about baggage allowance and excess fees",
        persona=(
            "You are calling to find out Aegis Airlines' baggage rules. You want "
            "to know: how heavy can a cabin bag be, how many checked bags are "
            "included with economy, and what excess weight costs per kilo. You "
            "are NOT booking anything. Once the agent has answered all three "
            "questions, thank them and end the call."
        ),
        first_message="Hi, can you tell me about your baggage rules?",
        expected_tools=("get_policy",),
    ),
    Scenario(
        id="cancel_refund",
        title="Cancel an existing booking",
        persona=(
            "You want to cancel a booking. Your reference is AGX-DEMO2. (The "
            "agent should find this reference and should explain that "
            "honestly.) Ask about the refund timing if cancelled. Be polite. "
            "End the call once the agent has explained the situation clearly."
        ),
        first_message="Hi, I need to cancel my booking please.",
        expected_tools=("get_booking", "get_policy"),
    ),
    Scenario(
        id="seat_preference",
        title="Book a flight with a specific seat preference",
        persona=(
            "You want to book a flight from Athens to Rome (FCO) sometime next "
            "week, in economy, and you'd like a window seat. Your name is "
            "{{NAME}}, email {{EMAIL}}. Be polite and "
            "concise. Once the agent confirms the booking with the seat "
            "preference noted, thank them and end the call."
        ),
        first_message="Hi, I want to book a flight to Rome and I'd like a window seat.",
        expected_tools=("search_flights", "book_flight"),
    ),
    Scenario(
        id="add_extra_bag",
        title="Add an extra bag or special item to an existing booking",
        persona=(
            "You have a booking (reference AGX-DEMO3) and you want to add an "
            "extra checked bag. (The agent should find this reference and "
            "should explain that.) Ask what an extra bag would cost so you "
            "have the info for next time. Be polite. End the call after the "
            "agent has explained both the missing-booking issue and the "
            "extra-bag fee policy."
        ),
        first_message="Hi, I'd like to add an extra bag to my booking.",
        expected_tools=("get_booking", "get_policy"),
    ),
    Scenario(
        id="checkin_status",
        title="Ask about check-in times and gate info",
        persona=(
            "You have a flight tomorrow and you want to know: when does online "
            "check-in open, when does the airport counter close, and how do you "
            "find your gate. You don't have a booking reference to give — you "
            "just want general policy information. End the call once the agent "
            "has answered all three questions."
        ),
        first_message="Hi, I have a flight tomorrow. When does check-in open?",
        expected_tools=("get_policy",),
    ),
    Scenario(
        id="special_assistance",
        title="Request assistance for a passenger with reduced mobility",
        persona=(
            "You are calling on behalf of your elderly father, Nikos Stavros, "
            "who has reduced mobility and needs wheelchair assistance at the "
            "airport. He has a flight booked already (reference AGX-DEMO4 — "
            "the agent should find this reference and should explain that) but you want to add the wheelchair "
            "request. Ask what the procedure is, whether there is a fee, and "
            "how far in advance you must request it. End the call once the "
            "agent has clearly explained the policy."
        ),
        first_message="Hello, my father needs wheelchair assistance for his upcoming flight. How do I arrange that?",
        expected_tools=("get_policy",),
    ),
]


def get_by_id(scenario_id: str) -> Scenario:
    for s in SCENARIOS:
        if s.id == scenario_id:
            return s
    raise KeyError(f"No scenario with id {scenario_id!r}")


if __name__ == "__main__":
    for s in SCENARIOS:
        print(f"  - {s.id}: {s.title}")
    print(f"\nTotal: {len(SCENARIOS)} scenarios")