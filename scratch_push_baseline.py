"""
Push the baseline prompt to the ElevenLabs agent, then run a real
simulated conversation so we can see the agent + tools + backend
working end to end.

Run with: python scratch_push_baseline.py
"""

import json

from agent.elevenlabs_client import (
    update_agent_config,
    simulate_conversation,
    transcript_from_simulation,
)
from agent.system_prompt import BASELINE_SYSTEM_PROMPT, BASELINE_VERSION


# --- Step 1: push the baseline prompt ----------------------------------- #

print(f"Pushing baseline ({BASELINE_VERSION}) to ElevenLabs...")
print(f"  Prompt: {len(BASELINE_SYSTEM_PROMPT)} chars")

update_agent_config(
    prompt=BASELINE_SYSTEM_PROMPT,
    first_message="Thank you for calling Aegis Airlines. How can I help you today?",
)
print("✅ Prompt pushed.\n")


# --- Step 2: run a real text-mode simulation ---------------------------- #

# A simple persona — easy scenario, exercises one tool (search_flights).
persona = (
    "You are calling Aegis Airlines because you want to fly from Athens to "
    "London (LHR) sometime next week. You'd prefer the cheapest available "
    "option in economy. Your name is Maria Papadopoulou and your email is "
    "maria.p@example.com. Be polite and concise. After the agent confirms "
    "the booking and gives you a reference, thank them and end the call."
)

print("Running simulated conversation...")
print("(This takes 30-90 seconds; ElevenLabs runs the whole exchange server-side)\n")

result = simulate_conversation(
    simulated_user_persona=persona,
    first_message="Hi, I'd like to book a flight to London.",
    new_turns_limit=20,
)

print("✅ Simulation complete.\n")


# --- Step 3: print a clean transcript ----------------------------------- #

transcript = transcript_from_simulation(result)

print("=" * 70)
print("TRANSCRIPT")
print("=" * 70)

for entry in transcript:
    role = entry.get("role")
    if role in ("user", "agent"):
        speaker = "CUSTOMER" if role == "user" else "AGENT   "
        print(f"\n[{speaker}] {entry.get('message', '')}")
    elif role == "tool_call":
        params = entry.get("params", "")
        print(f"\n  → tool_call: {entry['tool_name']}({params})")
    elif role == "tool_result":
        marker = "✗" if entry.get("is_error") else "✓"
        result_str = (entry.get("result") or "")[:120]
        print(f"  {marker} tool_result: {entry['tool_name']} → {result_str}...")

print("\n" + "=" * 70)
print(f"Total turns: {len(transcript)}")

# Also dump the full raw JSON to a file in case we want to inspect later.
with open("scratch_last_transcript.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)
print("Full raw response saved to scratch_last_transcript.json")