"""
Customer simulator: Claude Haiku plays the customer, ElevenLabs agent
plays the airline rep. We orchestrate the turn-taking until the customer
ends the call or we hit a turn cap.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from anthropic import Anthropic
from dotenv import load_dotenv

from agent.elevenlabs_client import TextConversation
from refinement.scenarios import Scenario

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
SIMULATOR_MODEL = os.getenv("SIMULATOR_MODEL", "claude-haiku-4-5")

if not ANTHROPIC_API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY is missing from .env")

_anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)

END_CALL_TOKEN = "[END_CALL]"

SIMULATOR_INSTRUCTIONS = """\
You are roleplaying a customer calling an airline. Stay in character. Keep your replies short — one or two sentences, like a real phone call. Do not narrate ("the customer says..."), just speak as the customer in first person.

When you are satisfied that the agent has fully handled your request — they confirmed your booking, or answered all your policy questions, or clearly explained why they cannot help — end the call by replying with a brief polite closing followed by the literal token [END_CALL] on the same line.

Examples:
  "Great, thanks for your help. [END_CALL]"
  "Thank you, that's all I needed to know. Goodbye. [END_CALL]"

Do NOT use [END_CALL] until your needs are met or it is clear the agent cannot help. Stay in character throughout.
"""


@dataclass
class SimulationResult:
    scenario_id: str
    transcript: list[dict] = field(default_factory=list)
    completed: bool = False  # customer ended naturally with [END_CALL]
    turn_count: int = 0
    error: str | None = None


def _next_customer_turn(scenario: Scenario, transcript: list[dict]) -> str:
    """Ask Claude Haiku what the customer says next, given the conversation so far."""
    # Convert our transcript into Anthropic message format.
    # Customer = "assistant" role here (the LLM IS the customer).
    # Agent     = "user" role here (the LLM is responding to the agent's words).
    messages = []
    for entry in transcript:
        role = entry.get("role")
        msg = entry.get("message", "").strip()
        if not msg:
            continue
        if role == "agent":
            messages.append({"role": "user", "content": msg})
        elif role == "user":
            messages.append({"role": "assistant", "content": msg})

    # If we have no agent messages yet, prime with the agent's first message
    # if any; otherwise the model needs at least one user turn to respond.
    if not messages or messages[0]["role"] != "user":
        messages.insert(0, {"role": "user", "content": "(call connected)"})

    system_prompt = f"{SIMULATOR_INSTRUCTIONS}\n\nYour scenario:\n{scenario.persona}"

    resp = _anthropic.messages.create(
        model=SIMULATOR_MODEL,
        max_tokens=200,
        system=system_prompt,
        messages=messages,
    )
    return resp.content[0].text.strip()


def run_scenario(scenario: Scenario,
                 max_turns: int = 12,
                 settle_seconds: float = 2.0) -> SimulationResult:
    """
    Run one scenario end-to-end.

    Flow:
      1. Open a TextConversation with the agent (uses dashboard config + pushed prompt)
      2. Send the scenario's deterministic first_message
      3. Loop:
         - Read agent's reply
         - Ask Haiku for the next customer turn
         - If [END_CALL] → stop
         - Else → send to agent, repeat
      4. Close the conversation cleanly

    Returns a SimulationResult with the full transcript.
    """
    result = SimulationResult(scenario_id=scenario.id)
    conv = TextConversation()

    try:
        conv.start()
        # Wait briefly so the agent's "first_message" lands in the transcript.
        time.sleep(settle_seconds)

        # Send the scenario's first message and get the agent's first real reply.
        conv.send(scenario.first_message)
        result.turn_count += 1

        for _ in range(max_turns - 1):
            # Hand the conversation to the simulator to produce the next customer turn.
            customer_text = _next_customer_turn(scenario, conv.transcript)

            if END_CALL_TOKEN in customer_text:
                # Strip the token, send the closing message, and stop.
                closing = customer_text.replace(END_CALL_TOKEN, "").strip()
                if closing:
                    conv.send(closing)
                result.completed = True
                break

            conv.send(customer_text)
            result.turn_count += 1
        else:
            # Loop ended via for-else: hit the turn cap.
            result.completed = False

    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
    finally:
        try:
            conv.end()
        except Exception:
            pass

        # Use ElevenLabs' authoritative transcript (includes tool events).
        # Fall back to our in-memory transcript if the fetch fails.
        try:
            from agent.elevenlabs_client import fetch_authoritative_transcript
            if conv.conversation_id:
                result.transcript = fetch_authoritative_transcript(conv.conversation_id)
            else:
                result.transcript = conv.transcript
        except Exception as e:
            result.transcript = conv.transcript
            if not result.error:
                result.error = f"transcript_fetch_failed: {type(e).__name__}: {e}"
    return result


if __name__ == "__main__":
    # Smoke test: run the easiest scenario (a policy inquiry — no booking side-effects).
    from refinement.scenarios import get_by_id

    scenario = get_by_id("baggage_inquiry")
    print(f"Running scenario: {scenario.id}")
    print(f"Title: {scenario.title}\n")

    result = run_scenario(scenario, max_turns=8)

    print(f"\n=== TRANSCRIPT ({len(result.transcript)} entries) ===\n")
    for entry in result.transcript:
        role = entry.get("role", "?")
        if role in ("agent", "user", "system"):
            print(f"[{role.upper()}] {entry.get('message', '')}\n")
        elif role == "tool_call":
            print(f"[TOOL_CALL] {entry.get('tool_name')}({entry.get('params')})\n")
        elif role == "tool_result":
            err = " (ERROR)" if entry.get("is_error") else ""
            preview = (entry.get("result") or "")[:200]
            print(f"[TOOL_RESULT{err}] {entry.get('tool_name')} → {preview}...\n")

    print("=" * 50)
    print(f"Completed naturally: {result.completed}")
    print(f"Turn count: {result.turn_count}")
    if result.error:
        print(f"Error: {result.error}")