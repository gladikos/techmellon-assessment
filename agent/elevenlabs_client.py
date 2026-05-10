"""
ElevenLabs client: prompt updates + text-mode conversations.

Uses the ElevenLabs SDK's Conversation class over WebSocket. Real tool
calls fire to our backend webhooks (no mocking).
"""

from __future__ import annotations

import os
from tabnanny import verbose
import threading
from typing import Any, Callable, Optional

import httpx
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs.conversational_ai.conversation import Conversation
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

ELEVENLABS_BASE = "https://api.elevenlabs.io/v1"
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_AGENT_ID = os.getenv("ELEVENLABS_AGENT_ID")

if not ELEVENLABS_API_KEY:
    raise RuntimeError("ELEVENLABS_API_KEY is missing from .env")
if not ELEVENLABS_AGENT_ID:
    raise RuntimeError("ELEVENLABS_AGENT_ID is missing from .env")


# --- Agent prompt update via REST --------------------------------------- #

def _headers() -> dict:
    return {"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"}


def _request(method: str, path: str, *, json: Optional[dict] = None,
             timeout: float = 60.0) -> dict:
    url = f"{ELEVENLABS_BASE}{path}"
    with httpx.Client(timeout=timeout) as client:
        resp = client.request(method, url, headers=_headers(), json=json)
        if resp.status_code >= 400:
            print(f"\n[ElevenLabs {resp.status_code}] {resp.text}\n")
        resp.raise_for_status()
        return resp.json() if resp.text else {}


@retry(stop=stop_after_attempt(3),
       wait=wait_exponential(multiplier=1, min=1, max=10))
def get_agent_config() -> dict:
    return _request("GET", f"/convai/agents/{ELEVENLABS_AGENT_ID}")


@retry(stop=stop_after_attempt(3),
       wait=wait_exponential(multiplier=1, min=1, max=10))
def update_agent_config(prompt: str,
                        first_message: Optional[str] = None,
                        tools: Optional[list[dict]] = None) -> dict:
    """
    Update the agent's prompt on ElevenLabs.

    Tools are configured in the dashboard one-time; the `tools` arg is
    accepted for forward compatibility but ignored.
    """
    payload: dict[str, Any] = {
        "conversation_config": {
            "agent": {
                "prompt": {"prompt": prompt}
            }
        }
    }
    if first_message is not None:
        payload["conversation_config"]["agent"]["first_message"] = first_message

    return _request("PATCH", f"/convai/agents/{ELEVENLABS_AGENT_ID}",
                    json=payload)


# --- Text-mode conversation via SDK ------------------------------------- #

class TextConversation:
    """
    Thin wrapper around the ElevenLabs SDK's Conversation class for
    text-only conversations. Captures every turn and tool event into a
    transcript the orchestrator can read.

    Usage:
        conv = TextConversation()
        conv.start()
        conv.send("Hi, I want to fly to London.")
        # ... wait, then send next user turn ...
        conv.end()
        transcript = conv.transcript
    """

    def __init__(self, verbose: bool = False,
                 on_turn: Optional[Callable[[dict], None]] = None) -> None:
        self.transcript: list[dict] = []
        self.conversation_id: Optional[str] = None
        self.verbose = verbose
        self.on_turn = on_turn
        self._agent_response_event = threading.Event()
        self._client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        self._conversation: Optional[Conversation] = None

    def _on_agent_response(self, text: str) -> None:
        self.transcript.append({"role": "agent", "message": text})
        if self.verbose:
            print(f"\n  [AGENT] {text}", flush=True)
        if self.on_turn:
            try:
                self.on_turn({"role": "agent", "message": text})
            except Exception:
                pass
        self._agent_response_event.set()

    def _on_user_transcript(self, text: str) -> None:
        if not self.transcript or self.transcript[-1].get("message") != text:
            self.transcript.append({"role": "user", "message": text})
            if self.verbose:
                print(f"\n  [USER]  {text}", flush=True)
            if self.on_turn:
                try:
                    self.on_turn({"role": "user", "message": text})
                except Exception:
                    pass

    def start(self) -> None:
        self._conversation = Conversation(
            client=self._client,
            agent_id=ELEVENLABS_AGENT_ID,
            requires_auth=False,
            audio_interface=None,  # text-only
            callback_agent_response=self._on_agent_response,
            callback_user_transcript=self._on_user_transcript,
        )
        self._conversation.start_session()

    def send(self, text: str, idle_seconds: float = 4.0,
             max_wait_seconds: float = 60.0) -> str:
        """
        Send one user message and wait for the agent's full reply.

        The agent may emit multiple agent_response events per turn (e.g.
        "Let me search..." then later "I found 3 flights..."). We wait
        until the agent has been silent for `idle_seconds` before treating
        the turn as complete, capped by `max_wait_seconds`.
        """
        if self._conversation is None:
            raise RuntimeError("Call start() first")
        self.transcript.append({"role": "user", "message": text})
        if self.verbose:
            print(f"\n  [USER]  {text}", flush=True)
        if self.on_turn:
            try:
                self.on_turn({"role": "user", "message": text})
            except Exception:
                pass

        # Mark how many agent messages we've seen before this turn.
        agent_messages_before = sum(
            1 for e in self.transcript if e.get("role") == "agent"
        )

        self._agent_response_event.clear()
        self._conversation.send_user_message(text)

        import time
        start = time.time()
        last_count = agent_messages_before
        last_change = time.time()

        while time.time() - start < max_wait_seconds:
            time.sleep(0.5)
            current_count = sum(
                1 for e in self.transcript if e.get("role") == "agent"
            )
            if current_count > last_count:
                last_count = current_count
                last_change = time.time()
            elif time.time() - last_change >= idle_seconds and current_count > agent_messages_before:
                break  # agent has gone quiet

        # Concatenate all agent messages since this turn started.
        new_messages = [
            e["message"] for e in self.transcript[-(last_count - agent_messages_before):]
            if e.get("role") == "agent"
        ]
        if not new_messages:
            self.transcript.append({
                "role": "system",
                "message": f"(no agent response within {max_wait_seconds}s)"
            })
            return ""
        return " ".join(new_messages)

    def end(self) -> None:
        if self._conversation is not None:
            self._conversation.end_session()
            self.conversation_id = self._conversation.wait_for_session_end()
            self._conversation = None

def fetch_authoritative_transcript(conversation_id: str,
                                    max_wait_seconds: float = 60.0,
                                    poll_interval: float = 1.5) -> list[dict]:
    """
    Fetch the conversation's full server-side transcript from ElevenLabs
    and return it in our flat, evaluator-friendly format.

    Polls until the conversation status is 'done' (post-processing
    complete), then translates ElevenLabs' nested transcript schema
    into a flat list of:
      - {"role": "agent", "message": "..."}
      - {"role": "user", "message": "..."}
      - {"role": "tool_call", "tool_name": "...", "params": "..."}
      - {"role": "tool_result", "tool_name": "...", "result": "...", "is_error": bool}

    This is the ground-truth transcript: it includes every tool event
    that fired server-side, in the correct order. Use this as the
    transcript fed to the evaluator.

    Raises:
        TimeoutError: if ElevenLabs does not finish post-processing
            within max_wait_seconds. Callers should fall back to the
            in-memory transcript captured during the live conversation.
        RuntimeError: if ElevenLabs reports status='failed'.
    """
    import time as _time

    # Poll until ElevenLabs finishes processing.
    deadline = _time.time() + max_wait_seconds
    data: dict = {}
    last_status: Optional[str] = None
    saw_done = False
    while _time.time() < deadline:
        data = _request("GET", f"/convai/conversations/{conversation_id}")
        last_status = data.get("status")
        if last_status == "done":
            saw_done = True
            break
        if last_status == "failed":
            raise RuntimeError(
                f"ElevenLabs reported status='failed' for conversation {conversation_id}"
            )
        _time.sleep(poll_interval)

    if not saw_done:
        # Timed out waiting for post-processing. Better to surface this as
        # an exception than to silently return an empty/partial transcript.
        print(
            f"[fetch_authoritative_transcript] Timed out after {max_wait_seconds}s "
            f"waiting for conversation {conversation_id} (last status: {last_status}). "
            f"Caller will fall back to in-memory transcript."
        )
        raise TimeoutError(
            f"ElevenLabs post-processing did not complete within {max_wait_seconds}s "
            f"for conversation {conversation_id} (last status: {last_status})"
        )

    # Translate ElevenLabs' transcript turns into our flat format.
    flat: list[dict] = []
    for turn in data.get("transcript", []):
        role = turn.get("role")
        message = turn.get("message")
        tool_calls = turn.get("tool_calls") or []
        tool_results = turn.get("tool_results") or []

        # In ElevenLabs' schema, tool_calls and tool_results live in their
        # own turns with role='agent' and message=null. Plain text turns
        # have a non-null message and empty tool arrays.
        if message:
            flat.append({"role": role, "message": message})

        for call in tool_calls:
            flat.append({
                "role": "tool_call",
                "tool_name": call.get("tool_name", ""),
                "params": call.get("params_as_json", ""),
            })

        for result in tool_results:
            flat.append({
                "role": "tool_result",
                "tool_name": result.get("tool_name", ""),
                "result": str(result.get("result_value", "")),
                "is_error": bool(result.get("is_error")),
            })

    return flat