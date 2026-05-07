"""
Evaluator: an LLM that scores a conversation transcript and classifies
each failure as prompt-fix-able or code-fix-able.

Output is structured JSON so the orchestrator can route fixes
mechanically. The evaluator is a Sonnet-class model because this is
where reasoning quality matters most.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Literal

from anthropic import Anthropic
from dotenv import load_dotenv

from refinement.scenarios import Scenario

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
EVALUATOR_MODEL = os.getenv("EVALUATOR_MODEL", "claude-sonnet-4-6")

if not ANTHROPIC_API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY is missing from .env")

_anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)


# --- Result types -------------------------------------------------------- #

RootCause = Literal["prompt", "code"]


@dataclass
class Failure:
    criterion: str
    quote: str
    issue: str
    root_cause: RootCause
    suggested_fix: str


@dataclass
class EvaluationResult:
    scenario_id: str
    scores: dict[str, int] = field(default_factory=dict)
    failures: list[Failure] = field(default_factory=list)
    overall_pass: bool = False
    summary: str = ""
    raw_response: str = ""

    @property
    def min_score(self) -> int:
        return min(self.scores.values()) if self.scores else 0

    @property
    def has_prompt_failures(self) -> bool:
        return any(f.root_cause == "prompt" for f in self.failures)

    @property
    def has_code_failures(self) -> bool:
        return any(f.root_cause == "code" for f in self.failures)


# --- Prompt -------------------------------------------------------------- #

EVALUATOR_SYSTEM_PROMPT = """\
You are an expert evaluator of airline customer service AI agents.

You will be given:
1. A scenario describing what the customer wanted
2. A transcript of a real conversation between a simulated customer and the airline's voice agent
3. (Optionally) the list of tools the agent SHOULD have called

Your job is to score the conversation across four criteria, identify specific failures with quoted evidence, and classify the root cause of each failure.

## Scoring criteria (1-10 each)

- **understanding**: Did the agent correctly identify the customer's request and any details they provided?
- **tool_use**: Did the agent call the right tools with correct parameters? (Read the [TOOL_CALL] and [TOOL_RESULT] lines in the transcript.)
- **outcome**: Did the agent complete the right action and confirm it clearly to the customer?
- **conversation_quality**: Was the agent natural, concise, professional? No repetition, no robotic phrasing, no piling many questions into one turn?

A score of 10 means "ideal." 8 means "good, minor improvement possible." 5 means "noticeably wrong but conversation continued." 1-3 means "major failure."

## Root-cause classification

For EVERY failure (anywhere a score is below 8), classify the root cause:

- **prompt**: The agent's instructions were inadequate. The agent had everything it needed (working tools, accurate data) but made a poor decision a better-instructed agent would not make. Examples: poor phrasing, repeated openers, didn't confirm key details before acting, gave up too quickly, hallucinated numbers it should have looked up, asked too many questions at once.

- **code**: A backend tool returned wrong data, an API errored unexpectedly, a tool response was malformed, or a backend behavior was incorrect. Examples: search_flights returned empty results when flights exist, book_flight returned an unexpected 500, knowledge endpoint returned wrong fields, an endpoint validation rule was too strict.

When in doubt, look at the TOOL_RESULT lines:
- If the tool returned plausible data and the agent still failed to use it well → **prompt**
- If the tool errored or returned obviously wrong data → **code**

## Output format

Reply with ONLY a single JSON object, no prose before or after, no markdown fences. Schema:

{
  "scores": {
    "understanding": <int 1-10>,
    "tool_use": <int 1-10>,
    "outcome": <int 1-10>,
    "conversation_quality": <int 1-10>
  },
  "failures": [
    {
      "criterion": "<one of: understanding, tool_use, outcome, conversation_quality>",
      "quote": "<exact text from the transcript that demonstrates the failure>",
      "issue": "<one-sentence description of what went wrong>",
      "root_cause": "<prompt OR code>",
      "suggested_fix": "<one concise sentence describing the fix>"
    }
  ],
  "summary": "<one to two sentences summarizing the conversation>"
}

If there are no failures (all scores >= 8), return an empty failures list.
"""


# --- Helpers ------------------------------------------------------------- #

def _format_transcript(transcript: list[dict]) -> str:
    """Render a transcript as a single block of plain text the LLM can read."""
    lines = []
    for entry in transcript:
        role = entry.get("role", "?")
        msg = entry.get("message", "")
        if role == "agent":
            lines.append(f"[AGENT] {msg}")
        elif role == "user":
            lines.append(f"[CUSTOMER] {msg}")
        elif role == "tool_call":
            tool = entry.get("tool_name", "?")
            params = entry.get("params", "")
            lines.append(f"[TOOL_CALL] {tool}({params})")
        elif role == "tool_result":
            tool = entry.get("tool_name", "?")
            result = entry.get("result", "")
            err = " (ERROR)" if entry.get("is_error") else ""
            lines.append(f"[TOOL_RESULT{err}] {tool} → {result}")
        elif role == "system":
            lines.append(f"[SYSTEM] {msg}")
    return "\n".join(lines)


def _extract_json(text: str) -> dict:
    """
    Pull the first {...} JSON object out of the model's reply.
    Tolerates accidental code fences or trailing prose.
    """
    # If wrapped in ```...``` fences, strip them.
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        return json.loads(fenced.group(1))

    # Otherwise find the first balanced JSON object.
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found in evaluator response")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("Unbalanced JSON in evaluator response")


# --- Main entry point --------------------------------------------------- #

def evaluate(scenario: Scenario, transcript: list[dict],
             pass_threshold: int = 8) -> EvaluationResult:
    """Score a transcript and classify any failures by root cause."""
    rendered = _format_transcript(transcript)

    user_message = f"""\
## SCENARIO ({scenario.id})

{scenario.title}

Customer's intent (from the simulator's persona):
{scenario.persona}

Tools the agent SHOULD reasonably have called:
{', '.join(scenario.expected_tools) or '(none specified)'}

## TRANSCRIPT

{rendered}

Now produce your JSON evaluation."""

    response = _anthropic.messages.create(
        model=EVALUATOR_MODEL,
        max_tokens=2000,
        system=EVALUATOR_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    raw_text = response.content[0].text

    result = EvaluationResult(scenario_id=scenario.id, raw_response=raw_text)

    try:
        parsed = _extract_json(raw_text)
    except (ValueError, json.JSONDecodeError) as e:
        result.summary = f"Failed to parse evaluator output: {e}"
        return result

    result.scores = parsed.get("scores", {})
    result.summary = parsed.get("summary", "")

    for f in parsed.get("failures", []):
        try:
            result.failures.append(Failure(
                criterion=f["criterion"],
                quote=f["quote"],
                issue=f["issue"],
                root_cause=f["root_cause"],
                suggested_fix=f["suggested_fix"],
            ))
        except KeyError:
            continue  # skip malformed entries silently

    result.overall_pass = (
        bool(result.scores)
        and all(score >= pass_threshold for score in result.scores.values())
    )
    return result


if __name__ == "__main__":
    # Smoke test: run the simulator on one scenario, then evaluate.
    from refinement.scenarios import get_by_id
    from refinement.simulator import run_scenario

    scenario = get_by_id("baggage_inquiry")
    print(f"Running scenario: {scenario.id}")
    sim_result = run_scenario(scenario, max_turns=8)

    print(f"\nSimulation completed. Now evaluating...\n")
    eval_result = evaluate(scenario, sim_result.transcript)

    print("=" * 60)
    print(f"SCORES")
    for k, v in eval_result.scores.items():
        print(f"  {k}: {v}/10")
    print(f"  → min: {eval_result.min_score}, pass: {eval_result.overall_pass}")

    if eval_result.failures:
        print(f"\nFAILURES ({len(eval_result.failures)}):")
        for f in eval_result.failures:
            print(f"  - [{f.root_cause}] {f.criterion}: {f.issue}")
            print(f"    Quote: {f.quote[:100]}")
            print(f"    Fix: {f.suggested_fix}")
    else:
        print("\nNo failures detected.")

    print(f"\nSummary: {eval_result.summary}")