"""
Prompt fixer: given the current system prompt and a list of failures
the evaluator classified as prompt-fixable, produce an improved prompt.

The fixer's job is to make targeted, surgical edits — not a full rewrite.
We feed it specific failure quotes from the transcript so it can address
the actual evidence rather than guessing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from anthropic import Anthropic
from dotenv import load_dotenv

from refinement.evaluator import EvaluationResult, Failure

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
PROMPT_FIXER_MODEL = os.getenv("PROMPT_FIXER_MODEL", "claude-sonnet-4-6")

if not ANTHROPIC_API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY is missing from .env")

_anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)


@dataclass
class PromptFixResult:
    """Output of one prompt-fix attempt."""
    new_prompt: str
    new_version: str
    changes_summary: str
    raw_response: str


PROMPT_FIXER_SYSTEM = """\
You are a prompt engineer specializing in airline customer service voice agents.

You will be given:
1. The CURRENT system prompt powering an Aegis Airlines voice agent
2. A list of FAILURES from a recent conversation, each with a quote, an issue description, and a suggested fix

Your job: produce an IMPROVED system prompt that targets these specific failures.

## Editing principles

- **Surgical, not catastrophic.** Keep what's working. Add, refine, or rewrite only the sections that need it. Do not rewrite the prompt from scratch.
- **Concrete instructions over vague principles.** "Always call get_policy before quoting a fee" is better than "Be accurate."
- **Address EVERY failure listed.** Don't ignore any.
- **Preserve structure.** If the prompt has section headings (## SECTION ##), preserve them and add to the relevant ones.
- **No new tools.** The agent's tool set is fixed; only adjust how/when tools are used.
- **No fabricated facts.** Don't add specific prices, dates, or policies that aren't already in the prompt.
- **Length cap.** Final prompt must be under 4000 characters. If you're approaching the limit, prefer concise rewording over deletion of working sections.

## Output format

Reply with ONLY valid JSON, no prose, no markdown fences:

{
  "new_prompt": "<the full updated system prompt as a single string>",
  "changes_summary": "<2-3 sentences describing what you changed and why, referencing the failures>"
}
"""


def _format_failures(failures: list[Failure]) -> str:
    if not failures:
        return "(no failures listed)"
    lines = []
    for i, f in enumerate(failures, 1):
        lines.append(f"### Failure {i}")
        lines.append(f"- Criterion: {f.criterion}")
        lines.append(f"- Issue: {f.issue}")
        lines.append(f"- Quote from transcript: {f.quote}")
        lines.append(f"- Suggested fix: {f.suggested_fix}")
        lines.append("")
    return "\n".join(lines)


def fix_prompt(current_prompt: str,
               current_version: str,
               evaluation: EvaluationResult) -> PromptFixResult:
    """
    Produce an improved prompt based on the evaluation's prompt-classified
    failures. Returns the new prompt, a bumped version label, and a
    summary of changes.

    Only failures with root_cause='prompt' are fed to the fixer.
    """
    prompt_failures = [f for f in evaluation.failures if f.root_cause == "prompt"]
    if not prompt_failures:
        # Nothing to fix prompt-wise; return current unchanged.
        return PromptFixResult(
            new_prompt=current_prompt,
            new_version=current_version,
            changes_summary="No prompt-classified failures; prompt unchanged.",
            raw_response="",
        )

    user_message = f"""\
## CURRENT PROMPT (version {current_version})

{current_prompt}

## FAILURES TO ADDRESS

{_format_failures(prompt_failures)}

Now produce the improved prompt as JSON."""

    response = _anthropic.messages.create(
        model=PROMPT_FIXER_MODEL,
        max_tokens=4500,
        system=PROMPT_FIXER_SYSTEM,
        messages=[{"role": "user", "content": user_message}],
    )
    raw_text = response.content[0].text

    # Reuse the evaluator's tolerant JSON extractor.
    from refinement.evaluator import _extract_json
    try:
        parsed = _extract_json(raw_text)
    except (ValueError, ValueError) as e:
        return PromptFixResult(
            new_prompt=current_prompt,
            new_version=current_version,
            changes_summary=f"Fixer output unparseable: {e}",
            raw_response=raw_text,
        )

    new_prompt = parsed.get("new_prompt", current_prompt).strip()
    summary = parsed.get("changes_summary", "(no summary)")

    # Bump the version: v1.0-baseline → v1.1, v1.1 → v1.2, etc.
    new_version = _bump_version(current_version)

    return PromptFixResult(
        new_prompt=new_prompt,
        new_version=new_version,
        changes_summary=summary,
        raw_response=raw_text,
    )


def _bump_version(version: str) -> str:
    """Bump 'v1.0-baseline' or 'v1.X' → 'v1.X+1'."""
    if version.endswith("-baseline"):
        return "v1.1"
    if version.startswith("v1.") and "." in version:
        try:
            minor = int(version.split(".")[1])
            return f"v1.{minor + 1}"
        except ValueError:
            pass
    return version + "-fixed"


if __name__ == "__main__":
    # Smoke test: run a scenario, evaluate it, and ask the fixer to address
    # any prompt failures. We pick a scenario likely to surface prompt issues.
    from refinement.scenarios import get_by_id
    from refinement.simulator import run_scenario
    from refinement.evaluator import evaluate
    from agent.system_prompt import BASELINE_SYSTEM_PROMPT, BASELINE_VERSION

    scenario = get_by_id("seat_preference")  # this one has the date bug
    print(f"Running scenario: {scenario.id}")
    sim_result = run_scenario(scenario, max_turns=10)

    print("\nEvaluating...")
    eval_result = evaluate(scenario, sim_result.transcript)

    print(f"\nScores: {eval_result.scores}")
    print(f"Pass: {eval_result.overall_pass}")
    print(f"Failures: {len(eval_result.failures)} ({sum(1 for f in eval_result.failures if f.root_cause == 'prompt')} prompt, "
          f"{sum(1 for f in eval_result.failures if f.root_cause == 'code')} code)")

    if not eval_result.has_prompt_failures:
        print("\nNo prompt failures detected — nothing to fix.")
    else:
        print("\nRunning prompt fixer...")
        fix = fix_prompt(BASELINE_SYSTEM_PROMPT, BASELINE_VERSION, eval_result)
        print(f"\nNew version: {fix.new_version}")
        print(f"Length: {len(fix.new_prompt)} chars (was {len(BASELINE_SYSTEM_PROMPT)})")
        print(f"\nChanges summary:\n{fix.changes_summary}")
        print(f"\n--- New prompt (first 1500 chars) ---\n{fix.new_prompt[:1500]}")