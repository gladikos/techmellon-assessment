"""
Code fixer: given code-classified failures from the evaluator, produce
targeted find/replace patches against backend route files.

Design principles:
- Whitelist of editable files. Anything outside is rejected.
- LLM produces {file, find, replace} triples; we apply literal
  string replacement only — no diff parsing, no AST manipulation.
- Each `find` must match exactly once. Zero or multiple matches → abort.
- A backup of every modified file is written before changes, so a bad
  fix is recoverable.
- After applying patches, FastAPI's --reload picks up the change
  automatically; we don't restart anything ourselves.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

from refinement.evaluator import EvaluationResult, Failure, _extract_json

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CODE_FIXER_MODEL = os.getenv("CODE_FIXER_MODEL", "claude-sonnet-4-6")

if not ANTHROPIC_API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY is missing from .env")

_anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)


# --- Whitelist & paths -------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EDITABLE_FILES = {
    "backend/routes/flights.py",
    "backend/routes/bookings.py",
    "backend/routes/knowledge.py",
}
BACKUP_DIR = PROJECT_ROOT / "logs" / "code_fix_backups"


# --- Result types ------------------------------------------------------- #

@dataclass
class CodePatch:
    file: str
    find: str
    replace: str
    rationale: str


@dataclass
class CodeFixResult:
    patches_proposed: list[CodePatch]
    patches_applied: list[CodePatch]
    patches_rejected: list[tuple[CodePatch, str]]  # (patch, reason)
    summary: str
    raw_response: str

    @property
    def success(self) -> bool:
        return bool(self.patches_applied) and not self.patches_rejected


# --- LLM prompt --------------------------------------------------------- #

CODE_FIXER_SYSTEM = """\
You are a backend engineer fixing a FastAPI application.

You will be given:
1. A list of FAILURES from a recent voice-agent conversation, each
   classified by an evaluator as having a 'code' root cause
2. The current source of one or more eligible backend files

Your job: produce a list of MINIMAL, surgical patches that fix the
failures.

## Rules

- You may only edit these files: backend/routes/flights.py,
  backend/routes/bookings.py, backend/routes/knowledge.py.
- Each patch is a (file, find, replace) triple. The `find` string
  must appear EXACTLY ONCE in the file — if it's ambiguous, the patch
  will be rejected. Use enough context (a few lines) to make it unique.
- Preserve indentation exactly. Python is whitespace-sensitive.
- Do not introduce new dependencies. The existing imports are what you have.
- Do not change function signatures or route paths — those are
  contract with the LLM agent and breaking them will cascade failures.
- Keep changes minimal. A two-line fix is better than a thirty-line
  refactor, even if the refactor is "cleaner".
- If you cannot produce a confident fix, return an empty patches list
  with an explanation in the rationale field.

## Output format

Reply with ONLY valid JSON, no prose, no markdown fences:

{
  "patches": [
    {
      "file": "backend/routes/flights.py",
      "find": "<exact text in the file, including indentation>",
      "replace": "<replacement text>",
      "rationale": "<one sentence: what this fixes>"
    }
  ],
  "summary": "<2-3 sentences summarizing what you changed and why>"
}
"""


# --- Helpers ------------------------------------------------------------ #

def _read_eligible_files() -> dict[str, str]:
    """Load the current contents of every whitelisted file."""
    contents = {}
    for rel_path in EDITABLE_FILES:
        full = PROJECT_ROOT / rel_path
        if full.exists():
            contents[rel_path] = full.read_text(encoding="utf-8")
    return contents


def _format_files_for_prompt(files: dict[str, str]) -> str:
    blocks = []
    for path, content in files.items():
        blocks.append(f"### {path}\n```python\n{content}\n```")
    return "\n\n".join(blocks)


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


def _backup(rel_path: str) -> Path:
    """Copy a file to logs/code_fix_backups/<timestamp>_<basename>."""
    import time
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    src = PROJECT_ROOT / rel_path
    name = rel_path.replace("/", "_").replace("\\", "_")
    dest = BACKUP_DIR / f"{int(time.time())}_{name}"
    shutil.copy2(src, dest)
    return dest


def _apply_patch(patch: CodePatch) -> tuple[bool, str]:
    """
    Apply a single patch by literal string replacement.

    Returns (applied, reason). If applied is False, reason explains why.
    """
    if patch.file not in EDITABLE_FILES:
        return False, f"file {patch.file!r} not in whitelist"

    full = PROJECT_ROOT / patch.file
    if not full.exists():
        return False, f"file {patch.file!r} does not exist"

    content = full.read_text(encoding="utf-8")
    occurrences = content.count(patch.find)
    if occurrences == 0:
        return False, "find string not found in file"
    if occurrences > 1:
        return False, f"find string is ambiguous (matches {occurrences} times)"

    _backup(patch.file)
    new_content = content.replace(patch.find, patch.replace, 1)
    full.write_text(new_content, encoding="utf-8")
    return True, "applied"


# --- Main entry point --------------------------------------------------- #

def fix_code(evaluation: EvaluationResult, dry_run: bool = False) -> CodeFixResult:
    """
    Produce and apply patches addressing the evaluation's
    code-classified failures. Returns a CodeFixResult describing
    everything attempted.
    """
    code_failures = [f for f in evaluation.failures if f.root_cause == "code"]
    if not code_failures:
        return CodeFixResult(
            patches_proposed=[],
            patches_applied=[],
            patches_rejected=[],
            summary="No code-classified failures; no changes attempted.",
            raw_response="",
        )

    files = _read_eligible_files()
    user_message = f"""\
## CURRENT BACKEND FILES (eligible for editing)

{_format_files_for_prompt(files)}

## CODE FAILURES TO ADDRESS

{_format_failures(code_failures)}

Now produce your JSON patches."""

    response = _anthropic.messages.create(
        model=CODE_FIXER_MODEL,
        max_tokens=4000,
        system=CODE_FIXER_SYSTEM,
        messages=[{"role": "user", "content": user_message}],
    )
    raw_text = response.content[0].text

    try:
        parsed = _extract_json(raw_text)
    except (ValueError, Exception) as e:
        return CodeFixResult(
            patches_proposed=[],
            patches_applied=[],
            patches_rejected=[],
            summary=f"Fixer output unparseable: {e}",
            raw_response=raw_text,
        )

    proposed: list[CodePatch] = []
    for p in parsed.get("patches", []):
        try:
            proposed.append(CodePatch(
                file=p["file"],
                find=p["find"],
                replace=p["replace"],
                rationale=p.get("rationale", ""),
            ))
        except KeyError:
            continue

    applied: list[CodePatch] = []
    rejected: list[tuple[CodePatch, str]] = []
    for patch in proposed:
        if dry_run:
            applied.append(patch)  # pretend success
            continue
        ok, reason = _apply_patch(patch)
        if ok:
            applied.append(patch)
        else:
            rejected.append((patch, reason))

    return CodeFixResult(
        patches_proposed=proposed,
        patches_applied=applied,
        patches_rejected=rejected,
        summary=parsed.get("summary", ""),
        raw_response=raw_text,
    )


if __name__ == "__main__":
    # Smoke test: this is harder to test without a real code failure.
    # We construct a fake EvaluationResult with a synthetic code failure
    # and confirm the fixer returns SOMETHING parseable. (We don't apply
    # the patches in the smoke test — the file should be left unchanged.)
    from refinement.evaluator import EvaluationResult, Failure

    fake_eval = EvaluationResult(
        scenario_id="synthetic_test",
        scores={"understanding": 9, "tool_use": 9, "outcome": 4, "conversation_quality": 9},
        failures=[
            Failure(
                criterion="outcome",
                quote="[TOOL_RESULT] search_flights → []",
                issue="search_flights returned an empty list when filtering by date even though flights exist on that date",
                root_cause="code",
                suggested_fix="Check the date filtering in flights.py — the comparison may be using the wrong column or wrong format",
            )
        ],
        overall_pass=False,
        summary="Synthetic test failure for code fixer smoke testing.",
    )

    print("Calling code fixer with synthetic failure...")
    result = fix_code(fake_eval, dry_run=True)
    print(f"\nProposed: {len(result.patches_proposed)}")
    print(f"Applied: {len(result.patches_applied)}")
    print(f"Rejected: {len(result.patches_rejected)}")
    print(f"\nSummary: {result.summary}")

    if result.patches_proposed:
        for i, p in enumerate(result.patches_proposed, 1):
            print(f"\n--- Patch {i} ---")
            print(f"File: {p.file}")
            print(f"Rationale: {p.rationale}")
            print(f"Find ({len(p.find)} chars):\n{p.find[:200]}")
            print(f"Replace ({len(p.replace)} chars):\n{p.replace[:200]}")

    if result.patches_applied:
        print("\n(dry_run=True: patches were proposed but NOT written to disk)")


    if result.patches_rejected:
        print("\nRejected:")
        for patch, reason in result.patches_rejected:
            print(f"  - {patch.file}: {reason}")