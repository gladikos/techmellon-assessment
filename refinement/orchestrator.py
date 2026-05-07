"""
The autonomous refinement loop.

For ONE scenario, repeat up to MAX_ITERATIONS times:
  1. Push the current prompt to ElevenLabs
  2. Run the scenario simulation
  3. Evaluate the transcript
  4. If all scores >= PASS_THRESHOLD → stop, success
  5. Otherwise:
     - For prompt-classified failures → call prompt_fixer → bump version
     - For code-classified failures → call code_fixer → apply patches
  6. Repeat from step 1

Logging: every iteration writes a JSON file to logs/run_<timestamp>/.
A run_summary.json is written at termination.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from agent.elevenlabs_client import update_agent_config
from agent.system_prompt import BASELINE_VERSION, get_baseline_prompt
from refinement.code_fixer import CodeFixResult, fix_code
from refinement.evaluator import EvaluationResult, evaluate
from refinement.prompt_fixer import PromptFixResult, fix_prompt
from refinement.scenarios import Scenario, get_by_id
from refinement.simulator import SimulationResult, run_scenario

load_dotenv()

PASS_THRESHOLD = int(os.getenv("PASS_THRESHOLD", "8"))
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "5"))

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_ROOT = PROJECT_ROOT / "logs"


# --- Result types ------------------------------------------------------- #

@dataclass
class IterationRecord:
    """Everything that happened in one iteration."""
    iteration: int
    prompt_version: str
    prompt_text: str
    scenario_id: str
    transcript: list[dict]
    scores: dict[str, int]
    overall_pass: bool
    failures: list[dict]
    summary: str
    prompt_fix_summary: Optional[str] = None
    code_fix_summary: Optional[str] = None
    code_patches_applied: int = 0
    code_patches_rejected: int = 0
    error: Optional[str] = None


@dataclass
class RunResult:
    """The full run: every iteration plus terminal state."""
    scenario_id: str
    started_at: str
    finished_at: str
    iterations: list[IterationRecord] = field(default_factory=list)
    converged: bool = False
    final_version: str = ""
    log_dir: str = ""
    pass_threshold: int = PASS_THRESHOLD
    max_iterations: int = MAX_ITERATIONS


# --- Logging helpers ---------------------------------------------------- #

def _make_log_dir(scenario_id: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = LOGS_ROOT / f"run_{timestamp}_{scenario_id}"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _write_iteration_log(log_dir: Path, record: IterationRecord) -> None:
    path = log_dir / f"iteration_{record.iteration}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(record), f, indent=2, ensure_ascii=False)


def _write_run_summary(log_dir: Path, run: RunResult) -> None:
    path = log_dir / "run_summary.json"
    summary = {
        "scenario_id": run.scenario_id,
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "converged": run.converged,
        "iterations_run": len(run.iterations),
        "max_iterations": run.max_iterations,
        "pass_threshold": run.pass_threshold,
        "final_version": run.final_version,
        "scores_by_iteration": [
            {
                "iteration": it.iteration,
                "version": it.prompt_version,
                "scores": it.scores,
                "overall_pass": it.overall_pass,
            }
            for it in run.iterations
        ],
        "first_scores": run.iterations[0].scores if run.iterations else {},
        "final_scores": run.iterations[-1].scores if run.iterations else {},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


# --- Pretty printer for the console -------------------------------------- #

def _print_iteration_header(iteration: int, version: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  ITERATION {iteration}  (prompt version: {version})")
    print(f"{'=' * 60}")


def _print_scores(scores: dict[str, int], pass_: bool) -> None:
    print("\n  Scores:")
    for k, v in scores.items():
        marker = "✅" if v >= PASS_THRESHOLD else "⚠️"
        print(f"    {marker} {k}: {v}/10")
    print(f"  → Overall pass: {pass_}")


# --- The main loop ------------------------------------------------------ #

def run_pipeline(scenario_id: str,
                 pass_threshold: int = PASS_THRESHOLD,
                 max_iterations: int = MAX_ITERATIONS,
                 max_turns_per_scenario: int = 12) -> RunResult:
    """
    Run the full refinement pipeline for one scenario. Returns a
    RunResult with every iteration's records.
    """
    scenario = get_by_id(scenario_id)
    log_dir = _make_log_dir(scenario_id)
    print(f"Logging to: {log_dir}")
    print(f"Pass threshold: {pass_threshold}/10  |  Max iterations: {max_iterations}")
    print(f"Scenario: {scenario.id} — {scenario.title}")

    run = RunResult(
        scenario_id=scenario_id,
        started_at=datetime.now().isoformat(),
        finished_at="",
        log_dir=str(log_dir),
        pass_threshold=pass_threshold,
        max_iterations=max_iterations,
    )

    current_prompt = get_baseline_prompt()
    current_version = BASELINE_VERSION

    for iteration_num in range(1, max_iterations + 1):
        _print_iteration_header(iteration_num, current_version)

        record = IterationRecord(
            iteration=iteration_num,
            prompt_version=current_version,
            prompt_text=current_prompt,
            scenario_id=scenario_id,
            transcript=[],
            scores={},
            overall_pass=False,
            failures=[],
            summary="",
        )

        # 1) Push current prompt
        try:
            print("\n  → Pushing prompt to ElevenLabs...")
            update_agent_config(prompt=current_prompt)
            print("    ✓ Prompt pushed.")
        except Exception as e:
            record.error = f"push_failed: {type(e).__name__}: {e}"
            run.iterations.append(record)
            _write_iteration_log(log_dir, record)
            print(f"    ✗ Push failed: {e}")
            break  # cannot continue without pushed prompt

        # 2) Run simulation
        print("\n  → Running simulation...")
        sim: SimulationResult = run_scenario(scenario, max_turns=max_turns_per_scenario)
        record.transcript = sim.transcript
        if sim.error:
            record.error = sim.error
            print(f"    ⚠️  Simulator error: {sim.error}")
        else:
            print(f"    ✓ Simulation complete ({len(sim.transcript)} turns).")

        # 3) Evaluate
        print("\n  → Evaluating transcript...")
        try:
            eval_result: EvaluationResult = evaluate(scenario, sim.transcript,
                                                     pass_threshold=pass_threshold)
            record.scores = eval_result.scores
            record.overall_pass = eval_result.overall_pass
            record.failures = [asdict(f) for f in eval_result.failures]
            record.summary = eval_result.summary
        except Exception as e:
            record.error = f"eval_failed: {type(e).__name__}: {e}"
            run.iterations.append(record)
            _write_iteration_log(log_dir, record)
            print(f"    ✗ Eval failed: {e}")
            break

        _print_scores(record.scores, record.overall_pass)
        if record.summary:
            print(f"\n  Summary: {record.summary}")

        run.iterations.append(record)
        _write_iteration_log(log_dir, record)

        # 4) Pass? Stop here.
        if record.overall_pass:
            print(f"\n🎉 All scores ≥ {pass_threshold}. Pipeline converged on iteration {iteration_num}.")
            run.converged = True
            run.final_version = current_version
            break

        # 5) Apply fixes — either or both fixers may fire.
        if iteration_num == max_iterations:
            print(f"\n  → Last iteration; not running fixers.")
            run.final_version = current_version
            break

        if eval_result.has_prompt_failures:
            print(f"\n  → {sum(1 for f in eval_result.failures if f.root_cause == 'prompt')} prompt failures — running prompt_fixer...")
            try:
                fix: PromptFixResult = fix_prompt(current_prompt, current_version, eval_result)
                if fix.new_prompt != current_prompt:
                    current_prompt = fix.new_prompt
                    current_version = fix.new_version
                    record.prompt_fix_summary = fix.changes_summary
                    print(f"    ✓ Prompt updated to {current_version} ({len(current_prompt)} chars).")
                    print(f"    Summary: {fix.changes_summary[:200]}")
                else:
                    print(f"    (Fixer chose not to change the prompt.)")
                    record.prompt_fix_summary = fix.changes_summary
            except Exception as e:
                print(f"    ✗ Prompt fixer error: {e}")

        if eval_result.has_code_failures:
            print(f"\n  → {sum(1 for f in eval_result.failures if f.root_cause == 'code')} code failures — running code_fixer...")
            try:
                cfix: CodeFixResult = fix_code(eval_result, dry_run=False)
                record.code_fix_summary = cfix.summary
                record.code_patches_applied = len(cfix.patches_applied)
                record.code_patches_rejected = len(cfix.patches_rejected)
                if cfix.patches_applied:
                    print(f"    ✓ {len(cfix.patches_applied)} patch(es) applied.")
                    print(f"    Summary: {cfix.summary[:200]}")
                    print("    (FastAPI --reload will pick up changes automatically.)")
                    time.sleep(2)  # let uvicorn finish reloading
                else:
                    print(f"    (No patches applied; {len(cfix.patches_rejected)} rejected.)")
            except Exception as e:
                print(f"    ✗ Code fixer error: {e}")

        # Re-write the iteration record now that fix info is filled in.
        _write_iteration_log(log_dir, record)

    run.finished_at = datetime.now().isoformat()
    if not run.final_version:
        run.final_version = current_version
    _write_run_summary(log_dir, run)

    print(f"\n{'=' * 60}")
    print(f"  RUN COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Converged: {run.converged}")
    print(f"  Iterations: {len(run.iterations)}/{max_iterations}")
    print(f"  Final version: {run.final_version}")
    if run.iterations:
        first = run.iterations[0].scores
        last = run.iterations[-1].scores
        print(f"  First scores: {first}")
        print(f"  Final scores: {last}")
    print(f"  Logs: {log_dir}")

    return run

def run_all_scenarios(scenario_ids: Optional[list[str]] = None,
                      pass_threshold: int = PASS_THRESHOLD,
                      max_iterations: int = MAX_ITERATIONS,
                      max_turns_per_scenario: int = 12) -> list[RunResult]:
    """
    Run the refinement pipeline once for each scenario in the list, or
    for ALL scenarios in scenarios.SCENARIOS if no list is given.

    Each scenario gets its own independent pipeline run with its own
    log directory. Returns a list of RunResult objects, one per scenario.

    Used by:
      - The Streamlit UI's 'Run full pipeline' button
      - CLI: python -m refinement.orchestrator --all
    """
    from refinement.scenarios import SCENARIOS

    if scenario_ids is None:
        scenario_ids = [s.id for s in SCENARIOS]

    print(f"\n{'#' * 70}")
    print(f"#  RUNNING PIPELINE FOR {len(scenario_ids)} SCENARIO(S)")
    print(f"#  {', '.join(scenario_ids)}")
    print(f"{'#' * 70}\n")

    results: list[RunResult] = []
    overall_start = time.time()

    for i, sid in enumerate(scenario_ids, 1):
        print(f"\n{'#' * 70}")
        print(f"#  RUN {i}/{len(scenario_ids)}: {sid}")
        print(f"{'#' * 70}")
        scenario_start = time.time()
        try:
            result = run_pipeline(
                sid,
                pass_threshold=pass_threshold,
                max_iterations=max_iterations,
                max_turns_per_scenario=max_turns_per_scenario,
            )
            results.append(result)
        except Exception as e:
            # One scenario failing should not abort the whole run.
            print(f"\n✗ Scenario {sid} crashed: {type(e).__name__}: {e}")
            # Build a minimal RunResult to keep the summary table coherent.
            results.append(RunResult(
                scenario_id=sid,
                started_at=datetime.now().isoformat(),
                finished_at=datetime.now().isoformat(),
                converged=False,
                final_version="(crashed)",
                log_dir="",
                pass_threshold=pass_threshold,
                max_iterations=max_iterations,
            ))
        elapsed = time.time() - scenario_start
        print(f"\n  ⏱ Scenario elapsed: {elapsed:.1f}s")

    # Summary table.
    total = time.time() - overall_start
    print(f"\n\n{'=' * 70}")
    print("FULL PIPELINE SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Scenario':<28} {'Iters':>6} {'Pass':>6} {'Final scores':>30}")
    print("-" * 70)
    converged_count = 0
    for r in results:
        iters = len(r.iterations)
        marker = "✅" if r.converged else "❌"
        if r.converged:
            converged_count += 1
        final = r.iterations[-1].scores if r.iterations else {}
        score_str = " ".join(f"{k[0]}:{v}" for k, v in final.items())
        print(f"{r.scenario_id:<28} {iters:>6} {marker:>6} {score_str:>30}")
    print("-" * 70)
    print(f"Converged: {converged_count}/{len(results)}  |  Total time: {total:.1f}s")

    return results

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m refinement.orchestrator <scenario_id>     # run one scenario")
        print("  python -m refinement.orchestrator --all              # run all 10 scenarios")
        print("  python -m refinement.orchestrator --list <id1> <id2> # run a subset")
        print("\nAvailable scenario IDs:")
        from refinement.scenarios import SCENARIOS
        for s in SCENARIOS:
            print(f"  - {s.id}")
        sys.exit(1)

    arg = sys.argv[1]
    if arg == "--all":
        run_all_scenarios()
    elif arg == "--list":
        if len(sys.argv) < 3:
            print("--list requires at least one scenario id")
            sys.exit(1)
        run_all_scenarios(scenario_ids=sys.argv[2:])
    else:
        run_pipeline(arg)