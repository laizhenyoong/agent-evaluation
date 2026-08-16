"""Score persisted traces and gate a run against a committed baseline.

The gate compares a run to a stored baseline rather than demanding perfection.
The agent is deliberately imperfect, so an absolute "100% or fail" gate can
never go green and therefore gates nothing; what CI can meaningfully catch is a
*drop* against the number this repo last agreed to.

Usage:
    .venv/bin/python -m evaluation.report --traces results/latest.jsonl
    .venv/bin/python -m evaluation.report --traces results/latest.jsonl \\
        --baseline results/baseline-summary.json
"""

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from evaluation.contracts import EvaluationCase
from evaluation.golden_dataset import GOLDEN_CASES
from evaluation.judge import LABELS, Judgement, load_judgements
from evaluation.scoring import ScoreResult, deterministic_score, summarize, trajectory_score
from evaluation.traces import AgentTrace, load_traces

# Rates the gate watches. A drop in any of them fails the run.
GATED_METRICS = (
    "deterministic.pass_rate",
    "deterministic.mean_score",
    "trajectory.pass_rate",
    "trajectory.mean_score",
)


def build_report(
    traces: Sequence[AgentTrace],
    judgements: Sequence[Judgement] = (),
    cases: Sequence[EvaluationCase] = GOLDEN_CASES,
) -> dict:
    """Score every trace and return a JSON-serialisable report."""
    cases_by_id = {case.case_id: case for case in cases}
    missing = [trace.case_id for trace in traces if trace.case_id not in cases_by_id]
    if missing:
        raise ValueError(f"No golden case for traces: {', '.join(sorted(set(missing)))}")

    judgements_by_id = {judgement.case_id: judgement for judgement in judgements}
    deterministic: list[ScoreResult] = []
    trajectory: list[ScoreResult] = []
    rows: list[dict] = []

    for trace in traces:
        case = cases_by_id[trace.case_id]
        deterministic_result = deterministic_score(case, trace)
        trajectory_result = trajectory_score(case, trace)
        deterministic.append(deterministic_result)
        trajectory.append(trajectory_result)
        judgement = judgements_by_id.get(trace.case_id)
        rows.append(
            {
                "case_id": trace.case_id,
                "answer": trace.answer,
                "timing_ms": trace.timing_ms,
                "error": trace.error,
                "tool_calls": [
                    {"name": call.name, "arguments": call.arguments} for call in trace.tool_calls
                ],
                "deterministic": asdict(deterministic_result),
                "trajectory": asdict(trajectory_result),
                "judge": None if judgement is None else asdict(judgement),
            }
        )

    report = {
        "agent_config": traces[0].agent_config if traces else {},
        "cases": rows,
        "deterministic": asdict(summarize(deterministic)),
        "trajectory": asdict(summarize(trajectory)),
    }
    if judgements:
        report["judge"] = _judge_summary(judgements)
    return report


def _judge_summary(judgements: Sequence[Judgement]) -> dict:
    """Per-label pass rates over the judgements that carry a real opinion."""
    usable = [judgement for judgement in judgements if judgement.error is None]
    summary: dict[str, object] = {
        "judged_count": len(usable),
        "unusable_count": len(judgements) - len(usable),
    }
    for label in LABELS:
        summary[f"{label}_rate"] = (
            round(sum(getattr(judgement, label) for judgement in usable) / len(usable), 3)
            if usable
            else 0.0
        )
    summary["pass_rate"] = (
        round(sum(judgement.passed for judgement in usable) / len(usable), 3) if usable else 0.0
    )
    return summary


def summary_only(report: dict) -> dict:
    """The part of a report worth committing as a baseline."""
    keys = ("agent_config", "deterministic", "trajectory", "judge")
    return {key: report[key] for key in keys if key in report}


def _metric(report: dict, path: str) -> float | None:
    section, _, name = path.partition(".")
    value = report.get(section, {}).get(name) if isinstance(report.get(section), dict) else None
    return float(value) if isinstance(value, (int, float)) else None


def check_regressions(
    report: dict, baseline: dict, *, tolerance: float = 0.0
) -> list[str]:
    """Return one message per gated metric that fell below the baseline."""
    if tolerance < 0:
        raise ValueError("tolerance must not be negative")

    failures: list[str] = []
    for path in GATED_METRICS:
        current = _metric(report, path)
        previous = _metric(baseline, path)
        if current is None or previous is None:
            continue
        if current < previous - tolerance:
            failures.append(
                f"{path} regressed: {current:.3f} < {previous:.3f} (tolerance {tolerance:.3f})"
            )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Score captured traces and gate on regressions.")
    parser.add_argument("--traces", type=Path, default=Path("results/latest.jsonl"))
    parser.add_argument("--judgements", type=Path, help="Optional judgement file to fold in.")
    parser.add_argument("--baseline", type=Path, help="Baseline summary to gate against.")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="How far a rate may fall before it counts as a regression.",
    )
    parser.add_argument("--output", type=Path, help="Write the full JSON report here.")
    parser.add_argument(
        "--write-baseline", type=Path, help="Write this run's summary as a new baseline."
    )
    arguments = parser.parse_args()

    judgements = load_judgements(arguments.judgements) if arguments.judgements else []
    report = build_report(load_traces(arguments.traces), judgements)

    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if arguments.write_baseline:
        arguments.write_baseline.parent.mkdir(parents=True, exist_ok=True)
        arguments.write_baseline.write_text(
            json.dumps(summary_only(report), indent=2) + "\n", encoding="utf-8"
        )

    print(json.dumps(summary_only(report), indent=2))

    if not arguments.baseline:
        return 0

    baseline = json.loads(arguments.baseline.read_text(encoding="utf-8"))
    failures = check_regressions(report, baseline, tolerance=arguments.tolerance)
    if failures:
        print("\nREGRESSION:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print(f"\nNo regression against {arguments.baseline}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
