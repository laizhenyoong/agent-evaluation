"""Run the golden evaluation set against the live support agent.

Usage:
    .venv/bin/python -m evaluation.runner
    .venv/bin/python -m evaluation.runner --judge ollama --case kb_refund_policy
"""

import argparse
import json
from collections.abc import Iterable

from evaluation.contracts import EvaluationCase
from evaluation.golden_dataset import GOLDEN_CASES
from evaluation.quality_judge import OllamaResponseJudge
from evaluation.scoring import ScoreResult, deterministic_score, trajectory_score
from evaluation.traces import AgentTrace, ObservedToolCall
from support_agent import build_agent, reset_demo_state


def capture_run(case: EvaluationCase) -> AgentTrace:
    """Run a fresh agent and extract its tool-use blocks as the trajectory."""
    reset_demo_state()
    agent = build_agent()
    answer = str(agent(case.prompt))
    calls: list[ObservedToolCall] = []
    for message in agent.messages:
        for block in message.get("content", []):
            if "toolUse" in block:
                tool_use = block["toolUse"]
                calls.append(
                    ObservedToolCall(
                        tool_use["name"],
                        dict(tool_use.get("input", {})),
                    )
                )
    return AgentTrace(case.case_id, case.prompt, {}, tuple(calls), answer)


def _summary(scores: Iterable[ScoreResult]) -> dict[str, object]:
    score_list = list(scores)
    passed = sum(score.passed for score in score_list)
    return {
        "passed": passed,
        "total": len(score_list),
        "pass_rate": round(passed / len(score_list), 3) if score_list else 0.0,
    }


def _score_data(result: ScoreResult) -> dict[str, object]:
    return {
        "passed": result.passed,
        "score": result.score,
        "reasons": result.reasons,
    }


def evaluate(
    cases: Iterable[EvaluationCase],
    judge: OllamaResponseJudge | None = None,
) -> dict[str, object]:
    """Return a JSON-serialisable report; the caller chooses where to persist it."""
    rows: list[dict[str, object]] = []
    deterministic_results: list[ScoreResult] = []
    trajectory_results: list[ScoreResult] = []
    judge_results: list[ScoreResult] = []
    for case in cases:
        run = capture_run(case)
        deterministic = deterministic_score(case, run)
        trajectory = trajectory_score(case, run)
        judge_score = judge.score(case, run) if judge else None
        deterministic_results.append(deterministic)
        trajectory_results.append(trajectory)
        if judge_score:
            judge_results.append(judge_score)
        rows.append(
            {
                "case_id": case.case_id,
                "answer": run.answer,
                "tool_calls": [
                    {"name": call.name, "arguments": call.arguments}
                    for call in run.tool_calls
                ],
                "deterministic": _score_data(deterministic),
                "trajectory": _score_data(trajectory),
                "judge": None if judge_score is None else _score_data(judge_score),
            }
        )
    report: dict[str, object] = {
        "cases": rows,
        "deterministic": _summary(deterministic_results),
        "trajectory": _summary(trajectory_results),
    }
    if judge:
        report["judge"] = _summary(judge_results)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate support-agent trajectories against the golden set.")
    parser.add_argument("--case", action="append", help="Golden case ID to run; may be repeated.")
    parser.add_argument("--judge", choices=("off", "ollama"), default="off", help="Run the qualitative Ollama judge.")
    parser.add_argument("--output", help="Write the full JSON report to this file.")
    args = parser.parse_args()
    selected = tuple(case for case in GOLDEN_CASES if not args.case or case.case_id in args.case)
    unknown = set(args.case or ()) - {case.case_id for case in selected}
    if unknown:
        parser.error(f"unknown case ID(s): {', '.join(sorted(unknown))}")
    report = evaluate(selected, OllamaResponseJudge() if args.judge == "ollama" else None)
    rendered = json.dumps(report, indent=2)
    if args.output:
        with open(args.output, "w") as file:
            file.write(rendered + "\n")
    print(rendered)
    # A non-zero exit makes this directly usable by a CI quality gate.
    summaries = [report["deterministic"], report["trajectory"]]
    if "judge" in report:
        summaries.append(report["judge"])
    return 0 if all(summary["passed"] == summary["total"] for summary in summaries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
