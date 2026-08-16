"""Run the golden cases against the live agent and persist the trajectories.

This step does no scoring.  It is the slow, expensive, non-deterministic half
of the loop -- a local 30B model over 50 cases -- so its only job is to record
what happened well enough that every later step can read it from disk.

Usage:
    .venv/bin/python -m evaluation.capture --output results/baseline.jsonl
"""

import argparse
import time
from collections.abc import Iterable, Sequence
from pathlib import Path

from evaluation.contracts import EvaluationCase
from evaluation.golden_dataset import GOLDEN_CASES
from evaluation.traces import AgentTrace, ObservedToolCall, write_traces
from support_agent import DEFAULT_MODEL_ID, SYSTEM_PROMPT, build_agent, reset_demo_state


def capture_case(
    case: EvaluationCase,
    *,
    model_id: str,
    system_prompt: str,
) -> AgentTrace:
    """Run one case on a fresh agent, recording a failure rather than raising.

    A crash on case 7 must not destroy the six trajectories already paid for,
    so the exception becomes a field on the trace and the sweep continues.
    """
    reset_demo_state()
    agent_config = {"model_id": model_id, "system_prompt": system_prompt}
    started_at = time.perf_counter()
    answer: str | None = None
    error: str | None = None
    calls: list[ObservedToolCall] = []

    try:
        agent = build_agent(model_id=model_id, system_prompt=system_prompt)
        answer = str(agent(case.prompt))
        for message in agent.messages:
            for block in message.get("content", []):
                if "toolUse" in block:
                    tool_use = block["toolUse"]
                    calls.append(
                        ObservedToolCall(tool_use["name"], dict(tool_use.get("input", {})))
                    )
    except Exception as exception:
        error = f"{type(exception).__name__}: {exception}"

    return AgentTrace(
        case_id=case.case_id,
        prompt=case.prompt,
        agent_config=agent_config,
        tool_calls=tuple(calls),
        answer=answer,
        timing_ms=round((time.perf_counter() - started_at) * 1000),
        error=error,
    )


def capture(
    cases: Iterable[EvaluationCase],
    *,
    model_id: str = DEFAULT_MODEL_ID,
    system_prompt: str = SYSTEM_PROMPT,
) -> list[AgentTrace]:
    """Capture every case in order, one fresh agent each."""
    return [
        capture_case(case, model_id=model_id, system_prompt=system_prompt) for case in cases
    ]


def select_cases(case_ids: Sequence[str] | None) -> tuple[EvaluationCase, ...]:
    """Resolve the requested case IDs, refusing silently-empty selections."""
    if not case_ids:
        return GOLDEN_CASES
    selected = tuple(case for case in GOLDEN_CASES if case.case_id in set(case_ids))
    unknown = set(case_ids) - {case.case_id for case in selected}
    if unknown:
        raise ValueError(f"unknown case ID(s): {', '.join(sorted(unknown))}")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture agent trajectories for the golden set.")
    parser.add_argument("--case", action="append", help="Golden case ID to run; may be repeated.")
    parser.add_argument("--output", type=Path, default=Path("results/latest.jsonl"))
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--system-prompt",
        help="Override the agent's system prompt, to show the eval catching a regression.",
    )
    parser.add_argument(
        "--system-prompt-file",
        type=Path,
        help="Read the system prompt override from a file.",
    )
    arguments = parser.parse_args()

    if arguments.system_prompt and arguments.system_prompt_file:
        parser.error("use --system-prompt or --system-prompt-file, not both")
    system_prompt = SYSTEM_PROMPT
    if arguments.system_prompt:
        system_prompt = arguments.system_prompt
    elif arguments.system_prompt_file:
        system_prompt = arguments.system_prompt_file.read_text(encoding="utf-8").strip()

    try:
        cases = select_cases(arguments.case)
    except ValueError as error:
        parser.error(str(error))

    traces = capture(cases, model_id=arguments.model, system_prompt=system_prompt)
    write_traces(arguments.output, traces)
    failed = sum(trace.error is not None for trace in traces)
    print(
        f"Captured {len(traces)} trajectories to {arguments.output} "
        f"(model={arguments.model}, errors={failed})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
