"""Deterministic and trajectory-level scorers.

These scorers intentionally inspect observable behaviour, rather than infer it
from the final answer.  A model can sound helpful while taking a needless or
unsafe path; that should still be a failed agent run.
"""

from collections import Counter

from dataclasses import dataclass

from evaluation.contracts import EvaluationCase, ToolCallExpectation
from evaluation.traces import AgentTrace, ObservedToolCall


@dataclass(frozen=True)
class ScoreResult:
    passed: bool
    score: float
    reasons: tuple[str, ...]


def _matches(call: ObservedToolCall, expected: ToolCallExpectation) -> bool:
    if call.name != expected.name:
        return False
    if any(call.arguments.get(key) != value for key, value in expected.exact_args.items()):
        return False
    return all(
        expected_value.lower() in str(call.arguments.get(key, "")).lower()
        for key, expected_value in expected.contains_args.items()
    )


def deterministic_score(case: EvaluationCase, run: AgentTrace) -> ScoreResult:
    """Check required calls and arguments, forbidden calls, bounds, and loops."""
    reasons: list[str] = []
    calls = run.tool_calls
    for expected in case.required_tools:
        if not any(_matches(call, expected) for call in calls):
            reasons.append(f"missing required call: {expected.name}({expected.exact_args or expected.contains_args})")
    forbidden = [call.name for call in calls if call.name in case.forbidden_tools]
    if forbidden:
        reasons.append(f"forbidden tool calls: {', '.join(forbidden)}")
    if len(calls) > case.max_tool_calls:
        reasons.append(f"too many tool calls: {len(calls)} > {case.max_tool_calls}")
    duplicate_names = [name for name, count in Counter(call.name for call in calls).items() if count > 1]
    if duplicate_names:
        reasons.append(f"tool loop/retry detected: {', '.join(duplicate_names)}")
    return ScoreResult(not reasons, 1.0 if not reasons else 0.0, tuple(reasons))


def trajectory_score(case: EvaluationCase, run: AgentTrace) -> ScoreResult:
    """Score whether the path was economical and followed the intended order."""
    reasons: list[str] = []
    actual = tuple(call.name for call in run.tool_calls)
    allowed = {tool.name for tool in case.required_tools}
    extras = [name for name in actual if name not in allowed]
    if extras:
        reasons.append(f"unnecessary tools: {', '.join(extras)}")
    position = 0
    for expected_name in case.preferred_sequence:
        try:
            position = actual.index(expected_name, position) + 1
        except ValueError:
            reasons.append(f"preferred path missing or out of order: {' -> '.join(case.preferred_sequence)}")
            break
    # This is deliberately separate from the hard max in deterministic_score:
    # it makes trajectory inefficiency visible even when a run is otherwise valid.
    if len(actual) != len(case.preferred_sequence):
        reasons.append(f"inefficient path: expected {len(case.preferred_sequence)} calls, got {len(actual)}")
    return ScoreResult(not reasons, 1.0 if not reasons else 0.0, tuple(reasons))
