"""Deterministic and trajectory-level scorers over persisted traces.

These scorers deliberately inspect observable behaviour rather than inferring
it from the final answer.  A model can sound helpful while taking a needless or
unsafe path; that is still a failed agent run.

Scores are graded rather than binary.  A binary scorer cannot tell "took one
wasteful detour" from "did something completely different", so a prompt change
that halves the wasted calls looks identical to one that changed nothing.
`passed` remains available for the CI gate as the strict case, score == 1.0.
"""

import statistics
from collections import Counter
from dataclasses import dataclass

from evaluation.contracts import EvaluationCase, ToolCallExpectation
from evaluation.traces import AgentTrace, ObservedToolCall

_AGENT_ERROR = "agent error"


@dataclass(frozen=True)
class ScoreResult:
    passed: bool
    score: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class ScoreSummary:
    """Aggregate one scorer across a run."""

    scored_count: int
    error_count: int
    pass_rate: float
    mean_score: float


def _matches(call: ObservedToolCall, expected: ToolCallExpectation) -> bool:
    if call.name != expected.name:
        return False
    if any(call.arguments.get(key) != value for key, value in expected.exact_args.items()):
        return False
    return all(
        expected_value.lower() in str(call.arguments.get(key, "")).lower()
        for key, expected_value in expected.contains_args.items()
    )


def deterministic_score(case: EvaluationCase, trace: AgentTrace) -> ScoreResult:
    """Check required calls and arguments, forbidden calls, budget, and repeats.

    The score is the fraction of the contract's individual checks that held, so
    a run finding two of three required calls scores above one finding none.
    """
    if trace.error is not None:
        return ScoreResult(False, 0.0, (f"{_AGENT_ERROR}: {trace.error}",))

    checks: list[bool] = []
    reasons: list[str] = []
    calls = trace.tool_calls

    for expected in case.required_tools:
        found = any(_matches(call, expected) for call in calls)
        checks.append(found)
        if not found:
            arguments = expected.exact_args or expected.contains_args
            reasons.append(f"missing required call: {expected.name}({arguments})")

    forbidden = sorted({call.name for call in calls if call.name in case.forbidden_tools})
    checks.append(not forbidden)
    if forbidden:
        reasons.append(f"forbidden tool calls: {', '.join(forbidden)}")

    within_budget = len(calls) <= case.max_tool_calls
    checks.append(within_budget)
    if not within_budget:
        reasons.append(f"too many tool calls: {len(calls)} > {case.max_tool_calls}")

    # A repeat is only a retry loop when it exceeds what the contract asked for.
    # Flagging every duplicate name would fail any legitimate multi-call path.
    expected_counts = Counter(expected.name for expected in case.required_tools)
    actual_counts = Counter(call.name for call in calls)
    excess = sorted(
        name for name, count in actual_counts.items() if count > max(expected_counts[name], 1)
    )
    checks.append(not excess)
    if excess:
        reasons.append(f"tool loop/retry detected: {', '.join(excess)}")

    score = statistics.fmean(checks) if checks else 1.0
    return ScoreResult(all(checks), round(score, 3), tuple(reasons))


def trajectory_score(case: EvaluationCase, trace: AgentTrace) -> ScoreResult:
    """Score whether the path was economical and followed the intended order.

    Two graded components, averaged:

    * coverage   -- how much of the preferred sequence appears, in order
    * efficiency -- how close the call count is to the preferred count

    An agent that walks the right path but adds one stray call lands well below
    1.0 without collapsing to 0.0, which is what makes a before/after
    comparison legible.
    """
    if trace.error is not None:
        return ScoreResult(False, 0.0, (f"{_AGENT_ERROR}: {trace.error}",))

    actual = tuple(call.name for call in trace.tool_calls)
    preferred = case.preferred_sequence
    if not preferred:
        if actual:
            return ScoreResult(False, 0.0, (f"expected no tool calls, got {len(actual)}",))
        return ScoreResult(True, 1.0, ())

    reasons: list[str] = []
    matched = _ordered_overlap(actual, preferred)
    coverage = matched / len(preferred)
    if matched < len(preferred):
        reasons.append(f"preferred path missing or out of order: {' -> '.join(preferred)}")

    efficiency = min(len(preferred) / len(actual), 1.0) if actual else 0.0
    if len(actual) != len(preferred):
        reasons.append(f"inefficient path: expected {len(preferred)} calls, got {len(actual)}")

    allowed = {expected.name for expected in case.required_tools} | set(preferred)
    extras = sorted({name for name in actual if name not in allowed})
    if extras:
        reasons.append(f"unnecessary tools: {', '.join(extras)}")

    return ScoreResult(
        passed=not reasons,
        score=round(statistics.fmean((coverage, efficiency)), 3),
        reasons=tuple(reasons),
    )


def _ordered_overlap(actual: tuple[str, ...], preferred: tuple[str, ...]) -> int:
    """How many of `preferred` appear in `actual` in order, greedily matched."""
    position = 0
    matched = 0
    for name in preferred:
        try:
            position = actual.index(name, position) + 1
        except ValueError:
            break
        matched += 1
    return matched


def summarize(results: list[ScoreResult]) -> ScoreSummary:
    """Aggregate a scorer's results into the rates a regression gate compares."""
    if not results:
        return ScoreSummary(0, 0, 0.0, 0.0)
    return ScoreSummary(
        scored_count=len(results),
        error_count=sum(
            any(reason.startswith(_AGENT_ERROR) for reason in result.reasons) for result in results
        ),
        pass_rate=round(statistics.fmean([result.passed for result in results]), 3),
        mean_score=round(statistics.fmean([result.score for result in results]), 3),
    )
