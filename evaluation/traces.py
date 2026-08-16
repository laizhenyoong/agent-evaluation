"""One inspectable record per agent run, persisted as JSONL.

Capturing a trajectory and judging it are separate jobs.  Writing the run to
disk first means a scorer, an LLM judge, or a failure-mining script can all
read the same evidence later without paying to re-run the agent -- and means a
committed trace file can act as the baseline a CI gate compares against.

Every trace carries the `agent_config` that produced it, so a pass-rate change
between two runs is attributable to a specific prompt or model rather than
being an unexplained drift.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ObservedToolCall:
    """A single tool invocation the agent actually made."""

    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class AgentTrace:
    """Everything needed to score one case without re-running the agent."""

    case_id: str
    prompt: str
    agent_config: dict[str, Any] = field(default_factory=dict)
    tool_calls: tuple[ObservedToolCall, ...] = ()
    answer: str | None = None
    timing_ms: int | None = None
    error: str | None = None


def write_traces(path: Path, traces: list[AgentTrace]) -> None:
    """Write one JSON object per trace, creating the output directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = (json.dumps(asdict(trace), ensure_ascii=False) for trace in traces)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_traces(path: Path) -> list[AgentTrace]:
    """Load a JSONL file written by `write_traces` for offline scoring."""
    traces: list[AgentTrace] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            traces.append(
                AgentTrace(
                    case_id=record["case_id"],
                    prompt=record["prompt"],
                    agent_config=record.get("agent_config", {}),
                    tool_calls=tuple(
                        ObservedToolCall(call["name"], call["arguments"])
                        for call in record.get("tool_calls", [])
                    ),
                    answer=record.get("answer"),
                    timing_ms=record.get("timing_ms"),
                    error=record.get("error"),
                )
            )
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid trace on line {line_number} of {path}") from error
    return traces
