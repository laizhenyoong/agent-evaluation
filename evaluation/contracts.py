"""What a golden case *requires*, stated independently of any observed run.

A contract is the intent: which evidence the agent must gather, which actions
it must not take, and what a correct answer may claim.  Nothing here knows how
a run turned out; that lives in `evaluation.traces`.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ToolCallExpectation:
    """A required call, with exact or substring checks for selected arguments."""

    name: str
    exact_args: dict[str, Any] = field(default_factory=dict)
    contains_args: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    prompt: str
    required_tools: tuple[ToolCallExpectation, ...] = ()
    forbidden_tools: tuple[str, ...] = ()
    preferred_sequence: tuple[str, ...] = ()
    max_tool_calls: int = 3
    reference_facts: tuple[str, ...] = ()
    tone: str = "professional, calm, and helpful"

    def __post_init__(self) -> None:
        # A bare string is iterable, so a stray `("fact")` would silently reach
        # the judge as a list of single characters instead of one fact.
        if isinstance(self.reference_facts, str):
            raise TypeError(
                f"{self.case_id}: reference_facts must be a tuple of strings, not a string. "
                'Did you write ("fact") instead of ("fact",)?'
            )
