"""An opt-in local LLM judge that grades stored answers, never live runs.

The judge is the expensive, non-deterministic half of scoring, so it is kept
off the capture path entirely.  It reads a trace file and writes a judgement
file, which means iterating on the judge prompt costs one judge sweep rather
than 50 fresh agent trajectories -- and means two judges can be pointed at the
same answers and compared (see `evaluation.agreement`).

Usage:
    .venv/bin/python -m evaluation.judge --traces results/baseline.jsonl
"""

import argparse
import json
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from evaluation.contracts import EvaluationCase
from evaluation.golden_dataset import GOLDEN_CASES
from evaluation.traces import AgentTrace, load_traces

# The boolean verdicts every judgement carries; `evaluation.agreement` compares
# these label by label rather than collapsing them into one number.
LABELS = ("factually_correct", "faithful_to_facts", "tone_appropriate")


@dataclass(frozen=True)
class Judgement:
    """One judge's verdict on one stored answer."""

    case_id: str
    factually_correct: bool
    faithful_to_facts: bool
    tone_appropriate: bool
    reason: str
    judge_model: str | None = None
    error: str | None = None

    @property
    def passed(self) -> bool:
        return all(getattr(self, label) for label in LABELS)


PROMPT_TEMPLATE = """You are grading a customer-support agent's answer.
Return only JSON with keys factually_correct (boolean), faithful_to_facts
(boolean), tone_appropriate (boolean), and reason (one short string).

Customer request: {prompt}
Reference facts the answer must not contradict:
{facts}
Required tone: {tone}
Agent answer: {answer}

factually_correct: the answer is right about the customer's question.
faithful_to_facts: it neither contradicts the reference facts nor claims an
action it did not take.
tone_appropriate: it matches the required tone.
Do not grade tool use; a separate scorer does that."""


class OllamaJudge:
    """Grade correctness, faithfulness, and tone using Ollama's chat API."""

    def __init__(self, host: str | None = None, model: str | None = None) -> None:
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.model = model or os.environ.get("EVAL_JUDGE_MODEL", "qwen3:30b")

    def judge(self, case: EvaluationCase, trace: AgentTrace) -> Judgement:
        if trace.error is not None:
            return _failed(case.case_id, self.model, f"not judged, agent errored: {trace.error}")

        facts = "\n".join(f"- {fact}" for fact in case.reference_facts) or "- (none supplied)"
        prompt = PROMPT_TEMPLATE.format(
            prompt=case.prompt, facts=facts, tone=case.tone, answer=trace.answer or ""
        )
        try:
            verdict = self._ask(prompt)
        except (URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as exception:
            return _failed(
                case.case_id, self.model, f"judge unavailable or returned invalid JSON: {exception}"
            )

        return Judgement(
            case_id=case.case_id,
            factually_correct=bool(verdict.get("factually_correct", False)),
            faithful_to_facts=bool(verdict.get("faithful_to_facts", False)),
            tone_appropriate=bool(verdict.get("tone_appropriate", False)),
            reason=str(verdict.get("reason", "")),
            judge_model=self.model,
        )

    def _ask(self, prompt: str) -> dict:
        body = json.dumps(
            {
                "model": self.model,
                "stream": False,
                "format": "json",
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode()
        request = Request(
            f"{self.host}/api/chat", data=body, headers={"Content-Type": "application/json"}
        )
        with urlopen(request, timeout=90) as response:
            payload = json.loads(response.read())
        return json.loads(payload["message"]["content"])


def _failed(case_id: str, model: str, reason: str) -> Judgement:
    """An unusable verdict, marked so agreement scoring can skip it.

    Placeholder falses are not an opinion; counting them would credit two
    judges with agreeing about something neither actually judged.
    """
    return Judgement(case_id, False, False, False, reason, judge_model=model, error=reason)


def judge_traces(
    traces: Sequence[AgentTrace],
    judge: OllamaJudge,
    cases: Sequence[EvaluationCase] = GOLDEN_CASES,
) -> list[Judgement]:
    """Grade every trace that has a matching golden case."""
    cases_by_id = {case.case_id: case for case in cases}
    missing = [trace.case_id for trace in traces if trace.case_id not in cases_by_id]
    if missing:
        raise ValueError(f"No golden case for traces: {', '.join(sorted(set(missing)))}")
    return [judge.judge(cases_by_id[trace.case_id], trace) for trace in traces]


def write_judgements(path: Path, judgements: Sequence[Judgement]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = (json.dumps(asdict(judgement), ensure_ascii=False) for judgement in judgements)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_judgements(path: Path) -> list[Judgement]:
    judgements: list[Judgement] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            judgements.append(
                Judgement(
                    case_id=record["case_id"],
                    factually_correct=record["factually_correct"],
                    faithful_to_facts=record["faithful_to_facts"],
                    tone_appropriate=record["tone_appropriate"],
                    reason=record["reason"],
                    judge_model=record.get("judge_model"),
                    error=record.get("error"),
                )
            )
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid judgement on line {line_number} of {path}") from error
    return judgements


def main() -> int:
    parser = argparse.ArgumentParser(description="Grade stored answers with a local LLM judge.")
    parser.add_argument("--traces", type=Path, default=Path("results/latest.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/judgements.jsonl"))
    parser.add_argument("--model", help="Judge model; defaults to EVAL_JUDGE_MODEL or qwen3:30b.")
    arguments = parser.parse_args()

    judgements = judge_traces(load_traces(arguments.traces), OllamaJudge(model=arguments.model))
    write_judgements(arguments.output, judgements)
    usable = [judgement for judgement in judgements if judgement.error is None]
    passed = sum(judgement.passed for judgement in usable)
    print(
        f"Judged {len(usable)}/{len(judgements)} answers into {arguments.output}. "
        f"passed={passed} unusable={len(judgements) - len(usable)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
