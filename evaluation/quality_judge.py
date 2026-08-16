"""An opt-in local Ollama LLM-as-judge for qualitative response checks."""

import json
import os
from urllib.error import URLError
from urllib.request import Request, urlopen

from evaluation.contracts import EvaluationCase
from evaluation.scoring import ScoreResult
from evaluation.traces import AgentTrace


class OllamaResponseJudge:
    """Judge correctness, KB faithfulness, and tone using Ollama's chat API."""

    def __init__(self, host: str | None = None, model: str | None = None) -> None:
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        self.model = model or os.environ.get("EVAL_JUDGE_MODEL", "qwen3:30b")

    def score(self, case: EvaluationCase, run: AgentTrace) -> ScoreResult:
        prompt = f"""You are grading a customer-support agent response.
Return only JSON with keys passed (boolean), score (number from 0 to 1), and reasons (array of short strings).

Customer request: {case.prompt}
Reference facts: {list(case.reference_facts)}
Required tone: {case.tone}
Agent answer: {run.answer}

Mark passed only if the answer is factually correct, does not contradict the
reference facts or invent unsupported actions, and has the required tone.
Do not grade tool use; another scorer does that."""
        body = json.dumps(
            {
                "model": self.model,
                "stream": False,
                "format": "json",
                "messages": [{"role": "user", "content": prompt}],
            }
        ).encode()
        request = Request(
            f"{self.host}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=90) as response:
                payload = json.loads(response.read())
            verdict = json.loads(payload["message"]["content"])
        except (URLError, TimeoutError, KeyError, ValueError, json.JSONDecodeError) as exc:
            return ScoreResult(
                False,
                0.0,
                (f"judge unavailable or returned invalid JSON: {exc}",),
            )

        passed = bool(verdict.get("passed", False))
        score = float(verdict.get("score", 1.0 if passed else 0.0))
        reasons = verdict.get("reasons", [])
        if not isinstance(reasons, list):
            reasons = [str(reasons)]
        return ScoreResult(
            passed,
            max(0.0, min(1.0, score)),
            tuple(str(reason) for reason in reasons),
        )
