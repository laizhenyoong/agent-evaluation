import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.contracts import EvaluationCase
from evaluation.judge import (
    Judgement,
    OllamaJudge,
    judge_traces,
    load_judgements,
    write_judgements,
)
from evaluation.traces import AgentTrace

CASE = EvaluationCase(
    "kb_refund_policy",
    "What is your refund policy?",
    reference_facts=("Refunds are issued within 14 days of purchase.",),
)
TRACE = AgentTrace("kb_refund_policy", CASE.prompt, answer="Refunds take 14 days.")


class JudgeTests(unittest.TestCase):
    def test_a_verdict_maps_onto_the_three_labels(self) -> None:
        verdict = {
            "factually_correct": True,
            "faithful_to_facts": True,
            "tone_appropriate": True,
            "reason": "matches the policy",
        }
        judge = OllamaJudge(model="test-model")
        with patch.object(OllamaJudge, "_ask", return_value=verdict):
            judgement = judge.judge(CASE, TRACE)

        self.assertTrue(judgement.passed)
        self.assertEqual(judgement.judge_model, "test-model")
        self.assertIsNone(judgement.error)

    def test_reference_facts_reach_the_prompt_as_whole_sentences(self) -> None:
        seen: dict[str, str] = {}

        def capture_prompt(self: OllamaJudge, prompt: str) -> dict:
            seen["prompt"] = prompt
            return {"factually_correct": True, "faithful_to_facts": True, "tone_appropriate": True, "reason": ""}

        with patch.object(OllamaJudge, "_ask", capture_prompt):
            OllamaJudge(model="m").judge(CASE, TRACE)

        self.assertIn("- Refunds are issued within 14 days of purchase.", seen["prompt"])

    def test_an_unreachable_judge_is_marked_unusable_not_failed(self) -> None:
        with patch.object(OllamaJudge, "_ask", side_effect=TimeoutError("no server")):
            judgement = OllamaJudge(model="m").judge(CASE, TRACE)

        self.assertIsNotNone(judgement.error)
        self.assertFalse(judgement.passed)

    def test_an_errored_trace_is_not_sent_to_the_judge(self) -> None:
        errored = AgentTrace("kb_refund_policy", CASE.prompt, error="RuntimeError: down")
        with patch.object(OllamaJudge, "_ask", side_effect=AssertionError("should not be called")):
            judgement = OllamaJudge(model="m").judge(CASE, errored)

        self.assertIn("agent errored", judgement.error or "")

    def test_a_trace_without_a_golden_case_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "ghost"):
            judge_traces([AgentTrace("ghost", "p")], OllamaJudge(model="m"), cases=[CASE])


class JudgementFileTests(unittest.TestCase):
    def test_judgements_survive_a_write_and_load_cycle(self) -> None:
        judgements = [
            Judgement("a", True, True, False, "tone was curt", judge_model="m"),
            Judgement("b", False, False, False, "unreachable", judge_model="m", error="timeout"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "judgements.jsonl"
            write_judgements(path, judgements)
            self.assertEqual(load_judgements(path), judgements)


if __name__ == "__main__":
    unittest.main()
