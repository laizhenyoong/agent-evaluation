import unittest
from unittest.mock import patch

from evaluation.capture import capture, capture_case, select_cases
from evaluation.contracts import EvaluationCase


class FakeAgent:
    """Stand in for a Strands agent without touching Ollama."""

    def __init__(self, messages: list[dict], answer: str = "done") -> None:
        self.messages = messages
        self._answer = answer

    def __call__(self, prompt: str) -> str:
        return self._answer


def fake_messages(*calls: tuple[str, dict]) -> list[dict]:
    return [
        {"content": [{"toolUse": {"name": name, "input": arguments}} for name, arguments in calls]}
    ]


CASE = EvaluationCase("c1", "check A123", reference_facts=("A123 is active.",))


class CaptureTests(unittest.TestCase):
    def test_capture_records_tool_calls_answer_and_config(self) -> None:
        agent = FakeAgent(fake_messages(("check_account", {"account_id": "A123"})), "A123 is active.")
        with patch("evaluation.capture.build_agent", return_value=agent):
            trace = capture_case(CASE, model_id="m", system_prompt="sp")

        self.assertIsNone(trace.error)
        self.assertEqual(trace.answer, "A123 is active.")
        self.assertEqual(trace.tool_calls[0].name, "check_account")
        self.assertEqual(trace.tool_calls[0].arguments, {"account_id": "A123"})
        self.assertEqual(trace.agent_config, {"model_id": "m", "system_prompt": "sp"})
        self.assertIsNotNone(trace.timing_ms)

    def test_an_agent_crash_becomes_a_trace_rather_than_an_exception(self) -> None:
        with patch("evaluation.capture.build_agent", side_effect=RuntimeError("model unreachable")):
            trace = capture_case(CASE, model_id="m", system_prompt="sp")

        self.assertEqual(trace.error, "RuntimeError: model unreachable")
        self.assertEqual(trace.tool_calls, ())
        self.assertIsNone(trace.answer)

    def test_one_failing_case_does_not_lose_the_rest_of_the_sweep(self) -> None:
        cases = [EvaluationCase(f"c{index}", "p") for index in range(4)]
        agents = [
            FakeAgent(fake_messages(("search_kb", {"query": "refund"}))),
            RuntimeError("boom"),
            FakeAgent(fake_messages(("search_kb", {"query": "hours"}))),
            RuntimeError("boom again"),
        ]

        def build(**_: object) -> FakeAgent:
            agent = agents.pop(0)
            if isinstance(agent, Exception):
                raise agent
            return agent

        with patch("evaluation.capture.build_agent", side_effect=build):
            traces = capture(cases, model_id="m", system_prompt="sp")

        self.assertEqual(len(traces), 4)
        self.assertEqual([trace.error is None for trace in traces], [True, False, True, False])


class SelectCasesTests(unittest.TestCase):
    def test_no_ids_selects_the_whole_golden_set(self) -> None:
        self.assertEqual(len(select_cases(None)), 50)

    def test_unknown_ids_are_rejected_instead_of_silently_dropped(self) -> None:
        with self.assertRaisesRegex(ValueError, "nope"):
            select_cases(["kb_refund_policy", "nope"])


if __name__ == "__main__":
    unittest.main()
