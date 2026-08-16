import json
import tempfile
import unittest
from pathlib import Path

from evaluation.contracts import EvaluationCase
from evaluation.golden_dataset import GOLDEN_CASES
from evaluation.traces import AgentTrace, ObservedToolCall, load_traces, write_traces


class TraceRoundTripTests(unittest.TestCase):
    def test_traces_survive_a_write_and_load_cycle(self) -> None:
        traces = [
            AgentTrace(
                case_id="kb_refund_policy",
                prompt="What is your refund policy?",
                agent_config={"model_id": "qwen3:30b", "system_prompt": "be helpful"},
                tool_calls=(ObservedToolCall("search_kb", {"query": "refund"}),),
                answer="Refunds within 14 days.",
                timing_ms=1234,
            ),
            AgentTrace(case_id="broken", prompt="x", error="RuntimeError: model unreachable"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "traces.jsonl"
            write_traces(path, traces)
            self.assertEqual(load_traces(path), traces)

    def test_each_line_is_one_self_describing_json_object(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "traces.jsonl"
            write_traces(path, [AgentTrace("a", "p"), AgentTrace("b", "q")])
            lines = path.read_text().strip().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(json.loads(lines[0])["case_id"], "a")

    def test_a_corrupt_line_names_its_line_number(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "traces.jsonl"
            path.write_text('{"case_id": "a", "prompt": "p"}\nnot json\n')
            with self.assertRaisesRegex(ValueError, "line 2"):
                load_traces(path)


class ContractTests(unittest.TestCase):
    def test_golden_set_has_fifty_cases_with_unique_ids(self) -> None:
        self.assertEqual(len(GOLDEN_CASES), 50)
        self.assertEqual(len({case.case_id for case in GOLDEN_CASES}), 50)

    def test_every_case_carries_reference_facts_as_a_tuple_of_whole_strings(self) -> None:
        for case in GOLDEN_CASES:
            with self.subTest(case=case.case_id):
                self.assertIsInstance(case.reference_facts, tuple)
                for fact in case.reference_facts:
                    # A stray ("fact") would leave single characters here.
                    self.assertGreater(len(fact), 1)

    def test_a_bare_string_of_reference_facts_is_rejected_at_construction(self) -> None:
        with self.assertRaisesRegex(TypeError, "reference_facts"):
            EvaluationCase("c", "p", reference_facts=("one fact"))


if __name__ == "__main__":
    unittest.main()
