import unittest

from evaluation.golden_dataset import GOLDEN_CASES
from evaluation.traces import AgentTrace, ObservedToolCall
from evaluation.scoring import deterministic_score, trajectory_score


class EvaluationTests(unittest.TestCase):
    def test_deterministic_scorer_accepts_required_call_and_arguments(self) -> None:
        case = next(case for case in GOLDEN_CASES if case.case_id == "account_alice_plan")
        run = AgentTrace(case.case_id, case.prompt, {}, (ObservedToolCall("check_account", {"account_id": "A123"}),), "Alice is on Pro.")
        self.assertTrue(deterministic_score(case, run).passed)

    def test_every_golden_contract_has_a_passing_reference_trajectory(self) -> None:
        for case in GOLDEN_CASES:
            calls = tuple(
                ObservedToolCall(expectation.name, expectation.exact_args | expectation.contains_args)
                for expectation in case.required_tools
            )
            run = AgentTrace(case.case_id, case.prompt, {}, calls, "reference answer")
            with self.subTest(case=case.case_id):
                self.assertTrue(deterministic_score(case, run).passed)
                self.assertTrue(trajectory_score(case, run).passed)

    def test_deterministic_scorer_rejects_wrong_argument_and_loop(self) -> None:
        case = next(case for case in GOLDEN_CASES if case.case_id == "account_alice_plan")
        run = AgentTrace(case.case_id, case.prompt, {}, (ObservedToolCall("check_account", {"account_id": "B456"}), ObservedToolCall("check_account", {"account_id": "A123"})), "")
        result = deterministic_score(case, run)
        self.assertFalse(result.passed)
        self.assertTrue(any("too many" in reason for reason in result.reasons))
        self.assertTrue(any("loop" in reason for reason in result.reasons))

    def test_trajectory_scorer_rejects_correct_but_wasteful_path(self) -> None:
        case = next(case for case in GOLDEN_CASES if case.case_id == "account_alice_plan")
        run = AgentTrace(case.case_id, case.prompt, {}, (ObservedToolCall("search_kb", {"query": "plan"}), ObservedToolCall("check_account", {"account_id": "A123"})), "Alice is on Pro.")
        result = trajectory_score(case, run)
        self.assertFalse(result.passed)
        self.assertTrue(any("unnecessary" in reason for reason in result.reasons))


if __name__ == "__main__":
    unittest.main()
