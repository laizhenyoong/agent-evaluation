import unittest

from evaluation.contracts import EvaluationCase, ToolCallExpectation as Tool
from evaluation.golden_dataset import GOLDEN_CASES
from evaluation.scoring import (
    ScoreResult,
    deterministic_score,
    summarize,
    trajectory_score,
)
from evaluation.traces import AgentTrace, ObservedToolCall


def trace(case: EvaluationCase, *calls: tuple[str, dict], answer: str = "ok", error: str | None = None) -> AgentTrace:
    return AgentTrace(
        case_id=case.case_id,
        prompt=case.prompt,
        tool_calls=tuple(ObservedToolCall(name, arguments) for name, arguments in calls),
        answer=answer,
        error=error,
    )


def case(case_id: str) -> EvaluationCase:
    return next(golden for golden in GOLDEN_CASES if golden.case_id == case_id)


class DeterministicScoreTests(unittest.TestCase):
    def test_accepts_the_required_call_and_arguments(self) -> None:
        result = deterministic_score(
            case("account_alice_plan"),
            trace(case("account_alice_plan"), ("check_account", {"account_id": "A123"})),
        )
        self.assertTrue(result.passed)
        self.assertEqual(result.score, 1.0)

    def test_every_golden_contract_has_a_passing_reference_trajectory(self) -> None:
        for golden in GOLDEN_CASES:
            calls = tuple(
                (expected.name, expected.exact_args | expected.contains_args)
                for expected in golden.required_tools
            )
            with self.subTest(case=golden.case_id):
                reference = trace(golden, *calls)
                self.assertTrue(deterministic_score(golden, reference).passed)
                self.assertTrue(trajectory_score(golden, reference).passed)

    def test_a_wrong_argument_and_an_over_budget_run_are_rejected(self) -> None:
        result = deterministic_score(
            case("account_alice_plan"),
            trace(
                case("account_alice_plan"),
                ("check_account", {"account_id": "B456"}),
                ("check_account", {"account_id": "A123"}),
            ),
        )
        self.assertFalse(result.passed)
        self.assertTrue(any("too many" in reason for reason in result.reasons))
        self.assertTrue(any("loop" in reason for reason in result.reasons))

    def test_partial_credit_sits_between_nothing_and_everything(self) -> None:
        golden = case("ticket_password_alice")  # requires search_kb then create_ticket
        none_found = deterministic_score(golden, trace(golden))
        one_found = deterministic_score(
            golden, trace(golden, ("search_kb", {"query": "password"}))
        )
        both_found = deterministic_score(
            golden,
            trace(
                golden,
                ("search_kb", {"query": "password"}),
                ("create_ticket", {"account_id": "A123"}),
            ),
        )
        self.assertLess(none_found.score, one_found.score)
        self.assertLess(one_found.score, both_found.score)
        self.assertEqual(both_found.score, 1.0)

    def test_a_legitimate_repeat_within_contract_is_not_called_a_loop(self) -> None:
        golden = EvaluationCase(
            "twice",
            "look up both accounts",
            required_tools=(
                Tool("check_account", exact_args={"account_id": "A123"}),
                Tool("check_account", exact_args={"account_id": "B456"}),
            ),
            preferred_sequence=("check_account", "check_account"),
            max_tool_calls=2,
        )
        result = deterministic_score(
            golden,
            trace(golden, ("check_account", {"account_id": "A123"}), ("check_account", {"account_id": "B456"})),
        )
        self.assertTrue(result.passed, result.reasons)

    def test_an_agent_error_scores_zero_and_says_why(self) -> None:
        golden = case("account_alice_plan")
        result = deterministic_score(golden, trace(golden, error="RuntimeError: down"))
        self.assertFalse(result.passed)
        self.assertEqual(result.score, 0.0)
        self.assertIn("agent error", result.reasons[0])


class TrajectoryScoreTests(unittest.TestCase):
    def test_rejects_a_correct_but_wasteful_path(self) -> None:
        golden = case("account_alice_plan")
        result = trajectory_score(
            golden,
            trace(golden, ("search_kb", {"query": "plan"}), ("check_account", {"account_id": "A123"})),
        )
        self.assertFalse(result.passed)
        self.assertTrue(any("unnecessary" in reason for reason in result.reasons))

    def test_one_stray_call_scores_between_a_perfect_and_an_empty_path(self) -> None:
        golden = case("ticket_password_alice")  # preferred: search_kb -> create_ticket
        perfect = trajectory_score(
            golden,
            trace(golden, ("search_kb", {"query": "password"}), ("create_ticket", {"account_id": "A123"})),
        )
        detour = trajectory_score(
            golden,
            trace(
                golden,
                ("search_kb", {"query": "password"}),
                ("search_kb", {"query": "password again"}),
                ("create_ticket", {"account_id": "A123"}),
            ),
        )
        empty = trajectory_score(golden, trace(golden))

        self.assertEqual(perfect.score, 1.0)
        self.assertEqual(empty.score, 0.0)
        self.assertLess(detour.score, perfect.score)
        self.assertGreater(detour.score, empty.score)
        self.assertFalse(detour.passed)

    def test_out_of_order_costs_coverage(self) -> None:
        golden = case("ticket_password_alice")
        reversed_path = trajectory_score(
            golden,
            trace(golden, ("create_ticket", {"account_id": "A123"}), ("search_kb", {"query": "password"})),
        )
        self.assertFalse(reversed_path.passed)
        self.assertLess(reversed_path.score, 1.0)
        self.assertTrue(any("out of order" in reason for reason in reversed_path.reasons))


class SummaryTests(unittest.TestCase):
    def test_summary_reports_both_a_pass_rate_and_a_mean_score(self) -> None:
        summary = summarize(
            [
                ScoreResult(True, 1.0, ()),
                ScoreResult(False, 0.5, ("inefficient path",)),
                ScoreResult(False, 0.0, ("agent error: RuntimeError: down",)),
                ScoreResult(True, 1.0, ()),
            ]
        )
        self.assertEqual(summary.scored_count, 4)
        self.assertEqual(summary.error_count, 1)
        self.assertEqual(summary.pass_rate, 0.5)
        self.assertEqual(summary.mean_score, 0.625)

    def test_an_empty_run_summarizes_to_zero_rather_than_dividing_by_zero(self) -> None:
        self.assertEqual(summarize([]).pass_rate, 0.0)


if __name__ == "__main__":
    unittest.main()
