import unittest

from evaluation.contracts import EvaluationCase, ToolCallExpectation as Tool
from evaluation.judge import Judgement
from evaluation.report import build_report, check_regressions, summary_only
from evaluation.traces import AgentTrace, ObservedToolCall

CASES = (
    EvaluationCase(
        "lookup",
        "What plan is A123 on?",
        required_tools=(Tool("check_account", exact_args={"account_id": "A123"}),),
        preferred_sequence=("check_account",),
        max_tool_calls=1,
        reference_facts=("A123 is on the Pro plan.",),
    ),
)


def trace(*calls: tuple[str, dict], error: str | None = None) -> AgentTrace:
    return AgentTrace(
        case_id="lookup",
        prompt="What plan is A123 on?",
        agent_config={"model_id": "m", "system_prompt": "sp"},
        tool_calls=tuple(ObservedToolCall(name, arguments) for name, arguments in calls),
        answer="Pro plan.",
        error=error,
    )


class BuildReportTests(unittest.TestCase):
    def test_a_perfect_run_summarizes_to_one(self) -> None:
        report = build_report([trace(("check_account", {"account_id": "A123"}))], cases=CASES)
        self.assertEqual(report["deterministic"]["pass_rate"], 1.0)
        self.assertEqual(report["trajectory"]["mean_score"], 1.0)
        self.assertEqual(report["agent_config"]["model_id"], "m")

    def test_an_errored_trace_is_counted_not_dropped(self) -> None:
        report = build_report([trace(error="RuntimeError: down")], cases=CASES)
        self.assertEqual(report["deterministic"]["scored_count"], 1)
        self.assertEqual(report["deterministic"]["error_count"], 1)
        self.assertEqual(report["deterministic"]["pass_rate"], 0.0)

    def test_judgements_are_folded_in_per_label(self) -> None:
        judgements = [Judgement("lookup", True, True, False, "curt", "m")]
        report = build_report(
            [trace(("check_account", {"account_id": "A123"}))], judgements, cases=CASES
        )
        self.assertEqual(report["judge"]["factually_correct_rate"], 1.0)
        self.assertEqual(report["judge"]["tone_appropriate_rate"], 0.0)
        self.assertEqual(report["judge"]["pass_rate"], 0.0)

    def test_unusable_judgements_are_excluded_from_the_rates(self) -> None:
        judgements = [Judgement("lookup", False, False, False, "timeout", "m", error="timeout")]
        report = build_report(
            [trace(("check_account", {"account_id": "A123"}))], judgements, cases=CASES
        )
        self.assertEqual(report["judge"]["judged_count"], 0)
        self.assertEqual(report["judge"]["unusable_count"], 1)

    def test_a_trace_without_a_golden_case_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "ghost"):
            build_report([AgentTrace("ghost", "p")], cases=CASES)

    def test_the_baseline_summary_omits_per_case_rows(self) -> None:
        report = build_report([trace(("check_account", {"account_id": "A123"}))], cases=CASES)
        self.assertNotIn("cases", summary_only(report))
        self.assertIn("deterministic", summary_only(report))


class RegressionGateTests(unittest.TestCase):
    BASELINE = {
        "deterministic": {"pass_rate": 0.80, "mean_score": 0.90},
        "trajectory": {"pass_rate": 0.60, "mean_score": 0.75},
    }

    def test_an_unchanged_run_passes(self) -> None:
        self.assertEqual(check_regressions(dict(self.BASELINE), self.BASELINE), [])

    def test_an_improved_run_passes(self) -> None:
        better = {
            "deterministic": {"pass_rate": 0.90, "mean_score": 0.95},
            "trajectory": {"pass_rate": 0.70, "mean_score": 0.80},
        }
        self.assertEqual(check_regressions(better, self.BASELINE), [])

    def test_a_dropped_rate_is_reported_with_both_numbers(self) -> None:
        worse = {
            "deterministic": {"pass_rate": 0.70, "mean_score": 0.90},
            "trajectory": {"pass_rate": 0.60, "mean_score": 0.75},
        }
        failures = check_regressions(worse, self.BASELINE)
        self.assertEqual(len(failures), 1)
        self.assertIn("deterministic.pass_rate", failures[0])
        self.assertIn("0.700", failures[0])
        self.assertIn("0.800", failures[0])

    def test_a_graded_drop_is_caught_even_when_the_pass_rate_holds(self) -> None:
        # The point of grading: the agent still fails the same cases, but is
        # taking more wasteful paths to get there.
        worse = {
            "deterministic": {"pass_rate": 0.80, "mean_score": 0.90},
            "trajectory": {"pass_rate": 0.60, "mean_score": 0.65},
        }
        failures = check_regressions(worse, self.BASELINE)
        self.assertEqual(len(failures), 1)
        self.assertIn("trajectory.mean_score", failures[0])

    def test_tolerance_absorbs_small_noise(self) -> None:
        worse = {
            "deterministic": {"pass_rate": 0.78, "mean_score": 0.90},
            "trajectory": {"pass_rate": 0.60, "mean_score": 0.75},
        }
        self.assertEqual(check_regressions(worse, self.BASELINE, tolerance=0.05), [])
        self.assertEqual(len(check_regressions(worse, self.BASELINE, tolerance=0.01)), 1)

    def test_a_metric_absent_from_the_baseline_is_skipped(self) -> None:
        self.assertEqual(check_regressions({"deterministic": {"pass_rate": 0.1}}, {}), [])

    def test_a_negative_tolerance_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            check_regressions({}, {}, tolerance=-0.1)


if __name__ == "__main__":
    unittest.main()
