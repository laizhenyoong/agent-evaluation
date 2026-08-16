import unittest

from evaluation.agreement import compare, disagreements
from evaluation.judge import Judgement


def judgement(
    case_id: str,
    correct: bool,
    faithful: bool = True,
    tone: bool = True,
    error: str | None = None,
) -> Judgement:
    return Judgement(case_id, correct, faithful, tone, "reason", "m", error)


def label(report, name: str):
    return next(entry for entry in report.labels if entry.label == name)


class AgreementTests(unittest.TestCase):
    def test_identical_verdicts_agree_perfectly(self) -> None:
        left = [judgement("a", True), judgement("b", False)]
        report = compare(left, list(left))
        self.assertEqual(report.compared_count, 2)
        self.assertEqual(label(report, "factually_correct").agreement_rate, 1.0)
        self.assertEqual(label(report, "factually_correct").cohen_kappa, 1.0)

    def test_kappa_discounts_agreement_that_is_only_luck(self) -> None:
        # Both sides say True on 9 of 10, but disagree on which one is False,
        # so raw agreement is high while kappa is not.
        baseline = [judgement(f"c{index}", index != 0) for index in range(10)]
        comparison = [judgement(f"c{index}", index != 1) for index in range(10)]
        scored = label(compare(baseline, comparison), "factually_correct")

        self.assertEqual(scored.agreement_rate, 0.8)
        self.assertLess(scored.cohen_kappa, scored.agreement_rate)
        self.assertLess(scored.cohen_kappa, 0.0)

    def test_the_confusion_counts_add_up_to_the_compared_count(self) -> None:
        baseline = [judgement("a", True), judgement("b", True), judgement("c", False)]
        comparison = [judgement("a", True), judgement("b", False), judgement("c", False)]
        scored = label(compare(baseline, comparison), "factually_correct")

        total = (
            scored.both_true
            + scored.both_false
            + scored.only_baseline_true
            + scored.only_comparison_true
        )
        self.assertEqual(total, scored.compared_count)
        self.assertEqual(scored.only_baseline_true, 1)
        self.assertEqual(scored.only_comparison_true, 0)

    def test_errored_and_missing_cases_are_skipped_not_counted_as_agreement(self) -> None:
        baseline = [judgement("a", True), judgement("b", False, error="timeout"), judgement("c", True)]
        comparison = [judgement("a", True), judgement("b", False)]
        report = compare(baseline, comparison)

        self.assertEqual(report.compared_case_ids, ("a",))
        self.assertEqual(report.skipped_case_ids, ("b", "c"))

    def test_every_label_is_scored_independently(self) -> None:
        baseline = [judgement("a", True, faithful=True, tone=True)]
        comparison = [judgement("a", True, faithful=False, tone=True)]
        report = compare(baseline, comparison)

        self.assertEqual(label(report, "factually_correct").agreement_rate, 1.0)
        self.assertEqual(label(report, "faithful_to_facts").agreement_rate, 0.0)
        self.assertEqual(label(report, "tone_appropriate").agreement_rate, 1.0)

    def test_an_empty_comparison_does_not_divide_by_zero(self) -> None:
        report = compare([], [])
        self.assertEqual(report.compared_count, 0)
        self.assertEqual(label(report, "factually_correct").cohen_kappa, 0.0)


class DisagreementTests(unittest.TestCase):
    def test_disagreements_return_the_pairs_worth_reading(self) -> None:
        baseline = [judgement("a", True), judgement("b", True)]
        comparison = [judgement("a", True), judgement("b", False)]
        pairs = disagreements(baseline, comparison)

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0][0].case_id, "b")

    def test_errored_pairs_are_never_reported_as_disagreements(self) -> None:
        baseline = [judgement("a", True, error="timeout")]
        comparison = [judgement("a", False)]
        self.assertEqual(disagreements(baseline, comparison), ())


if __name__ == "__main__":
    unittest.main()
