"""Measure how far two sets of verdicts agree about the same answers.

An LLM judge is only worth its cost if it agrees with a human on the cases a
human bothered to label.  This compares two judgement files label by label --
model against model, or model against your own hand labels -- and reports both
raw agreement and Cohen's kappa.

Usage:
    .venv/bin/python -m evaluation.agreement results/judgements.jsonl labels/mine.jsonl
"""

import argparse
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from evaluation.judge import LABELS, Judgement, load_judgements


@dataclass(frozen=True)
class LabelAgreement:
    """How often two judges gave the same verdict for one boolean label."""

    label: str
    compared_count: int
    agreement_rate: float
    both_true: int
    both_false: int
    only_baseline_true: int
    only_comparison_true: int
    cohen_kappa: float


@dataclass(frozen=True)
class AgreementReport:
    """Per-label agreement over the cases both sides actually judged."""

    compared_case_ids: tuple[str, ...]
    skipped_case_ids: tuple[str, ...]
    labels: tuple[LabelAgreement, ...]

    @property
    def compared_count(self) -> int:
        return len(self.compared_case_ids)


def compare(
    baseline: Sequence[Judgement],
    comparison: Sequence[Judgement],
    *,
    labels: Sequence[str] = LABELS,
) -> AgreementReport:
    """Compare two judges case by case, ignoring anything either did not score.

    A case is skipped when either side is missing it or recorded an error. An
    errored judgement carries placeholder falses rather than an opinion, so
    counting it would credit the judges with agreeing about nothing.
    """
    baseline_by_id = {judgement.case_id: judgement for judgement in baseline}
    comparison_by_id = {judgement.case_id: judgement for judgement in comparison}

    compared: list[str] = []
    skipped: list[str] = []
    for case_id in sorted(baseline_by_id.keys() | comparison_by_id.keys()):
        left = baseline_by_id.get(case_id)
        right = comparison_by_id.get(case_id)
        if left is None or right is None or left.error is not None or right.error is not None:
            skipped.append(case_id)
        else:
            compared.append(case_id)

    return AgreementReport(
        compared_case_ids=tuple(compared),
        skipped_case_ids=tuple(skipped),
        labels=tuple(
            _score_label(
                label,
                [getattr(baseline_by_id[case_id], label) for case_id in compared],
                [getattr(comparison_by_id[case_id], label) for case_id in compared],
            )
            for label in labels
        ),
    )


def disagreements(
    baseline: Sequence[Judgement],
    comparison: Sequence[Judgement],
    *,
    label: str = "factually_correct",
) -> tuple[tuple[Judgement, Judgement], ...]:
    """Return the judgement pairs that differ on one label, for manual reading.

    Reading the disagreements is how you find out whether the judge is wrong or
    your own labels were.
    """
    comparison_by_id = {judgement.case_id: judgement for judgement in comparison}
    return tuple(
        (left, comparison_by_id[left.case_id])
        for left in baseline
        if left.error is None
        and left.case_id in comparison_by_id
        and comparison_by_id[left.case_id].error is None
        and getattr(left, label) != getattr(comparison_by_id[left.case_id], label)
    )


def _score_label(
    label: str, baseline_values: Sequence[bool], comparison_values: Sequence[bool]
) -> LabelAgreement:
    pairs = list(zip(baseline_values, comparison_values))
    compared_count = len(pairs)
    both_true = sum(1 for left, right in pairs if left and right)
    both_false = sum(1 for left, right in pairs if not left and not right)

    return LabelAgreement(
        label=label,
        compared_count=compared_count,
        agreement_rate=(both_true + both_false) / compared_count if compared_count else 0.0,
        both_true=both_true,
        both_false=both_false,
        only_baseline_true=sum(1 for left, right in pairs if left and not right),
        only_comparison_true=sum(1 for left, right in pairs if not left and right),
        cohen_kappa=_cohen_kappa(baseline_values, comparison_values),
    )


def _cohen_kappa(baseline_values: Sequence[bool], comparison_values: Sequence[bool]) -> float:
    """Agreement discounted by the agreement two judges would hit by luck.

    Raw agreement flatters a judge whenever one verdict dominates. If 95 of 100
    answers are correct, two judges that always answer true agree 95% of the
    time while sharing no judgement at all. Kappa subtracts that expected
    coincidence: 1.0 is perfect, 0.0 is no better than chance, below 0.0 is
    worse than chance.
    """
    total = len(baseline_values)
    if total == 0:
        return 0.0

    observed = sum(1 for left, right in zip(baseline_values, comparison_values) if left == right) / total
    baseline_true = sum(baseline_values) / total
    comparison_true = sum(comparison_values) / total
    expected = baseline_true * comparison_true + (1 - baseline_true) * (1 - comparison_true)

    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else 0.0
    return (observed - expected) / (1 - expected)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two sets of judgements.")
    parser.add_argument("baseline", type=Path, help="Judgement file, e.g. the LLM judge.")
    parser.add_argument("comparison", type=Path, help="Judgement file, e.g. your hand labels.")
    parser.add_argument(
        "--show-disagreements",
        metavar="LABEL",
        help="Print the case IDs where the two sides differ on this label.",
    )
    arguments = parser.parse_args()

    baseline = load_judgements(arguments.baseline)
    comparison = load_judgements(arguments.comparison)
    report = compare(baseline, comparison)

    print(f"compared {report.compared_count} cases, skipped {len(report.skipped_case_ids)}")
    print(f"{'label':<20}{'agree':>8}{'kappa':>8}{'n':>6}")
    for label in report.labels:
        print(
            f"{label.label:<20}{label.agreement_rate:>8.3f}"
            f"{label.cohen_kappa:>8.3f}{label.compared_count:>6}"
        )

    if arguments.show_disagreements:
        pairs = disagreements(baseline, comparison, label=arguments.show_disagreements)
        print(f"\n{len(pairs)} disagreements on {arguments.show_disagreements}:")
        for left, right in pairs:
            print(f"  {left.case_id}: baseline={getattr(left, arguments.show_disagreements)} "
                  f"comparison={getattr(right, arguments.show_disagreements)} -- {left.reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
