"""Check metric arithmetic against simple independent reference implementations."""

import importlib.util
import itertools
import random
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "manual_reference_metrics",
    Path(__file__).resolve().parents[2]
    / "scripts/parser_evaluation/evaluate_manual_reference_corpus.py",
)
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


def dynamic_distance(a, b, substring=False):
    row = [0] * (len(b) + 1) if substring else list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        new = [i]
        for j, y in enumerate(b, 1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j - 1] + (x != y)))
        row = new
    return min(row) if substring else row[-1]


def test_exact_and_region_distances_against_dynamic_programming():
    rng = random.Random(17)
    for _ in range(100):
        a = "".join(rng.choices("abc", k=rng.randrange(1, 9)))
        b = "".join(rng.choices("abc", k=rng.randrange(1, 15)))
        for substring in [False, True]:
            assert metrics.edits(a, b, substring) == dynamic_distance(a, b, substring)
    assert metrics.edits("2+2=5", "prefix 2+2=5 suffix", True) == 0
    assert metrics.edits("2+2=5", "2+2=4", True) == 1
    assert metrics.edits("abc", "", True) == 3


def test_assignment_matches_exhaustive_optimum_with_missing_predictions():
    costs = [[-0.8, -0.7], [-0.75, -0.1], [-0.2, -0.3]]
    rows, cols = metrics.minimum_assignment(costs)
    actual = sum(costs[r][c] for r, c in zip(rows, cols, strict=True))
    expected = min(
        sum(costs[r][c] for c, r in enumerate(order))
        for order in itertools.permutations(range(3), 2)
    )
    assert abs(actual - expected) < 1e-9
    assert len(set(cols)) == len(cols) == 2


def test_text_normalization_preserves_inequalities_symbols_and_numbers():
    assert metrics.normalized("<p>−0.034 &amp; 0.043</p>") == "-0.034 & 0.043"
    assert metrics.normalized("1 < x < 3") == "1 < x < 3"
    assert metrics.normalized("•") != metrics.normalized("*")
    assert metrics.normalized("2+2=5") != metrics.normalized("2+2=4")


def test_count_matching_does_not_credit_repetitions_more_than_once():
    assert metrics.count_match(["A", "B"], ["A", "A", "A"]) == dict(
        reference=2, prediction=3, matched=1
    )


def test_equivalent_bullets_ellipses_and_embedded_objective_keys():
    from multimodal_doc_parser.document_schema.document import TableCell

    assert metrics.normalized("●") == metrics.normalized("•")
    assert metrics.normalized(". . .") == metrics.normalized("...")
    assert metrics.normalized("●") != metrics.normalized("*")
    row = [
        TableCell(row=1, column=0, text="1.1 Understand Earth."),
        TableCell(row=1, column=1, text="1,974"),
        TableCell(row=1, column=2, text="1,740"),
    ]
    assert metrics.keyed_tail(row, "1.1") == [["Understand Earth.", "1,974", "1,740"]]
    assert metrics.keyed_tail(row, "1.2") == []
