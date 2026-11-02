"""Protect source fidelity and row/column alignment in the selected math checks."""

import importlib.util
import json
from pathlib import Path


def evaluator():
    spec = importlib.util.spec_from_file_location(
        "math_evaluation", "scripts/parser_evaluation/evaluate_handwritten_math_comparison.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_math_formatting_preserves_negation_and_the_written_false_statement():
    module = evaluator()
    assert module.normalize(r"$\neg P \land \neg Q$") == module.normalize("¬P∧¬Q")
    assert module.normalize(r"$2x^{2}+1$") == module.normalize("2x^2+1")
    assert module.normalize("¬P∧¬Q") != module.normalize("P∧Q")
    assert module.normalize("2+2=5") != module.normalize("2+2=4")
    assert module.normalize(r"M=\{x\in\mathbb{N}|1\leq x\leq3\}") == module.normalize(
        "M={x∈N|1≤x≤3}"
    )
    assert module.normalize("A={x∈N}") not in module.normalize("A={x∈N:A={x∈N:A=")


def test_truth_table_requires_ordered_headers_and_cells():
    module = evaluator()
    reference = json.loads(
        Path(
            "evaluation/reference_annotations/author-handwritten-mathematics-selected-v1.json"
        ).read_text()
    )["truth_table"]
    rows = [reference["headers"], *reference["rows"]]

    def html(items):
        return (
            "<table>"
            + "".join(
                "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in items
            )
            + "</table>"
        )

    assert module.table_score(html(rows), reference)["passed"] == 20
    wrong = [rows[0], rows[2], rows[1], *rows[3:]]
    assert module.table_score(html(wrong), reference)["passed"] == 16
    wrong = [["Q", "P", *rows[0][2:]], *rows[1:]]
    assert module.table_score(html(wrong), reference)["passed"] == 0
    assert module.table_score("T T F F F " * 4, reference)["passed"] == 0


def test_experimental_layout_coordinates_do_not_interrupt_or_supply_transcription(tmp_path):
    module = evaluator()
    (tmp_path / "page_0001.json").write_text(
        json.dumps(
            {"text": "<PAGE><BLOCK>text[1,2,3,4]<FORK>The sun is<BLOCK>text[5,6,7,8]<FORK>shining"}
        )
    )
    text, _ = module.page_text(tmp_path, 1, "Unlimited-OCR 2")
    assert module.normalize(text) == module.normalize("The sun is shining")
