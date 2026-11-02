"""Test result-to-page identity mapping without importing a GPU framework."""

import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_window_maps_out_of_order_results_to_the_correct_page(tmp_path: Path) -> None:
    iterate = runpy.run_path("backends/paddleocr_runtime/run_inference.py")["iter_page_results"]
    pages = [
        {"page_number": number, "image": {"path": f"page_{number:04d}.png"}} for number in [29, 50]
    ]

    class Predictor:
        def predict(self, inputs):
            assert isinstance(inputs, list)
            for path in reversed(inputs):
                yield SimpleNamespace(json={"res": {"input_path": path}})

    results = list(iterate(Predictor(), pages, tmp_path, 2))
    assert [page["page_number"] for _, page, _ in results] == [50, 29]


@pytest.mark.parametrize("failure", ["missing", "duplicate", "unexpected"])
def test_window_rejects_unreliable_result_identity(tmp_path: Path, failure: str) -> None:
    iterate = runpy.run_path("backends/paddleocr_runtime/run_inference.py")["iter_page_results"]
    pages = [{"page_number": 1, "image": {"path": "page_0001.png"}}]

    class Predictor:
        def predict(self, inputs):
            if failure == "missing":
                return
            filename = "wrong.png" if failure == "unexpected" else inputs
            yield SimpleNamespace(json={"input_path": filename})
            if failure == "duplicate":
                yield SimpleNamespace(json={"input_path": filename})

    with pytest.raises(RuntimeError):
        list(iterate(Predictor(), pages, tmp_path, 1))
