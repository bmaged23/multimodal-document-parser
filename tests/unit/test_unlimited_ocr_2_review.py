"""Untrusted predicted image coordinates must never be evaluated as Python."""

import importlib.util
from pathlib import Path

from PIL import Image


def review_module():
    spec = importlib.util.spec_from_file_location(
        "unlimited_review", "scripts/visualization/build_unlimited_ocr_2_review.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_predicted_crop_uses_source_pixels(tmp_path: Path):
    source = tmp_path / "page.png"
    Image.new("RGB", (999, 999), "red").save(source)
    (tmp_path / "assets").mkdir()
    text = "<PAGE><|ref|>image<|/ref|><|det|>[[10, 20, 110, 220]]<|/det|>"
    markdown = review_module().visual_markdown(text, source, tmp_path, 1)
    crop = tmp_path / "assets/page_0001_crop_001.png"
    with Image.open(crop) as image:
        assert image.size == (100, 200)
    assert "assets/page_0001_crop_001.png" in markdown
    assert "<PAGE>" not in markdown


def test_invalid_coordinates_do_not_execute_or_create_crops(tmp_path: Path):
    source = tmp_path / "page.png"
    Image.new("RGB", (100, 100)).save(source)
    (tmp_path / "assets").mkdir()
    module = review_module()
    for payload in [
        "__import__('os').system('false')",
        "[[-1, 0, 100, 100]]",
        "[[100, 0, 0, 100]]",
        "[[0, 0, 1000, 100]]",
    ]:
        markdown = module.visual_markdown(
            f"<|ref|>image<|/ref|><|det|>{payload}<|/det|>", source, tmp_path, 1
        )
        assert "coordinates invalid" in markdown
    assert list((tmp_path / "assets").iterdir()) == []


def test_block_format_retains_table_and_builds_visual_crops(tmp_path: Path):
    source = tmp_path / "page.png"
    Image.new("RGB", (1000, 1000), "white").save(source)
    (tmp_path / "assets").mkdir()
    text = (
        "<PAGE><BLOCK>title[0, 0, 500, 30]<FORK>Section"
        "<BLOCK>table[0, 30, 500, 100]<FORK><table><tr><td>18.4</td></tr></table>"
        "<BLOCK>chart[100, 200, 400, 600]<FORK>"
        "<BLOCK>text[0, 700, 400, 800]<FORK>Caption"
    )
    markdown = review_module().visual_markdown(text, source, tmp_path, 1)
    assert "## Section" in markdown
    assert "<td>18.4</td>" in markdown
    assert "<BLOCK>" not in markdown and "<FORK>" not in markdown
    assert "Caption" in markdown
    with Image.open(tmp_path / "assets/page_0001_crop_001.png") as crop:
        assert crop.size == (300, 400)
