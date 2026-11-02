import pytest

from multimodal_doc_parser.parser_adapters.dots_mocr_adapter import decode_layout
from multimodal_doc_parser.parser_adapters.dots_mocr_inference import input_size


def test_layout_decoder_rejects_malformed_and_nonfinite_coordinates():
    with pytest.raises(ValueError):
        decode_layout('[{"category":"Text","bbox":[0,0,NaN,4],"text":"x"}]')
    with pytest.raises(ValueError):
        decode_layout('[{"category":"Text","bbox":[0,0,3,4],"extra":"x"}]')
    with pytest.raises(ValueError):
        decode_layout("partial [{")
    assert decode_layout("```json\n[]\n```") == []


@pytest.mark.parametrize("width,height,budget", [(1190, 1684, 2_100_000), (1413, 1244, 1_000_000)])
def test_input_resizing_respects_visual_grid_and_pixel_budget(width, height, budget):
    w, h = input_size(width, height, budget)
    assert w % 28 == h % 28 == 0
    assert w * h <= budget
    assert abs(w / h - width / height) < 0.05
