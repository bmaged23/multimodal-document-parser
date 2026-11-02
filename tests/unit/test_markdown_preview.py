"""Check preview fidelity and sanitization of model-generated Markdown."""

from multimodal_doc_parser.review_interface.html_report import render_markdown_preview


def test_preview_renders_images_tables_and_code() -> None:
    result = render_markdown_preview(
        "# Results\n\n![chart](assets/crops/chart.png)\n\n"
        '<table><tr><td colspan="2">18.4</td></tr></table>\n\n'
        "```python\nx = 2\n```"
    )
    assert "<h1>Results</h1>" in result
    assert 'src="assets/crops/chart.png"' in result
    assert 'loading="lazy"' in result
    assert 'colspan="2"' in result
    assert "<pre><code" in result


def test_preview_rejects_active_html_and_dangerous_links() -> None:
    result = render_markdown_preview(
        "<script>window.injected=true</script>\n\n"
        '<img src="assets/chart.png" onerror="alert(1)">\n\n'
        '<a href="javascript:alert(1)">click</a>'
    )
    assert "<script" not in result
    assert "onerror" not in result
    assert "javascript:" not in result
    assert 'src="assets/chart.png"' in result
