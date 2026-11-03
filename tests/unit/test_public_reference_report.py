"""The published report links only allowlisted pages and exposes no private artifacts."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location(
    "public_reference_report", ROOT / "scripts/visualization/build_manual_reference_report.py"
)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def test_public_report_maps_pages_and_removes_private_links(tmp_path):
    metrics = json.loads(
        (ROOT / "evaluation/results/manual-reference-full-corpus-v1.json").read_text()
    )
    (tmp_path / "metrics.json").write_text(json.dumps(metrics))
    (tmp_path / "excluded-references.json").write_text("[]")
    (tmp_path / "page-metrics.json").write_text(
        json.dumps(
            [
                dict(case="irena", page=3, profile="bf16"),
                dict(case="irena", page=1, profile="bf16"),
                dict(case="imf", page=14, profile="dots"),
            ]
        )
    )
    demo = tmp_path / "demo"
    demo.mkdir()
    (demo / "catalog.json").write_text(
        json.dumps(
            dict(
                cases=[
                    dict(id="irena-pipeline", original_pages=[15]),
                    dict(id="imf-pipeline", original_pages=[14]),
                ]
            )
        )
    )
    report.build(tmp_path, demo_directory=demo)
    html = (tmp_path / "index.html").read_text()
    assert "localhost" not in html
    assert 'href="failure-traces.json"' not in html
    assert "frozen-reference.json" not in html
    assert 'href="../"' in html
    payload = json.loads(html.split('id="data">')[1].split("</script>")[0])
    assert payload["pages"][0]["source_url"] == (
        "../?case=irena-pipeline&page=15&run=irena-bf16-gemma"
    )
    assert payload["pages"][1]["source_url"] is None
    assert payload["pages"][2]["source_url"] == ("../?case=imf-pipeline&page=14&run=imf-dots-gemma")
    assert payload["public"] is True
