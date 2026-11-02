"""Score selected source-derived chart and flowchart assertions, not global accuracy."""

import argparse
import json
import re
from pathlib import Path


def normalized(text: str) -> str:
    return re.sub(r"[^a-z0-9><=.%]", "", text.casefold())


def evaluate(run: Path, reference: dict) -> dict:
    manifest = json.loads((run / "enrichment-manifest.json").read_text())
    if manifest["document_id"] != reference["document_id"]:
        raise ValueError("Reference belongs to a different source PDF")
    analyses = {r["element_id"]: r.get("analysis", {}) for r in manifest["visuals"]}
    checks = []
    for expected in reference.get("visual_types", []):
        actual = analyses.get(expected["element_id"], {}).get("kind")
        checks.append(
            {
                "kind": "visual_type",
                "reference": expected,
                "actual": actual,
                "passed": actual == expected["kind"],
            }
        )
    for expected in reference.get("chart_points", []):
        actual = analyses.get(expected["element_id"], {})
        matches = [
            point
            for series in actual.get("series", [])
            if normalized(series.get("unit") or "") == normalized(expected["unit"])
            for point in series["points"]
            if normalized(point["label"]) == normalized(expected["label"])
        ]
        passed = len(matches) == 1 and matches[0]["value"] == expected["value"]
        passed = passed and matches[0]["value_basis"] == "printed_label"
        checks.append(
            {
                "kind": "printed_chart_point",
                "reference": expected,
                "actual": matches,
                "passed": passed,
            }
        )
    for expected in reference.get("flow_edges", []):
        actual = analyses.get(expected["element_id"], {})
        nodes = {node["id"]: normalized(node["label"]) for node in actual.get("nodes", [])}
        matches = [
            edge
            for edge in actual.get("edges", [])
            if nodes.get(edge["source"]) == normalized(expected["source"])
            and nodes.get(edge["target"]) == normalized(expected["target"])
            and normalized(edge.get("label") or "") == normalized(expected.get("label") or "")
        ]
        checks.append(
            {
                "kind": "flow_connection",
                "reference": expected,
                "actual": matches,
                "passed": len(matches) == 1,
            }
        )
    return {
        "annotation_provenance": reference["annotation_provenance"],
        "scope": "Selected assistant source-derived assertions; no full accuracy score. "
        "Labels ignore case/spacing/punctuation, preserving > < = . %.",
        "passed": sum(c["passed"] for c in checks),
        "checked": len(checks),
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("reference", type=Path)
    args = parser.parse_args()
    result = evaluate(args.run, json.loads(args.reference.read_text()))
    (args.run / "visual-spot-checks.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Selected visual checks: {result['passed']}/{result['checked']}")


if __name__ == "__main__":
    main()
