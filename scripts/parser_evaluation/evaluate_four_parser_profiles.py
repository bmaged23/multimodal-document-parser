"""Re-score four saved parser profiles using the unchanged handwriting reference."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    source = ROOT / "scripts/parser_evaluation/evaluate_handwritten_math_comparison.py"
    spec = importlib.util.spec_from_file_location("frozen_handwriting", source)
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    scorer.RUNS["dots.mocr original HF BF16 · Colab T4 · SDPA"] = (
        "dots-mocr-bf16-colab-author-mathematics-4-pages-01"
    )
    report = scorer.evaluate(scorer.REFERENCE)
    report["gpu"] = "Baseline profiles: GTX 1660 Ti 6 GiB; original HF BF16: Colab Tesla T4"
    report["limitations"].append(
        "Different hardware, backends and conversion lineage; not a causal quantization test."
    )
    target = ROOT / "evaluation/results/author-handwritten-mathematics-four-profiles-v1.json"
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    manifest = json.loads((ROOT / "evaluation/showcase-cases.json").read_text())
    precision = json.loads(
        (ROOT / "evaluation/dots-mocr-precision-showcase-cases.json").read_text()
    )
    mapping = {"nasa-financial": 1, "handwriting": 3, "original-private": 0}
    for case in manifest["cases"]:
        if case["id"] in mapping:
            addition = precision["cases"][mapping[case["id"]]]["runs"][1].copy()
            case["runs"].append(addition)
    diary = json.loads(
        (
            ROOT / "evaluation/showcase_history/showcase-cases-with-historical-diary-v1.json"
        ).read_text()
    )["cases"][1]
    diary["id"] = "historical-diary"
    diary["runs"].append(precision["cases"][2]["runs"][1])
    manifest["cases"].append(diary)
    operational = json.loads(
        (ROOT / "evaluation/results/parser-comparison-showcase-v1.json").read_text()
    )
    cloud = json.loads(
        (ROOT / "evaluation/results/dots-mocr-colab-bf16-comparison-v1.json").read_text()
    )
    nasa = next(c for c in cloud["cases"] if c["id"] == "nasa-first-20-pages")
    operational["parsers"].append(
        {
            "model_identity": ["dots.mocr original HF BF16 · Colab T4 · SDPA"],
            "accepted_pages": nasa["bf16_accepted_pages"],
            "selected_comparison_pages": 20,
            "median_page_seconds": nasa["bf16_accepted_page_median_seconds"],
            "quality_reference_status": "not_scored",
            "gpu": cloud["cloud_runtime"]["gpu"],
        }
    )
    operational["limitations"] = str(operational.get("limitations", "")) + (
        " BF16 uses a Colab T4; other profiles use the local GTX 1660 Ti. "
        "Different backends and hardware prevent causal quantization or speedup claims."
    )
    op_path = ROOT / "evaluation/results/parser-four-profiles-operational-v1.json"
    op_path.write_text(json.dumps(operational, indent=2, ensure_ascii=False) + "\n")
    manifest["parser_evaluation"] = op_path.name
    manifest["math_evaluation"] = target.name
    (ROOT / "evaluation/four-parser-profiles-showcase-cases.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    for row in report["results"]:
        print(row["model"], row["selected_assertions"], row["truth_table"]["passed"])


if __name__ == "__main__":
    main()
