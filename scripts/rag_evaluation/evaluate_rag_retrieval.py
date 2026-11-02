"""Small source-derived retrieval checks; no answer generation or OCR calls."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from multimodal_doc_parser.rag_preparation.gpu_embeddings import GPUEmbeddings
from multimodal_doc_parser.rag_preparation.retrieval import Retriever


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=False)
    cases = [
        dict(
            id="sensor-table",
            document="original",
            query="What are the mean latency and error rate for sensor BY-04?",
            element="element_00007",
        ),
        dict(
            id="cost-comparison-table",
            document="irena",
            query=(
                "Which table compares installed costs, capacity factors and "
                "levelised electricity costs in 2010 and 2024?"
            ),
            element="element_00020",
        ),
        dict(
            id="oil-price-assumptions",
            document="imf",
            query="What oil prices per barrel are assumed for 2025 and 2026?",
            element="element_00009",
        ),
    ]
    manifest = json.loads((args.index / "index_manifest.json").read_text())
    model = GPUEmbeddings(revision=manifest["embedding"]["revision"])
    retriever = Retriever(args.index, model)
    rows = []
    for case in cases:
        # Only the query string reaches the embedding model; expected locations
        # are checked after retrieval, not included in search text.
        document = json.loads(
            Path(
                f"outputs/parser_runs/{case['document']}-final-source-checked-03/document.json"
            ).read_text()
        )
        for mode in ["keyword", "hybrid"]:
            for expanded in [False, True]:
                result = retriever.search(
                    case["query"], mode=mode, expand=expanded, top_k=5, context_tokens=2400
                )
                hit = any(
                    retriever.by_id[m["child_id"]].document_id == document["document_id"]
                    and case["element"] in retriever.by_id[m["child_id"]].source_element_ids
                    for m in result["matched_children"]
                )
                context_hit = any(
                    c["document_id"] == document["document_id"]
                    and case["element"] in c["source_element_ids"]
                    for c in result["contexts"]
                )
                assert result["context_tokens"] <= 2400
                rows.append(
                    dict(
                        case=case["id"],
                        mode=mode,
                        expanded=expanded,
                        expected_document_id=document["document_id"],
                        expected_element_id=case["element"],
                        hit_at_5=hit,
                        context_contains_expected_source=context_hit,
                        result=result,
                    )
                )
                print(case["id"], mode, expanded, hit, context_hit, flush=True)
    report = dict(
        checked_at_utc=datetime.now(UTC).isoformat(),
        scope=(
            "Three easy source-derived development queries; not held-out retrieval accuracy "
            "or evidence that adaptive chunking is best. No fixed-size baseline "
            "or answer evaluation."
        ),
        index_manifest_sha256=hashlib.sha256(
            (args.index / "index_manifest.json").read_bytes()
        ).hexdigest(),
        rows=rows,
    )
    (args.output_directory / "results.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
