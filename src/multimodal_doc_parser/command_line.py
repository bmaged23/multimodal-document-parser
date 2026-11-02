"""Command-line interface for PDF preparation and schema export."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from multimodal_doc_parser.document_schema.document import Document
from multimodal_doc_parser.evaluation.parser_comparison import compare_parser_runs
from multimodal_doc_parser.parser_adapters.parsing_run import (
    ParsingRunError,
    parse_prepared_document,
)
from multimodal_doc_parser.pdf_preparation.prepare_document import (
    DocumentPreparationError,
    prepare_document,
)
from multimodal_doc_parser.visual_enrichment.ollama import enrich_document


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare traceable PDF document assets")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Render a PDF and create document.json")
    prepare.add_argument("pdf", type=Path)
    prepare.add_argument("--output-directory", type=Path, default=Path("outputs/parsed_documents"))
    prepare.add_argument("--dpi", type=int, default=144)
    prepare.add_argument("--max-pages", type=int, default=200)
    prepare.add_argument("--max-page-pixels", type=int, default=25_000_000)
    parse = commands.add_parser("parse", help="Parse a prepared bundle with a selected backend")
    parse.add_argument("bundle", type=Path)
    parse.add_argument("--output-directory", type=Path)
    parse.add_argument("--runtime-directory", type=Path, default=Path("backends/paddleocr_runtime"))
    parse.add_argument(
        "--page-window",
        type=int,
        choices=[1, 2],
        default=1,
        help="Pages in flight through one shared GPU model",
    )
    parse.add_argument("--device", choices=["gpu:0", "cpu"], default="gpu:0")
    parse.add_argument(
        "--raw-results-directory",
        type=Path,
        help="Import saved page_XXXX.json results instead of running inference",
    )
    parse.add_argument("--backend", choices=["paddleocr", "dots-mocr"], default="paddleocr")
    parse.add_argument("--base-url", default="http://127.0.0.1:8010/v1")
    parse.add_argument("--model", default="dots.mocr-Q4_K_M")
    parse.add_argument("--model-version", help="Exact revision and quantization of dots.mocr")
    parse.add_argument("--server-kind", choices=["llama-cpp", "vllm"], default="llama-cpp")
    parse.add_argument("--timeout-seconds", type=float, default=300)
    parse.add_argument("--max-tokens", type=int, default=8192)
    parse.add_argument("--max-pixels", type=int, default=1_000_000)
    parse.add_argument("--runtime-metadata", type=Path)
    compare = commands.add_parser(
        "compare", help="Compare parser runs for identical prepared inputs"
    )
    compare.add_argument("left_run", type=Path)
    compare.add_argument("right_run", type=Path)
    compare.add_argument("--output-directory", type=Path, required=True)
    compare.add_argument("--pages", type=int, nargs="+")
    compare.add_argument(
        "--reference", type=Path, help="Manually verified reference transcription JSON"
    )
    enrich = commands.add_parser(
        "enrich", help="Describe source visuals with Ollama Cloud by default"
    )
    enrich.add_argument("source_run", type=Path)
    enrich.add_argument("--output-directory", type=Path, required=True)
    enrich.add_argument("--base-url", default="http://127.0.0.1:11435")
    enrich.add_argument("--model", default="gemma4:31b")
    enrich.add_argument(
        "--provider", choices=["ollama", "qwen-cloud", "ollama-cloud"], default="ollama-cloud"
    )
    enrich.add_argument("--timeout-seconds", type=float, default=300)
    commands.add_parser("schema", help="Print the versioned document JSON Schema")
    chunk = commands.add_parser("chunk", help="Build adaptive source-linked parents and children")
    chunk.add_argument("source_run", type=Path)
    chunk.add_argument("--output-directory", type=Path, required=True)
    chunk.add_argument("--child-target-tokens", type=int, default=350)
    chunk.add_argument("--child-max-tokens", type=int, default=500)
    index = commands.add_parser("index", help="Build BM25 and 1024-dimensional GPU BGE index")
    index.add_argument("bundles", type=Path, nargs="+")
    index.add_argument("--output-directory", type=Path, required=True)
    index.add_argument("--batch-size", type=int, default=2)
    retrieve = commands.add_parser(
        "retrieve", help="Retrieve evidence with bounded parent expansion"
    )
    retrieve.add_argument("index", type=Path)
    retrieve.add_argument("--query", required=True)
    retrieve.add_argument("--mode", choices=["hybrid", "keyword", "dense"], default="hybrid")
    retrieve.add_argument("--top-k", type=int, default=5)
    retrieve.add_argument("--context-tokens", type=int, default=2400)
    retrieve.add_argument("--no-parent-expansion", action="store_true")
    args = parser.parse_args()
    if args.command in {"chunk", "index", "retrieve"}:
        from multimodal_doc_parser.rag_preparation.chunk_bundle import prepare_chunks
        from multimodal_doc_parser.rag_preparation.chunk_schema import ChunkSettings
        from multimodal_doc_parser.rag_preparation.gpu_embeddings import GPUEmbeddings
        from multimodal_doc_parser.rag_preparation.retrieval import Retriever, build_index

        try:
            if args.command == "chunk":
                result = prepare_chunks(
                    args.source_run,
                    args.output_directory,
                    ChunkSettings(
                        child_target_tokens=args.child_target_tokens,
                        child_max_tokens=args.child_max_tokens,
                    ),
                )
                print(result / "review.html")
            elif args.command == "index":
                print(
                    build_index(
                        args.bundles,
                        args.output_directory,
                        embedder=GPUEmbeddings(),
                        batch_size=args.batch_size,
                    )
                )
            else:
                print(
                    json.dumps(
                        Retriever(args.index).search(
                            args.query,
                            top_k=args.top_k,
                            context_tokens=args.context_tokens,
                            expand=not args.no_parent_expansion,
                            mode=args.mode,
                        ),
                        indent=2,
                    )
                )
        except (OSError, ValueError) as exc:
            parser.exit(1, f"RAG operation failed: {exc}\n")
        return
    if args.command == "enrich":
        try:
            result = enrich_document(
                args.source_run,
                args.output_directory,
                provider=args.provider,
                base_url=args.base_url,
                model=args.model,
                timeout=args.timeout_seconds,
            )
        except (ValueError, OSError) as exc:
            parser.exit(1, f"Enrichment failed: {exc}\n")
        print(result / "review.html")
        return
    if args.command == "compare":
        try:
            result = compare_parser_runs(
                args.left_run,
                args.right_run,
                args.output_directory,
                pages=args.pages,
                reference_path=args.reference,
            )
        except (ValueError, OSError) as exc:
            parser.exit(1, f"Comparison failed: {exc}\n")
        print(result / "comparison.html")
        return
    if args.command == "schema":
        print(json.dumps(Document.model_json_schema(), indent=2))
        return
    if args.command == "parse":
        destination = args.output_directory or Path("outputs/parser_runs") / (
            args.backend + "-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        )
        try:
            result = parse_prepared_document(
                args.bundle,
                destination,
                args.runtime_directory,
                device=args.device,
                page_window=args.page_window,
                raw_results_directory=args.raw_results_directory,
                backend=args.backend,
                base_url=args.base_url,
                model=args.model,
                model_version=args.model_version,
                server_kind=args.server_kind,
                timeout_seconds=args.timeout_seconds,
                max_tokens=args.max_tokens,
                max_pixels=args.max_pixels,
                runtime_metadata_path=args.runtime_metadata,
            )
        except (ParsingRunError, ValueError, OSError) as exc:
            parser.exit(1, f"Parsing failed: {exc}\n")
        print(result / "review.html")
        return
    try:
        result = prepare_document(
            args.pdf,
            args.output_directory,
            dpi=args.dpi,
            max_pages=args.max_pages,
            max_page_pixels=args.max_page_pixels,
        )
    except (DocumentPreparationError, OSError) as exc:
        parser.exit(1, f"Preparation failed: {exc}\n")
    print(result)


if __name__ == "__main__":
    main()
