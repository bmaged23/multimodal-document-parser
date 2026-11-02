"""Run the complete PaddleOCR pipeline in its isolated inference environment."""

import argparse
import json
import os
import subprocess
import threading
import time
from importlib.metadata import version
from pathlib import Path

# Keep downloaded models inside the project by default, including under CI/manual runs.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(PROJECT_ROOT / ".cache" / "paddlex"))
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))


def iter_page_results(pipeline, pages, bundle, page_window=1):
    """Use one predictor with bounded input windows; verify each result's source identity."""
    if page_window not in (1, 2):
        raise ValueError("Page window must be 1 or 2")
    for offset in range(0, len(pages), page_window):
        window = pages[offset : offset + page_window]
        paths = [str((bundle / page["image"]["path"]).resolve()) for page in window]
        lookup = {Path(path).name: page for path, page in zip(paths, window, strict=True)}
        seen = set()
        started = time.perf_counter()
        inputs = paths[0] if page_window == 1 else paths
        for result in pipeline.predict(inputs):
            saved = result.json
            raw = saved.get("res", saved)
            filename = Path(raw["input_path"]).name
            if filename not in lookup or filename in seen:
                raise RuntimeError(f"Unexpected or duplicate result input: {filename}")
            seen.add(filename)
            yield result, lookup[filename], time.perf_counter() - started
        if seen != set(lookup):
            raise RuntimeError("Missing results in page window")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("raw_directory", type=Path)
    parser.add_argument("--device", default="gpu:0")
    parser.add_argument("--page-window", type=int, choices=[1, 2], default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    document = json.loads((args.bundle / "document.json").read_text(encoding="utf-8"))
    args.raw_directory.mkdir(parents=True, exist_ok=args.resume)
    started = time.perf_counter()
    memory_samples: list[int] = []
    stop = threading.Event()

    def sample_memory() -> None:
        while not stop.is_set():
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=3,
                    check=True,
                )
                memory_samples.append(int(result.stdout.splitlines()[0]))
            except (OSError, ValueError, subprocess.SubprocessError, IndexError):
                pass
            stop.wait(1)

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    timings = []
    previous_elapsed = 0.0
    previous_peak_memory = 0
    previous_initialization = 0.0
    if args.resume:
        previous_path = args.raw_directory / "inference-manifest.json"
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        expected = {
            "document_id": document["document_id"],
            "device": args.device,
            "paddleocr_version": version("paddleocr"),
            "paddlepaddle_version": version("paddlepaddle-gpu"),
        }
        if any(previous.get(key) != value for key, value in expected.items()):
            raise ValueError("Resume checkpoint does not match input, runtime, or device")
        if previous.get("page_window", 1) != args.page_window:
            raise ValueError("Resume page window differs from checkpoint")
        timings = previous["pages"]
        for timing in timings:
            path = args.raw_directory / f"page_{timing['page_number']:04d}.json"
            raw = json.loads(path.read_text(encoding="utf-8"))
            data = raw.get("res", raw)
            page = document["pages"][timing["page_number"] - 1]
            if (
                data.get("width") != page["width_pixels"]
                or data.get("height") != page["height_pixels"]
            ):
                raise ValueError("Saved page dimensions do not match preparation")
        previous_elapsed = previous["total_seconds"]
        previous_peak_memory = previous.get("peak_sampled_device_memory_mib") or 0
        previous_initialization = previous.get("initialization_seconds") or 0.0
        history = args.raw_directory / f"inference-manifest.before-resume-{time.time_ns()}.json"
        history.write_text(json.dumps(previous, indent=2) + "\n", encoding="utf-8")
    initialization_seconds = previous_initialization
    manifest_path = args.raw_directory / "inference-manifest.json"

    def checkpoint(state, error=None):
        manifest = {
            "status": state,
            "document_id": document["document_id"],
            "backend": "PaddleOCR-VL-1.6",
            "pipeline_version": "v1.6",
            "paddleocr_version": version("paddleocr"),
            "paddlepaddle_version": version("paddlepaddle-gpu"),
            "device": args.device,
            "page_window": args.page_window,
            "gpu_model_instances": 1,
            "initialization_seconds": initialization_seconds,
            "total_seconds": previous_elapsed + time.perf_counter() - started,
            "current_attempt_seconds": time.perf_counter() - started,
            "pages": timings,
            "completed_pages": len(timings),
            "total_pages": len(document["pages"]),
            "peak_sampled_device_memory_mib": max([previous_peak_memory, *memory_samples]) or None,
            "memory_measurement": (
                "GPU 0 total device memory sampled once per second; not allocator peak"
            ),
            "orientation_classification": False,
            "document_unwarping": False,
            "error": error,
        }
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        temporary.replace(manifest_path)

    checkpoint("initializing")
    try:
        from paddleocr import PaddleOCRVL

        pipeline = PaddleOCRVL(
            pipeline_version="v1.6",
            device=args.device,
            precision="fp16" if args.device.startswith("gpu") else "fp32",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )
        initialization_seconds = previous_initialization + time.perf_counter() - started
        checkpoint("running")
        completed = {timing["page_number"] for timing in timings}
        pending_pages = [page for page in document["pages"] if page["page_number"] not in completed]
        print(
            f"Resuming with {len(completed)} completed pages; {len(pending_pages)} pending",
            flush=True,
        )
        for result, page, completion_offset in iter_page_results(
            pipeline, pending_pages, args.bundle, args.page_window
        ):
            result.save_to_json(str(args.raw_directory / f"page_{page['page_number']:04d}.json"))
            timings.append(
                {
                    "page_number": page["page_number"],
                    "elapsed_seconds": completion_offset if args.page_window == 1 else None,
                    "completion_seconds_since_window_start": completion_offset,
                }
            )
            checkpoint("running")
            print(
                f"Completed page {page['page_number']}/{len(document['pages'])} "
                f"(window completion offset {completion_offset:.1f}s)",
                flush=True,
            )
    except BaseException as exc:
        checkpoint("failed", f"{type(exc).__name__}: {exc}")
        raise
    finally:
        stop.set()
        sampler.join(timeout=4)
    checkpoint("completed")


if __name__ == "__main__":
    main()
