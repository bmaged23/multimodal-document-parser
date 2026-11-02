"""Small shadow comparison of serial page calls and shared-model queued windows."""

import argparse
import json
import statistics
import subprocess
import threading
import time
from importlib.metadata import version
from pathlib import Path

# Reuse the production cache locations, without starting inference at import time.
import run_inference  # noqa: F401


def content_signature(raw):
    result = raw.get("res", raw)
    return [
        [block["block_label"], " ".join(str(block.get("block_content") or "").split())]
        for block in result["parsing_res_list"]
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--pages", type=int, nargs="+", default=[29, 50])
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()
    document = json.loads((args.bundle / "document.json").read_text(encoding="utf-8"))
    if args.rounds < 1 or len(set(args.pages)) != len(args.pages):
        parser.error("Use positive rounds and unique page numbers")
    if any(page < 1 or page > len(document["pages"]) for page in args.pages):
        parser.error("Page outside the prepared document")
    args.output_directory.mkdir(parents=True, exist_ok=False)
    summary_path = args.output_directory / "shadow-results.json"
    started = time.perf_counter()
    stop = threading.Event()
    memory_samples = []
    utilization_samples = []
    trials = []
    report = {
        "status": "initializing",
        "document_id": document["document_id"],
        "sample_pages": args.pages,
        "rounds": args.rounds,
        "gpu_model_instances": 1,
        "device": "gpu:0",
        "paddleocr_version": version("paddleocr"),
        "paddlepaddle_version": version("paddlepaddle-gpu"),
        "comparison": "one-page calls versus queued two-page windows; not GPU model replicas",
        "trials": trials,
    }

    def checkpoint():
        temporary = summary_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(summary_path)

    def sample_memory():
        while not stop.is_set():
            try:
                output = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=3,
                ).stdout.splitlines()[0]
                memory, utilization = [int(item.strip()) for item in output.split(",")]
                memory_samples.append(memory)
                utilization_samples.append(utilization)
            except (OSError, ValueError, IndexError, subprocess.SubprocessError):
                pass
            stop.wait(1)

    checkpoint()
    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    try:
        from paddleocr import PaddleOCRVL

        pipeline = PaddleOCRVL(
            pipeline_version="v1.6",
            device="gpu:0",
            precision="fp16",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_queues=True,
        )
        report["initialization_seconds"] = time.perf_counter() - started
        report["status"] = "warming_up"
        checkpoint()
        warmup_started = time.perf_counter()
        for _ in pipeline.predict(str(args.bundle / document["pages"][0]["image"]["path"])):
            pass
        report["warmup_seconds"] = time.perf_counter() - warmup_started
        print(f"Warm-up finished in {report['warmup_seconds']:.1f}s", flush=True)
        report["status"] = "running"
        checkpoint()
        for round_index in range(args.rounds):
            # Reverse order on alternating rounds to reduce simple order/thermal bias.
            windows = [1, 2] if round_index % 2 == 0 else [2, 1]
            for window in windows:
                directory = args.output_directory / f"round_{round_index + 1}" / f"window_{window}"
                directory.mkdir(parents=True)
                memory_samples.clear()
                utilization_samples.clear()
                trial_started = time.perf_counter()
                completed = []
                for offset in range(0, len(args.pages), window):
                    numbers = args.pages[offset : offset + window]
                    images = [
                        str(
                            (args.bundle / document["pages"][number - 1]["image"]["path"]).resolve()
                        )
                        for number in numbers
                    ]
                    lookup = {
                        Path(image).name: number
                        for image, number in zip(images, numbers, strict=True)
                    }
                    inputs = images[0] if window == 1 else images
                    for result in pipeline.predict(inputs):
                        raw = result.json
                        data = raw.get("res", raw)
                        filename = Path(data["input_path"]).name
                        if filename not in lookup:
                            raise RuntimeError(f"Unexpected result input: {filename}")
                        number = lookup[filename]
                        if number in completed:
                            raise RuntimeError(f"Duplicate result for page {number}")
                        result.save_to_json(str(directory / f"page_{number:04d}.json"))
                        completed.append(number)
                        print(
                            f"Round {round_index + 1}, window {window}: page {number}", flush=True
                        )
                if sorted(completed) != sorted(args.pages):
                    raise RuntimeError("Missing results in trial")
                elapsed = time.perf_counter() - trial_started
                trials.append(
                    {
                        "round": round_index + 1,
                        "page_window": window,
                        "gpu_model_instances": 1,
                        "elapsed_seconds": elapsed,
                        "pages_per_minute": len(completed) * 60 / elapsed,
                        "peak_sampled_device_memory_mib": max(memory_samples)
                        if memory_samples
                        else None,
                        "mean_sampled_gpu_utilization_percent": (
                            statistics.mean(utilization_samples) if utilization_samples else None
                        ),
                        "completed_pages": completed,
                    }
                )
                checkpoint()
                print(
                    f"Trial finished: round={round_index + 1} "
                    f"window={window} seconds={elapsed:.2f}",
                    flush=True,
                )
        differences = []
        for round_index in range(1, args.rounds + 1):
            for number in args.pages:
                baseline_path = (
                    args.output_directory / "round_1/window_1" / f"page_{number:04d}.json"
                )
                baseline = content_signature(json.loads(baseline_path.read_text()))
                for window in [1, 2]:
                    path = (
                        args.output_directory
                        / f"round_{round_index}/window_{window}"
                        / f"page_{number:04d}.json"
                    )
                    candidate = content_signature(json.loads(path.read_text()))
                    if candidate != baseline:
                        differences.append(
                            {"round": round_index, "page_window": window, "page_number": number}
                        )
        medians = {
            str(window): statistics.median(
                trial["elapsed_seconds"] for trial in trials if trial["page_window"] == window
            )
            for window in [1, 2]
        }
        speedup = medians["1"] / medians["2"]
        report.update(
            status="completed",
            content_differences=differences,
            median_trial_seconds=medians,
            queued_speedup=speedup,
            recommended_page_window=2 if speedup >= 1.05 and not differences else 1,
            recommendation_scope=(
                "Provisional pilot on two pages; not a general optimum worker count"
            ),
            memory_measurement=(
                "Total GPU 0 device memory sampled once per second, not allocator peak"
            ),
            total_seconds=time.perf_counter() - started,
        )
        checkpoint()
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        checkpoint()
        raise
    finally:
        stop.set()
        sampler.join(timeout=4)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
