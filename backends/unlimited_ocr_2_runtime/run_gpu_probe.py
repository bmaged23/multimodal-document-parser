"""Bounded, checkpointed Unlimited-OCR 2 NF4 experiment on the original PDF pages."""

import argparse
import importlib
import json
import os
import sys
import time
import traceback
import types
from pathlib import Path

import torch
from transformers import AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[2]
REVISION = "c14071b549f4fcab10909f3c4acc6a2edb366947"


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--pages", type=int, default=2)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--page-numbers", type=int, nargs="+")
    args = parser.parse_args()
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot = (
        Path(os.environ.get("UNLIMITED_OCR_CACHE_DIR", str(ROOT / ".cache/unlimited_ocr_2")))
        / REVISION
    )
    bundle = (
        args.bundle.resolve()
        if args.bundle
        else next((ROOT / "outputs/parsed_documents").glob("be2ebf*-dpi-144"))
    )
    source = json.loads((bundle / "document.json").read_text())
    numbers = args.page_numbers or list(range(1, min(args.pages, len(source["pages"])) + 1))
    if (
        not numbers
        or len(set(numbers)) != len(numbers)
        or any(number < 1 or number > len(source["pages"]) for number in numbers)
    ):
        parser.error("Page selection must contain unique valid source pages")
    compiler = ROOT / ".cache/dots_mocr/cuda126-build/toolchain/bin/x86_64-conda-linux-gnu-cc"
    if compiler.exists():
        os.environ.setdefault("CC", str(compiler))
    os.environ.setdefault("TRITON_CACHE_DIR", str(ROOT / ".cache/unlimited_ocr_2/triton"))
    status = {
        "status": "loading",
        "model": "baidu/Unlimited-OCR-2",
        "revision": REVISION,
        "document_id": source["document_id"],
        "quantization": "bitsandbytes NF4, double quant",
        "compute_dtype": "float16",
        "vision_modules": "unquantized FP16",
        "attention": "upstream dense fallback with sparse routing; Triton disabled for sm75",
        "input_dtype_override": "upstream BF16 image tensors cast to FP16 by wrapper",
        "max_new_tokens": args.max_new_tokens,
        "temperature": 0,
        "crop_mode": False,
        "mode": "single-page base view; official default Multi page parsing prompt",
        "total_pages": len(numbers),
        "prepared_bundle": str(bundle),
        "selected_pages": numbers,
        "completed_pages": 0,
        "pages": [],
        "runtime": {"python": sys.version, "torch": torch.__version__},
    }
    write(output / "run-status.json", status)
    started = time.perf_counter()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; this probe does not fall back to CPU")
        status["gpu"] = torch.cuda.get_device_name(0)
        status["cuda_capability"] = torch.cuda.get_device_capability(0)
        # Load the reviewed pinned local source as a package; no unpinned remote code.
        package = types.ModuleType("unlimited_ocr_2_pinned")
        package.__path__ = [str(snapshot)]
        sys.modules[package.__name__] = package
        module = importlib.import_module(package.__name__ + ".modeling_unlimitedocr")
        config = module.UnlimitedOCRConfig.from_pretrained(snapshot)
        config.sparse_attention_config["dash_use_triton"] = False
        config._attn_implementation = "eager"
        original_builder = module._uocr_build_inputs

        def fp16_inputs(*values, **kwargs):
            ids, images, mask, spatial = original_builder(*values, **kwargs)
            if images is not None:
                images = tuple(x.to(dtype=torch.float16) for x in images)
            return ids, images, mask, spatial

        module._uocr_build_inputs = fp16_inputs
        tokenizer = AutoTokenizer.from_pretrained(snapshot, trust_remote_code=False)
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["sam_model", "vision_model", "projector", "lm_head"],
        )
        model = module.UnlimitedOCRForCausalLM.from_pretrained(
            snapshot,
            config=config,
            use_safetensors=True,
            torch_dtype=torch.float16,
            quantization_config=quantization,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        ).eval()
        status["load_seconds"] = time.perf_counter() - started
        status["quantized_linear_modules"] = sum(
            type(m).__name__ == "Linear4bit" for m in model.modules()
        )
        status["runtime"].update(
            {
                name: importlib.import_module(name).__version__
                for name in ["transformers", "bitsandbytes", "accelerate"]
            }
        )
        for number in numbers:
            page = source["pages"][number - 1]
            status.update(status="running", current_page=number)
            write(output / "run-status.json", status)
            torch.cuda.reset_peak_memory_stats()
            page_started = time.perf_counter()
            result = model.infer(
                tokenizer,
                image_file=str(bundle / page["image"]["path"]),
                crop_mode=False,
                eval_mode=True,
                return_dict=True,
                max_new_tokens=args.max_new_tokens,
                max_page_tokens=args.max_new_tokens,
                temperature=0,
                save_results=False,
                tps_interval=100,
            )
            torch.cuda.synchronize()
            row = {
                "page_number": number,
                "elapsed_seconds": time.perf_counter() - page_started,
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
                "source_image": str(bundle / page["image"]["path"]),
                **result,
            }
            write(output / f"page_{number:04d}.json", row)
            (output / f"page_{number:04d}-raw.md").write_text(result["text"])
            status["pages"].append(
                {k: v for k, v in row.items() if k not in ["text", "token_ids", "prompt_token_ids"]}
            )
            status["completed_pages"] += int(result["finish_reason"] == "stop")
            write(output / "run-status.json", status)
            print(f"Saved page {number}: {result['finish_reason']}", flush=True)
        status["status"] = (
            "completed" if status["completed_pages"] == len(numbers) else "completed_with_errors"
        )
    except Exception as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        (output / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        status["total_seconds_including_load"] = time.perf_counter() - started
        write(output / "run-status.json", status)


if __name__ == "__main__":
    main()
