"""Run the pinned original dots.mocr weights on a separate >=12 GiB cloud GPU.

Standalone for Colab; never falls back to CPU and never enables low-bit loading.
Inputs are the exact PNG bytes used by the preserved quantized GPU baseline.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import shutil
import sys
import time
import types
from collections import Counter
from pathlib import Path


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=6144)
    parser.add_argument("--page-seconds", type=int, default=300)
    parser.add_argument("--weight-dtype", choices=["auto", "bfloat16", "float16"], default="auto")
    args = parser.parse_args()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("A cloud GPU is required; CPU fallback is disabled")
    props = torch.cuda.get_device_properties(0)
    if props.total_memory < 12 * 1024**3:
        raise RuntimeError("Use a cloud GPU with >=12 GiB VRAM; this is not the laptop profile")
    if args.output.exists():
        raise FileExistsError("Use a fresh output directory; previous results are immutable")
    args.output.mkdir(parents=True)
    manifest = json.loads((args.inputs / "input-manifest.json").read_text())
    for relative, digest in manifest["files"].items():
        path = (args.inputs / relative).resolve()
        if not path.is_relative_to(args.inputs.resolve()):
            raise ValueError("Unsafe input path")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Input checksum mismatch: " + relative)
    shutil.copyfile(args.inputs / "input-manifest.json", args.output / "input-manifest.json")

    from PIL import Image
    from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

    # The pinned vision module imports FlashAttention unconditionally. Its SDPA
    # class does not call it. On T4, permit that unused import without installing
    # an unsupported kernel; fail loudly if execution ever selects the stub.
    unused_import_shim = importlib.util.find_spec("flash_attn") is None
    if unused_import_shim:
        stub = types.ModuleType("flash_attn")
        stub.__spec__ = importlib.util.spec_from_loader("flash_attn", loader=None)

        def forbidden_flash_attention(*values, **kwargs):
            raise RuntimeError("FlashAttention was called despite the explicit SDPA configuration")

        stub.flash_attn_varlen_func = forbidden_flash_attention
        sys.modules["flash_attn"] = stub

    use_bf16 = args.weight_dtype == "bfloat16" or (
        args.weight_dtype == "auto" and torch.cuda.is_bf16_supported()
    )
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    model_id, revision = manifest["model_id"], manifest["model_revision"]
    config = AutoConfig.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    config.vision_config.attn_implementation = "sdpa"
    started = time.perf_counter()
    print(
        f"Loading original {model_id}@{revision} on {props.name}, {dtype}, no quantization",
        flush=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        config=config,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map={"": 0},
        attn_implementation="sdpa",
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    if getattr(model, "is_quantized", False) or getattr(model.config, "quantization_config", None):
        raise RuntimeError("Unexpected quantized model; refusing to label it unquantized")
    if any(type(m).__name__ in {"Linear4bit", "Linear8bitLt"} for m in model.modules()):
        raise RuntimeError("Low-bit modules found")
    parameter_dtypes = Counter()
    for parameter in model.parameters():
        parameter_dtypes[str(parameter.dtype)] += parameter.numel()
    metadata = {
        "model": model_id,
        "revision": revision,
        "gpu": props.name,
        "gpu_vram_bytes": props.total_memory,
        "device": "cuda:0",
        "weight_dtype": str(dtype),
        "parameter_dtype_counts": dict(parameter_dtypes),
        "quantization": None,
        "attention": "sdpa",
        "unused_flash_import_shim": unused_import_shim,
        "model_load_seconds": time.perf_counter() - started,
        "max_new_tokens": args.max_new_tokens,
        "page_seconds": args.page_seconds,
        "temperature": 0,
        "sampling": False,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ["torch", "transformers", "accelerate", "pillow"]
        },
        "limits": [
            "GGUF converter lineage is unverified; this is not a causal quantization experiment.",
            "Cloud GPU, attention backend and processor differ from the local GGUF runtime.",
            "Torch checks can include BF16 emulation; actual parameter dtype is recorded.",
        ],
    }
    save(args.output / "runtime.json", metadata)
    summary = {
        "status": "running",
        "total_pages": sum(len(c["pages"]) for c in manifest["cases"]),
        "attempted_pages": 0,
        "completed_pages": 0,
        "failed_pages": 0,
        "cases": [],
    }
    save(args.output / "progress.json", summary)
    torch.manual_seed(0)
    prompt_template = (args.inputs / "layout-prompt.txt").read_text()
    for case in manifest["cases"]:
        case_status = {
            "id": case["id"],
            "source_document_id": case["source_document_id"],
            "pages": [],
        }
        summary["cases"].append(case_status)
        for page in case["pages"]:
            number = page["page_number"]
            destination = args.output / case["id"] / f"page_{number:04d}.json"
            row = {
                "page_number": number,
                "original_pdf_page": page["original_pdf_page"],
                "status": "running",
            }
            case_status["pages"].append(row)
            summary.update(current_case=case["id"], current_page=number)
            save(args.output / "progress.json", summary)
            print(f"START {case['id']} page {number}/{len(case['pages'])}", flush=True)
            tick = time.perf_counter()
            try:
                image = Image.open(args.inputs / page["input_image"]).convert("RGB")
                prompt = prompt_template.format(width=image.width, height=image.height)
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt},
                        ],
                    }
                ]
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = processor(
                    text=[text], images=[image], padding=True, return_tensors="pt"
                ).to("cuda")
                torch.cuda.reset_peak_memory_stats()
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        max_time=args.page_seconds,
                        do_sample=False,
                    )
                torch.cuda.synchronize()
                tokens = generated[:, inputs.input_ids.shape[1] :]
                content = processor.batch_decode(
                    tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
                eos = model.generation_config.eos_token_id
                eos = [eos] if isinstance(eos, int) else (eos or [])
                finish = (
                    "stop"
                    if int(tokens[0, -1]) in eos
                    else ("length" if tokens.shape[1] >= args.max_new_tokens else "time_limit")
                )
                result = {
                    **page,
                    "content": content,
                    "finish_reason": finish,
                    "elapsed_seconds": time.perf_counter() - tick,
                    "output_tokens": tokens.shape[1],
                    "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
                }
                save(destination, result)
                destination.with_suffix(".txt").write_text(content)
                row.update(
                    status="completed" if finish == "stop" else "incomplete",
                    finish_reason=finish,
                    elapsed_seconds=result["elapsed_seconds"],
                )
                summary["completed_pages"] += int(finish == "stop")
                summary["failed_pages"] += int(finish != "stop")
                del inputs, generated, tokens
            except Exception as exc:
                row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                summary["failed_pages"] += 1
                save(destination, row)
                torch.cuda.empty_cache()
            summary["attempted_pages"] += 1
            save(args.output / "progress.json", summary)
            print(
                f"DONE {summary['attempted_pages']}/{summary['total_pages']}: {row['status']}",
                flush=True,
            )
    summary.update(
        status="completed_with_errors" if summary["failed_pages"] else "completed",
        current_case=None,
        current_page=None,
    )
    save(args.output / "progress.json", summary)
    files = {
        str(p.relative_to(args.output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in args.output.rglob("*")
        if p.is_file()
    }
    save(args.output / "file-checksums.json", files)


if __name__ == "__main__":
    main()
