"""Import only the matching 46-page BF16 Colab result, preserving its original ZIP."""

import argparse
import json
import time
import zipfile
from pathlib import Path

from run_demo_model_matrix import ROOT, load_script


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads", type=Path, default=Path("/mnt/d/stepn/Downloads"))
    args = parser.parse_args()
    inputs = ROOT / "outputs/maintenance/four-model-demo-colab-inputs-01"
    destination = ROOT / "outputs/maintenance/four-model-demo-colab-import-01"
    if destination.exists():
        raise ValueError("An import already exists; retain it before another attempt")
    print("Waiting for the matching 46-page BF16 Colab result ZIP in Downloads", flush=True)
    while True:
        candidates = sorted(
            args.downloads.glob("dots_mocr_unquantized_demo_46_pages_results*.zip"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for archive in candidates:
            try:
                with zipfile.ZipFile(archive) as zipped:
                    if zipped.testzip() is not None:
                        continue
                    manifest = json.loads(zipped.read("input-manifest.json"))
                    runtime = json.loads(zipped.read("runtime.json"))
                if manifest != json.loads((inputs / "input-manifest.json").read_text()):
                    continue
                if runtime["weight_dtype"] != "torch.bfloat16":
                    raise ValueError("This demo column requires the declared BF16 storage profile")
                entries = load_script("import_dots_mocr_cloud_results").import_results(
                    archive, inputs, destination
                )
                archiver = load_script("archive_evaluation_runs")
                for entry in entries:
                    archiver.snapshot(entry, ROOT / "outputs/evaluation_archive")
                print("Matched cloud results imported and preserved", flush=True)
                return
            except (zipfile.BadZipFile, EOFError, KeyError):
                # Chrome can create its final filename before all bytes arrive.
                continue
        time.sleep(15)


if __name__ == "__main__":
    main()
