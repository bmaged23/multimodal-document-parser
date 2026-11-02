"""Run real preparation/parsing workers on a synthetic PDF in the release image.

Uses already-cached, checksum-verified quantized weights; no cloud credentials,
private evaluation inputs, or parsing-quality benchmark claims.
"""

import argparse
import subprocess
from pathlib import Path

from reportlab.pdfgen.canvas import Canvas

PROBE = r"""
import json, shutil
from pathlib import Path
from multimodal_doc_parser.pipeline_services.settings import ServiceSettings
from multimodal_doc_parser.pipeline_services.storage import ArtifactStore, JobStore
from multimodal_doc_parser.pipeline_services.service_runtime import work_once
settings = ServiceSettings.from_environment()
store = ArtifactStore(settings.storage)
identity, directory = store.reserve()
bundle = directory / 'upload'; bundle.mkdir()
shutil.copyfile('/probe.pdf', bundle / 'probe.pdf')
store.publish(identity, 'upload', bundle, filename='probe.pdf')
preparation = JobStore(settings.storage, 'documents')
job, _ = preparation.enqueue({'source_artifact_id': identity, 'dpi': 144}, None)
assert work_once('documents', settings)
prepared = preparation.get(job['id'])
assert prepared['status'] == 'completed', prepared
parsing = JobStore(settings.storage, 'parsing')
request = {'source_artifact_id': prepared['result']['artifact_id'], 'enrich_visuals': False}
job, _ = parsing.enqueue(request, None)
assert work_once('parsing', settings)
parsed = parsing.get(job['id'])
assert parsed['status'] == 'completed', parsed
result = store.resolve(parsed['result']['artifact_id'], 'parsed')
document = json.loads((result / 'document.json').read_text())
text = (result / 'document.md').read_text()
normalized = ''.join(c.lower() for c in text if c.isalnum())
for expected in ['orbit42', 'alpha', 'beta', '12', '34']:
    assert expected in normalized, (expected, text)
assert any(e['type'] == 'table' for e in document['elements'])
manifest = json.loads((result / 'raw/dots_mocr/inference-manifest.json').read_text())
assert manifest['backend'] == 'dots.mocr'
assert manifest['completed_pages'] == 1
assert 'Q4_K_M' in manifest['model_version'] and 'Q8_0' in manifest['model_version']
assert (result / 'review.html').is_file()
servers = []
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit(): continue
    try:
        command = (proc / 'cmdline').read_bytes().split(b'\0')[0]
        if command.endswith(b'/llama-server'): servers.append(proc.name)
    except (OSError, PermissionError): pass
assert not servers, servers
report = {'result': 'passed', 'image_default_backend': 'dots-mocr',
          'prepared_job': prepared['id'], 'parsed_job': parsed['id'],
          'model_version': manifest['model_version'], 'completed_pages': 1,
          'synthetic_reference_checks': 5, 'table_detected': True,
          'owned_server_released': True, 'cloud_requests': 0,
          'artifact_id': parsed['result']['artifact_id']}
Path('/data/gpu-release-verification.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report))
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--model-directory", type=Path, default=Path(".cache/dots_mocr"))
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=False)
    pdf = output / "probe.pdf"
    canvas = Canvas(str(pdf), pagesize=(400, 500))
    canvas.setFont("Helvetica-Bold", 17)
    canvas.drawString(35, 455, "GPU parsing release verification")
    canvas.setFont("Helvetica", 13)
    canvas.drawString(35, 420, "Reference code: ORBIT-42")
    for y in [350, 310, 270, 230]:
        canvas.line(35, y, 355, y)
    for x in [35, 235, 355]:
        canvas.line(x, 230, x, 350)
    for y, left, right in [(325, "Sensor", "Count"), (285, "Alpha", "12"), (245, "Beta", "34")]:
        canvas.drawString(50, y, left)
        canvas.drawString(250, y, right)
    canvas.save()
    script = output / "probe.py"
    script.write_text(PROBE)
    command = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges:true",
        "--tmpfs",
        "/tmp",
        "-v",
        f"{output}:/data",
        "-v",
        f"{args.model_directory.resolve()}:/cache/dots_mocr:ro",
        "-v",
        f"{pdf}:/probe.pdf:ro",
        "-v",
        f"{script}:/probe.py:ro",
        "-e",
        "PIPELINE_API_KEY=synthetic-release-probe-not-a-real-secret",
        "--entrypoint",
        "/app/.venv/bin/python",
        args.image,
        "/probe.py",
    ]
    subprocess.run(command, check=True, timeout=600)


if __name__ == "__main__":
    main()
