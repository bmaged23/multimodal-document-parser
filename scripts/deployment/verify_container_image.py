"""Smoke-test the distributable parsing image without inference or credentials."""

import argparse
import json
import secrets
import subprocess
import sys
import time
import uuid


def docker(*arguments: str, timeout: int = 120, quiet: bool = False) -> str:
    try:
        return subprocess.run(
            ["docker", *arguments], check=True, capture_output=True, text=True, timeout=timeout
        ).stdout.strip()
    except subprocess.CalledProcessError as error:
        if not quiet:
            print(error.stderr, file=sys.stderr, flush=True)
        raise


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("--gpu", action="store_true", help="Also verify native CUDA GPU discovery")
    args = parser.parse_args()
    name = "parser-smoke-" + uuid.uuid4().hex[:12]
    key = secrets.token_urlsafe(32)
    try:
        docker(
            "run",
            "-d",
            "--name",
            name,
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges:true",
            "--tmpfs",
            "/tmp",
            "--tmpfs",
            "/data:uid=1000,gid=1000",
            "--tmpfs",
            "/cache:uid=1000,gid=1000",
            "-e",
            "PIPELINE_API_KEY=" + key,
            args.image,
            "api",
            "--service",
            "parsing",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            timeout=600,  # First use may unpack several GiB of GPU libraries.
        )
        probe = """
import json, os, pathlib, urllib.request, urllib.error
assert os.getuid() != 0
root = pathlib.Path('/app')
for name in ['.env', '.cache', 'outputs', 'data/input_documents', '.git']:
    assert not (root / name).exists(), name
assert pathlib.Path('/app/THIRD_PARTY_NOTICES.md').exists()
assert pathlib.Path('/opt/dots/bin/llama-server').exists()
assert not pathlib.Path('/usr/local/cuda-12.6/lib64/stubs/libcuda.so.1').exists()
assert not pathlib.Path('/app/backends/paddleocr_runtime/.venv').exists()
base = 'http://127.0.0.1:8000'
assert json.load(urllib.request.urlopen(base + '/health/ready'))['status'] == 'ready'
try:
    urllib.request.urlopen(base + '/v1/jobs/' + '0' * 32)
    raise AssertionError('Authentication missing')
except urllib.error.HTTPError as exc:
    assert exc.code == 401
assert 'paths' in json.load(urllib.request.urlopen(base + '/openapi.json'))
print('Non-root API, storage, authentication, OpenAPI and image exclusions passed')
"""
        for attempt in range(30):
            try:
                print(docker("exec", name, "/app/.venv/bin/python", "-c", probe, quiet=True))
                break
            except subprocess.CalledProcessError:
                if attempt == 29:
                    print(docker("logs", name))
                    raise
                time.sleep(1)
        # A CPU-only hosted runner has no host libcuda: inspect linkage there,
        # and execute the native binary only in an NVIDIA-enabled container.
        native_probe = """
import subprocess
from multimodal_doc_parser.pipeline_services.contracts import ParsingJob, PipelineJob
from multimodal_doc_parser.parser_adapters.defaults import DOTS_RUNTIME
assert ParsingJob.model_fields['backend'].default == 'dots-mocr'
assert PipelineJob.model_fields['backend'].default == 'dots-mocr'
assert len([a for a in DOTS_RUNTIME['artifacts'] if a['kind']=='model']) == 2
result = subprocess.run(['ldd','/opt/dots/bin/libggml-cuda.so'],
                        capture_output=True, text=True, check=True)
missing = {line.split()[0] for line in result.stdout.splitlines() if 'not found' in line}
assert missing <= {'libcuda.so.1'}, missing
print('Pinned quantized dots configuration, native shared libraries and API defaults passed')
"""
        print(docker("exec", name, "/app/.venv/bin/python", "-c", native_probe))
        print(docker("exec", name, "libreoffice", "--version"))
        preparation_probe = """
import json, tempfile, zipfile
from pathlib import Path
from multimodal_doc_parser.pipeline_services.stage_handlers import convert_docx
from multimodal_doc_parser.pdf_preparation.prepare_document import prepare_document
from multimodal_doc_parser.visual_enrichment.ollama import enrich_document
from multimodal_doc_parser.visual_enrichment.ollama_cloud import OllamaCloudClient
root = Path(tempfile.mkdtemp(prefix='container-docx-', dir='/tmp'))
source = root / 'smoke.docx'
with zipfile.ZipFile(source, 'w') as archive:
    archive.writestr('[Content_Types].xml',
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" '
        'ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        '</Types>')
    archive.writestr('_rels/.rels',
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="word/document.xml"/></Relationships>')
    archive.writestr('word/document.xml',
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        '<w:body><w:p><w:r><w:t>Container preparation smoke test</w:t></w:r></w:p>'
        '</w:body></w:document>')
pdf = convert_docx(source, root / 'conversion')
bundle = prepare_document(pdf, root / 'prepared', dpi=72, max_pages=2)
document = json.loads((bundle / 'document.json').read_text())
assert len(document['pages']) == 1
assert (bundle / document['pages'][0]['image']['path']).is_file()
assert callable(enrich_document) and callable(OllamaCloudClient)
print('Worker/VLM imports, DOCX-to-PDF conversion and page rendering passed')
"""
        print(docker("exec", name, "/app/.venv/bin/python", "-c", preparation_probe))
        if args.gpu:
            print(
                docker(
                    "run",
                    "--rm",
                    "--gpus",
                    "all",
                    "--entrypoint",
                    "/opt/dots/bin/llama-server",
                    args.image,
                    "--version",
                )
            )
            devices = docker(
                "run",
                "--rm",
                "--gpus",
                "all",
                "--entrypoint",
                "/opt/dots/bin/llama-server",
                args.image,
                "--list-devices",
            )
            assert "CUDA0:" in devices, devices
            print(devices)
        print(
            json.dumps(
                dict(
                    image=args.image,
                    gpu_test=args.gpu,
                    inference_requests=0,
                    cloud_requests=0,
                    result="passed",
                )
            )
        )
    finally:
        # Remove only the transient container created by this check.
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)


if __name__ == "__main__":
    main()
