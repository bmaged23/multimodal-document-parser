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
    parser.add_argument("--gpu", action="store_true", help="Also execute a small CUDA tensor")
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
assert pathlib.Path('/app/backends/paddleocr_runtime/.venv/bin/python').exists()
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
        runtime = "/app/backends/paddleocr_runtime/.venv/bin/python"
        print(
            docker(
                "exec",
                name,
                runtime,
                "-c",
                "import subprocess; from pathlib import Path; "
                "from importlib.metadata import version; "
                "assert version('paddleocr') == '3.6.0'; "
                "assert version('paddlepaddle-gpu') == '3.2.1'; "
                "core=Path('/app/backends/paddleocr_runtime/.venv/lib/python3.12/'"
                "'site-packages/paddle/base/libpaddle.so'); assert core.is_file(); "
                "linked=subprocess.run(['ldd',str(core)],check=True,"
                "capture_output=True,text=True); "
                "missing={line.split()[0] for line in linked.stdout.splitlines() "
                "if 'not found' in line}; "
                "assert missing <= {'libcuda.so.1'}, 'Missing runtime libraries: '+str(missing); "
                "print('GPU package metadata/shared libraries passed; "
                "host libcuda is supplied by the NVIDIA runtime')",
            )
        )
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
                    "-e",
                    "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True",
                    "-e",
                    "DISABLE_MODEL_SOURCE_CHECK=True",
                    "--entrypoint",
                    runtime,
                    args.image,
                    "-c",
                    "import paddle; assert paddle.is_compiled_with_cuda(); "
                    "from paddleocr import PaddleOCRVL; assert callable(PaddleOCRVL); "
                    "paddle.set_device('gpu:0'); "
                    "assert float(paddle.ones([2, 2]).sum()) == 4; "
                    "print('CUDA tensor passed; GPUs:', paddle.device.cuda.device_count())",
                )
            )
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
