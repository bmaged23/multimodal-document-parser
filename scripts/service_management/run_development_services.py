"""Start or stop separate local service processes using the existing uv environment."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / "outputs/service_data/development"
MODULE = "multimodal_doc_parser.pipeline_services.service_runtime"
PORTS = dict(
    pipeline=8100,
    documents=8101,
    parsing=8102,
    chunking=8103,
    indexing=8104,
    retrieval=8105,
    rag=8106,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stop", action="store_true")
    args = parser.parse_args()
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    registry = DIRECTORY / "processes.json"
    if registry.exists():
        previous = json.loads(registry.read_text())
        active = []
        for process in previous:
            try:
                command = Path(f"/proc/{process['pid']}/cmdline").read_bytes().split(b"\0")
                if MODULE.encode() in command:
                    active.append(process)
            except OSError:
                pass
        if args.stop:
            for process in active:
                os.kill(process["pid"], signal.SIGTERM)
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                remaining = []
                for process in active:
                    try:
                        command = Path(f"/proc/{process['pid']}/cmdline").read_bytes().split(b"\0")
                        if MODULE.encode() in command:
                            remaining.append(process)
                    except OSError:
                        pass
                if not remaining:
                    break
                time.sleep(0.1)
            if remaining:
                raise ValueError("Some services are still stopping; inspect their logs")
            registry.write_text("[]\n")
            print(f"Stopped {len(active)} project service processes")
            return
        if active:
            raise ValueError("Development services are already active; stop them before restarting")
    if args.stop:
        return
    records = []
    try:
        for service, port in PORTS.items():
            for role in ["api"] if service == "retrieval" else ["api", "worker"]:
                command = [sys.executable, "-m", MODULE, role, "--service", service]
                if role == "api":
                    command.extend(["--port", str(port)])
                log_path = DIRECTORY / f"{service}-{role}.log"
                with log_path.open("a") as log:
                    process = subprocess.Popen(
                        command,
                        cwd=ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                records.append(
                    dict(
                        service=service,
                        role=role,
                        pid=process.pid,
                        port=port if role == "api" else None,
                    )
                )
        registry.write_text(json.dumps(records, indent=2) + "\n")
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        pending = set(PORTS.values())
        deadline = time.monotonic() + 30
        while pending and time.monotonic() < deadline:
            for port in list(pending):
                try:
                    with opener.open(
                        f"http://127.0.0.1:{port}/health/ready", timeout=1
                    ) as response:
                        if response.status == 200:
                            pending.remove(port)
                except OSError:
                    pass
            if pending:
                time.sleep(0.1)
        if pending:
            raise ValueError(f"Service APIs failed readiness: {sorted(pending)}; inspect logs")
        print("Started independent local APIs and workers. Public API docs:")
        print("Document pipeline: http://127.0.0.1:8100/docs")
        print("Agentic RAG: http://127.0.0.1:8106/docs")
    except BaseException:
        for record in records:
            try:
                os.kill(record["pid"], signal.SIGTERM)
            except ProcessLookupError:
                pass
        raise


if __name__ == "__main__":
    main()
