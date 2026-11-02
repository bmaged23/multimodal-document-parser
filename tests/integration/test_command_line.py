"""Check the installed user interface and its error contract."""

import json
import subprocess
from pathlib import Path


def test_schema_command_and_missing_input(tmp_path: Path) -> None:
    result = subprocess.run(
        ["document-parser", "schema"], capture_output=True, text=True, check=True
    )
    assert json.loads(result.stdout)["title"] == "Document"
    failed = subprocess.run(
        ["document-parser", "prepare", str(tmp_path / "missing.pdf")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert failed.returncode == 1
    assert "does not exist" in failed.stderr
    assert "Traceback" not in failed.stderr
