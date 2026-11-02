"""Optional Ragas adapter caches complete responses and preserves failed attempts."""

import json

import pytest
from pydantic import BaseModel

pytest.importorskip("ragas")

from multimodal_doc_parser.evaluation import ragas_cloud_judge as module


class Verdict(BaseModel):
    supported: bool


def test_judge_cache_replays_without_network_or_credentials_in_records(tmp_path, monkeypatch):
    calls = []

    class Client:
        model = "gpt-oss:120b"
        key = "synthetic-secret-do-not-record"

        def __init__(self, *args):
            pass

        def request(self, *args, **kwargs):
            calls.append(args)
            return dict(
                done=True,
                done_reason="stop",
                message=dict(content='```json\n{"supported":true}\n```'),
            )

    monkeypatch.setattr(module, "OllamaCloudClient", Client)
    judge = module.OllamaRagasJudge(tmp_path)
    assert judge.generate("synthetic evidence", Verdict).supported
    assert judge.generate("synthetic evidence", Verdict).supported
    assert len(calls) == 1 and judge.cache_hits == 1
    record = next(tmp_path.glob("*.json")).read_text()
    assert Client.key not in record and "Authorization" not in record


def test_incomplete_judge_output_is_not_a_score_or_an_automatic_retry(tmp_path, monkeypatch):
    calls = []

    class Client:
        model = "gpt-oss:120b"
        key = "synthetic-secret"

        def __init__(self, *args):
            pass

        def request(self, *args, **kwargs):
            calls.append(args)
            return dict(done=True, done_reason="length", message=dict(content='{"supported":true}'))

    monkeypatch.setattr(module, "OllamaCloudClient", Client)
    judge = module.OllamaRagasJudge(tmp_path)
    for _ in range(2):
        with pytest.raises(ValueError, match="incomplete"):
            judge.generate("synthetic", Verdict)
    assert len(calls) == 1
    assert (
        json.loads(next(tmp_path.glob("*.json")).read_text())["response"]["done_reason"] == "length"
    )
