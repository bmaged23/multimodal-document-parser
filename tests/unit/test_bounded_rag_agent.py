"""The agent may retrieve twice, must preserve evidence IDs, and records failures."""

import json

import pytest

pytest.importorskip("httpx")

from multimodal_doc_parser.pipeline_services.contracts import RAGRequest
from multimodal_doc_parser.pipeline_services.settings import ServiceSettings
from multimodal_doc_parser.pipeline_services.stage_handlers import run_agent


def fake_dependencies(monkeypatch, answer_citations):
    import httpx

    import multimodal_doc_parser.pipeline_services.stage_handlers as handlers

    calls = []
    evidence = dict(
        contexts=[
            dict(
                matched_child_ids=["child:fixture"],
                text="[extracted] BY-04 mean latency is 23.1 ms.",
            )
        ]
    )

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return evidence

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, **kwargs):
            calls.append(kwargs["json"]["query"])
            return Response()

    class Cloud:
        def __init__(self, *args):
            self.count = 0

        def request(self, *args, **kwargs):
            self.count += 1
            value = (
                dict(action="rewrite", query="BY-04 mean latency table", reason="Find exact row")
                if self.count == 1
                else dict(
                    answer="BY-04 mean latency is 23.1 ms.",
                    citations=answer_citations,
                    insufficient_evidence=False,
                )
            )
            return dict(
                done_reason="stop", message=dict(content="```json\n" + json.dumps(value) + "\n```")
            )

    monkeypatch.setattr(httpx, "Client", Client)
    monkeypatch.setattr(handlers, "OllamaCloudClient", Cloud)
    return calls


def test_agent_rewrite_is_bounded_and_trace_has_no_service_secret(tmp_path, monkeypatch):
    calls = fake_dependencies(monkeypatch, ["child:fixture"])
    settings = ServiceSettings(
        storage=tmp_path,
        api_key="synthetic-private-service-key",
        urls=dict(retrieval="http://retrieval.invalid"),
    )
    result = run_agent(
        RAGRequest(index_artifact_id="a" * 32, question="Latency for BY-04?"), settings, tmp_path
    )
    assert len(calls) == 2 and result["retrieval_calls"] == 2
    assert result["answer"]["citations"] == ["child:fixture"]
    assert settings.api_key not in (tmp_path / "agent_trace.json").read_text()


def test_agent_rejects_unknown_citations_and_preserves_trace(tmp_path, monkeypatch):
    fake_dependencies(monkeypatch, ["child:invented"])
    settings = ServiceSettings(
        storage=tmp_path,
        api_key="synthetic-private-service-key",
        urls=dict(retrieval="http://retrieval.invalid"),
    )
    with pytest.raises(ValueError, match="citations"):
        run_agent(
            RAGRequest(index_artifact_id="a" * 32, question="Latency for BY-04?"),
            settings,
            tmp_path,
        )
    assert (tmp_path / "agent_trace.json").exists()
    assert not (tmp_path / "answer.json").exists()
