# Service APIs

**Model choice:** dots.mocr is our preferred model family overall on the 70-page test corpus. Original BF16 offers the strongest overall balance; **Q4_K_M language + Q8_0 projector** is the production default for the **GTX 1660 Ti with 6 GB VRAM**. BF16 reached **8.53 GiB peak PyTorch allocation** on Colab T4. [Comparison, measured memory and limits](../evaluation/parser-model-selection.md).

Parse PDF and `.docx` files with Docker and a compatible NVIDIA GPU; cloud chart/image descriptions also require an Ollama API key.
Legacy `.doc` and encrypted documents are unsupported, and extraction quality varies with the document.

The primary APIs prepare PDF/DOCX inputs and parse them into canonical artifacts.
Secondary APIs handle chunking, embedding, retrieval and agentic answers. The
agentic RAG API has no upload or document-processing routes.

## Install from the repository

Follow [clone and source installation](../../README.md#clone-and-install-from-source),
then run these commands from the repository root:

```bash
uv sync --locked --extra api --extra chunking
uv run python scripts/model_runtimes/setup_dots_mocr_runtime.py
uv run python scripts/model_runtimes/build_dots_mocr_runtime.py --jobs 4
# Separate terminal: uv run python scripts/model_runtimes/serve_dots_mocr.py
uv run --locked --extra api python scripts/service_management/configure_service_secrets.py
```

The helper generates a private service key in ignored `.env` and
`.cache/service_secrets/pipeline_api_key`, retaining an existing key. Live OCR
requires the NVIDIA GPU runtime; DOCX preparation also needs host LibreOffice.
The [Docker path](../docker/README.md) includes those application dependencies.

For cloud visual descriptions, set `OLLAMA_API_KEY` in the ignored `.env` before
running the helper. Keep credentials out of requests and source control. With no
Ollama access, set `enrich_visuals` to `false` in parsing jobs.

## Start the preparation and parsing services

Run each command in a separate terminal from the repository root. APIs serve
requests; workers consume queued jobs from shared `PIPELINE_STORAGE`.

```bash
# Terminal 1: preparation API
uv run --locked --extra api document-pipeline-service api --service documents --port 8101
```

```bash
# Terminal 2: preparation worker
uv run --locked --extra api document-pipeline-service worker --service documents
```

```bash
# Terminal 3: parsing API
uv run --locked --extra api document-pipeline-service api --service parsing --port 8102
```

```bash
# Terminal 4: GPU parsing worker
uv run --locked --extra api document-pipeline-service worker --service parsing
```

Open <http://127.0.0.1:8101/docs> and <http://127.0.0.1:8102/docs>. Read the private
service key locally and enter it in Swagger's **Authorize** control. Do not publish
screenshots containing the key. `/health/ready` checks storage/API readiness;
it does not prove a model is downloaded or a worker is available.

## Upload, prepare, parse and download

The same requests work against native services and parsing-only Compose. Load the
local service key without printing it:

```bash
PIPELINE_API_KEY="$(cat .cache/service_secrets/pipeline_api_key)"

curl --fail-with-body http://127.0.0.1:8101/v1/documents \
  -H "Authorization: Bearer $PIPELINE_API_KEY" \
  -F 'file=@data/input_documents/example.pdf'
```

Copy the returned `artifact_id` into `UPLOAD_ARTIFACT_ID`, then submit preparation:

```bash
UPLOAD_ARTIFACT_ID=replace_with_uploaded_artifact_id
curl --fail-with-body http://127.0.0.1:8101/v1/jobs \
  -H "Authorization: Bearer $PIPELINE_API_KEY" \
  -H 'Content-Type: application/json' \
  -d "{\"source_artifact_id\":\"$UPLOAD_ARTIFACT_ID\",\"dpi\":144}"
```

Poll `GET /v1/jobs/{job_id}` on port 8101 using the same bearer header. Once
`status` is `completed`, use `result.artifact_id` as `PREPARED_ARTIFACT_ID`:

```bash
PREPARED_ARTIFACT_ID=replace_with_completed_preparation_artifact_id
curl --fail-with-body http://127.0.0.1:8102/v1/jobs \
  -H "Authorization: Bearer $PIPELINE_API_KEY" \
  -H 'Content-Type: application/json' \
  -d "{\"source_artifact_id\":\"$PREPARED_ARTIFACT_ID\",\"backend\":\"dots-mocr\",\"enrich_visuals\":false}"
```

Poll the parsing job on port 8102. Its completed `result.artifact_id` identifies
the parsed output. `GET /v1/artifacts/{artifact_id}` returns metadata;
`GET /v1/artifacts/{artifact_id}/files/document.json` downloads canonical JSON.
The same file route serves `document.md`, `review.html` and relative assets.
Use the HTML file route in the authenticated client, or download the run with its
asset directories before opening the report locally.

Jobs return 202 when queued. An `Idempotency-Key` header reuses an identical job;
reusing that key with different input returns 409. Failed/interrupted jobs retain
diagnostics and do not silently retry expensive inference.

## Full service map

| Service | Native port | Contract and details |
| --- | --- | --- |
| Document pipeline orchestration | 8100 | [Pipeline API](../../services/document_pipeline_api/README.md) |
| PDF/DOCX preparation | 8101 | [Preparation](../../services/document_preparation/README.md) |
| OCR and visual enrichment | 8102 | [Parsing](../../services/document_parsing/README.md) |
| Structure-aware chunking | 8103 | [Chunking](../../services/rag_chunking/README.md) |
| GPU embedding and indexing | 8104 | [Indexing](../../services/embedding_indexing/README.md) |
| Evidence retrieval | 8105 | [Retrieval](../../services/retrieval/README.md) |
| Agentic RAG | 8106 | [RAG API](../../services/agentic_rag_api/README.md) |

For all seven native APIs/workers, install the optional RAG dependencies and use:

```bash
uv sync --locked --extra api --extra chunking --extra rag
uv run --locked --extra api --extra rag python scripts/service_management/configure_service_secrets.py
uv run --locked --extra api --extra rag python scripts/service_management/run_development_services.py
# Stop only the project processes started by that launcher:
uv run --locked --extra api --extra rag python scripts/service_management/run_development_services.py --stop
```

Native ports differ from the full RAG Compose exposure: parsing-only Compose
publishes 8101/8102; full `compose.yaml` publishes pipeline 8100 and RAG 8106.
These are independently running single-host services backed by shared artifacts,
SQLite job stores and a GPU lease. See [architecture and limits](../architecture/service-pipeline.md)
for storage, worker recovery, GPU serialization and the secondary retrieval flow.
