# Cloud model selection history

Historical research; the selected default is now Gemma 4 31B on Ollama Cloud.

## Visual enrichment and cloud model selection

Visual enrichment adds generated descriptions, chart values and flowchart connections
in a fresh copy of an OCR run. Exact model identity, prompts, raw responses, image
hashes and source links are retained separately from extracted OCR content.

The local Qwen3-VL 2B and Qwen3.5 4B experiments are preserved for evaluation.
Qwen3.5 4B completed three visual responses and passed 6/9 selected assertions;
these results do not establish overall accuracy. Its local model weights were
removed at the user's request; the project and saved outputs remain intact.

### Cloud VLM selection (pending)

The selected cloud VLM is **Qwen3.8-Flash** (`qwen3.8-flash`), accessed through
Qwen Cloud's regional API. The transport and validation pipeline are implemented;
real cloud inference is pending credentials. No cloud quality score is claimed.

| Property | Qwen3.8-Flash reference specification |
| --- | --- |
| Provider / API model ID | Qwen Cloud / `qwen3.8-flash` |
| Architecture | Multimodal mixture of experts (MoE) |
| Published parameter count | **125B main-model parameters; approximately 6B activated per token** |
| Input modalities | Text, images and video |
| Project role | Selected cloud model for visual enrichment |
| Integration / evaluation status | Adapter implemented; real inference pending credentials |

The parameter figures are provider-reported architecture specifications, not
measurements made by this repository. Qwen Cloud's overview describes 125B parameters
and 6B activated per token, and separately discusses an additional 51B n-gram
embedding component. For the related **open-weight Qwen3.8-Flash-Next** checkpoint,
Qwen lists 125B main-model + 51B n-gram embedding + 4B multi-token prediction (MTP)
parameters, approximately **180B across those components**. We do not assume that
this full checkpoint count describes the exact hosted API deployment.

Sources, checked 2026-10-31:
[Qwen Cloud model specification](https://docs.qwencloud.com/developer-guides/getting-started/latest-model)
and [official Flash-Next model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).

Qwen Cloud and Ollama Cloud are separate providers. The current public Ollama Cloud
catalog does not list this Qwen model, using this candidate would require Qwen Cloud
credentials and endpoint; it is not an implemented Ollama Cloud integration.
Cloud model availability depends on region. Credentials must stay outside Git.

Authentication check (2026-10-31): an Ollama API credential was retrieved from the
user's connected Google Drive and stored only in the ignored local `.env`.
The authenticated Ollama Cloud catalog listed no Qwen models, and a minimal
`qwen3.8-flash` request returned HTTP 404 (model not found). This is not a Qwen
Cloud authentication test. Direct Qwen Cloud inference still requires a matching
regional `DASHSCOPE_API_KEY`; no cloud visual evaluation has run yet.


Copy `.env.example` to `.env`, set `DASHSCOPE_API_KEY` and the matching regional
`QWEN_BASE_URL`, then run:

```bash
uv run document-parser enrich outputs/parser_runs/paddleocr-vl-1.6-reviewed \
  --provider qwen-cloud --model qwen3.8-flash \
  --output-directory outputs/parser_runs/paddleocr-qwen-flash-cloud-01
```

The cloud receives the selected crop and page-context images. Requests use JSON
Object mode with the schema in the prompt; responses undergo the same strict
local validation. Native responses retain provider IDs, returned model alias and
usage. A hosted alias does not pin an immutable checkpoint revision. API keys are
never written to manifests or reports. Local-only context/seed/memory settings
are not claimed for cloud execution.


The existing local Ollama adapter remains available for reproducing historical
experiments with the project uv environment. See
[setup](../development/ollama-visual-enrichment.md) and
[measured results and limits](ollama-visual-enrichment-validation.md).
