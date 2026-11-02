# Document processing design

Planned flow:

```text
Input PDF → page preparation → parser adapter → canonical document JSON
                                                    ↓
                                  chart and image enrichment
                                                    ↓
                            Markdown export / retrieval chunks (JSONL)
```

Parser-specific raw results will be preserved independently of normalized data.
Each normalized element will carry source page and coordinates. Generated
descriptions will be distinct from extracted text. Chart values will distinguish
printed labels, visual estimates, and unknown values.

Chunking will consume canonical documents so changing chunk settings does not
require rerunning inference. The schema and PDF preparation are implemented in milestone 2. Parser adapters,
enrichment, and export stages are planned for subsequent milestones.

Add implementation modules as their milestones begin. Keep incompatible model
runtimes in separate environments or services if compatibility testing requires it.
