# Local Agent Operations

For the complete file inventory, architecture, process-graph explanation and user
walkthrough, start with [the migration guide](MIGRATION_GUIDE.md). Commands below run
from the repository root.

## Architecture and migration

`local_agent.py` replaces the Claude Code runtime, not the industrial query layer.
It starts `historian_mcp_server.py` with the project virtual environment and connects
using MCP stdio. Ollama chat and BGE-M3 embeddings use the private plant network.
HTTP proxy environment variables are not inherited by model HTTP clients, redirects
are not followed, public inference addresses are rejected and Chroma telemetry is disabled.

The existing DuckDB queries, alarm state machine, equipment graph, document tools,
plots, PCA MSPC, Isolation Forest, XGBoost and audit renderer remain in place.
`CLAUDE.md` remains the original domain reference. Its operating rules, query workflows
and audited RCA method are loaded unchanged; duplicate API/tag tables and codebase
layout are omitted from the runtime prompt because MCP supplies authoritative schemas
and tag metadata. Original ML/MSPC command documents are loaded when selected or when
their analytic tool runs, rather than consuming context for every routine question.
The local execution contract overrides
only Claude-specific mechanics: subprocess startup, allowlisted analytics, automatic
audit lifecycle and artifact links. Claude itself and `.mcp.json` are not needed.

The former prototype had a hard-coded endpoint/model, one question per process,
ten model rounds, no analytics execution tools, no evidence-step transport and no
automated regression suite. It caught tool failures but could still print an answer
or finish an incomplete investigation as if successful. The old manual inspection,
patch and backup files are retained without content edits under
`archive/migration_experiments/`; they are not part of the new runtime.

The historical JSON-RPC errors show newline-only input to the stdio server. A stdio
MCP server is not an interactive shell: start the client, not the server directly.
Closing an audit from a second MCP process cannot close the first process's in-memory
session. The client now owns startup, tool calls and finalization in one connection.

The in-progress RAG migration already used BGE-M3, but search still added the old
BGE-small query prefix. Indexing and querying now share an embedding implementation,
empty prefix, endpoint configuration and response validation. Indexes record embedding
model and dimension on build. A different recorded model is rejected; a legacy index
without provenance emits a warning and checks dimensions at query time. Matching
dimensions cannot prove matching embedding spaces: rebuild such an index before
relying on its provenance. Exact tag/equipment filters no longer match substrings.

## Start and configure

```powershell
.\.venv\Scripts\python.exe local_agent.py --check
.\.venv\Scripts\python.exe local_agent.py
.\.venv\Scripts\python.exe local_agent.py "What happened to steam temperature in the last hour?"
.\.venv\Scripts\python.exe local_agent.py --diagnostic "Why did steam temperature drop on March 29?"
.\.venv\Scripts\python.exe local_agent.py "/mspc-analysis"
.\.venv\Scripts\python.exe local_agent.py "/ml-fault-detection"
.\.venv\Scripts\python.exe local_agent.py --audit-trail
.\.venv\Scripts\python.exe local_agent.py --resume RUN_ID
```

| Setting | Default | Purpose |
|---------|---------|---------|
| `OLLAMA_URL` / `--url` | `http://172.19.120.10:11434` | Private Ollama origin; also passed to MCP embeddings |
| `OLLAMA_MODEL` / `--model` | `qwen3.5:35b` | Tool-capable chat model |
| `OLLAMA_EMBED_MODEL` | `bge-m3:latest` | Embedding model; changing it requires rebuilding the index |
| `--context` | `32768` | Requested Ollama context budget |
| `--max-rounds` | `80` | Maximum model responses per question, not a completion guarantee |
| `--timeout` | `300` seconds | Each Ollama request timeout |
| `--diagnostic` | Off | Force full diagnostic evidence/completion checks |
| `--think` / `--no-think` | Disabled | Enable/disable native thinking on compatible models; structured audit reasoning remains required |
| `--protocol` | `json` | Schema-enforced JSON actions; `native` retains experimental native Ollama function calls |

English/Spanish diagnostic cues also activate diagnostic checks. Use `--diagnostic`
for ambiguous requests; intent recognition is not a universal language classifier.
History is conversational context only: each question gets new evidence and a new audit.
`/new` clears in-memory conversation context. Resume restores the completed
conversation saved with the selected run, not an open audit or hidden internal state.
For older transcripts without a history field, it restores the last question and answer.

The CLI validates installed models and tool capability before querying plant data.
It never automatically switches models, repeats a state-changing tool call, installs
packages during a question or exposes a general shell/file-reading tool to the model.
Analytic tools accept validated arguments and invoke only the two existing scripts,
with timeouts and child-process cleanup.

The default JSON action protocol supplies an Ollama response schema for the currently
allowed actions, then translates validated actions to MCP calls. Hypotheses require IDs;
post-query observations require the exact returned evidence step, an ID and a structured
confirmed/rejected/inconclusive outcome. Conclusions require confidence and evidence.
These requirements are enforced by the response schema as well as local validation.
Native tool calling remains available for comparison, but cannot be assumed equivalent:
live tests exposed repeated calls to tools unavailable in the current diagnostic phase.
Routine lookups and analysis commands do not expose diagnostic reasoning tools. Slash
analysis commands execute their requested existing script deterministically through MCP
before asking the model for interpretation. Their mandatory cross-references also run
deterministically: MSPC retrieves alarm context for the three worst events per statistic
and traces each leading contributor; ML retrieves context and active alarms for the five
worst periods, graph/aggregate alarm evidence for the three leading features, and a
relevant troubleshooting document. Repeated queries are deduplicated without dropping
distinct contributors at a shared timestamp. Both commands query the actual steam
temperature extremes and display a host-generated out-of-range warning.

## Model selection

See [the seven-model evaluation](MODEL_EVALUATION.md) for the latest common-profile
comparison and its machine-readable results. Earlier probes below used different
profiles and are historical observations, not directly comparable speed rankings.

```powershell
.\.venv\Scripts\python.exe local_agent.py --benchmark qwen3.5:35b qwen3.6:27b qwen3.8:27b
```

This synthetic two-turn probe checks exact function arguments and interpretation of
the tool result. It sends no plant data. It is not a clinical, industrial accuracy
or reasoning benchmark. During migration, all three models passed once:
12.69, 18.69 and 15.44 seconds respectively, including model loading effects.
Keep the existing Qwen 3.5 35B default until a representative industrial evaluation
supports a different choice. The installed 122B model is not automatically preferred:
GPU memory, latency and diagnostic quality require measurement.

The 122B model also passed a synthetic native protocol probe (29.02 seconds with
native thinking disabled), but did not solve the diagnostic contract by size alone.
With schema-enforced JSON actions and native thinking disabled, the 35B model passed
the synthetic probe in 1.38 seconds and completed a real audited investigation.
These single observations are not controlled speed or diagnostic-quality rankings:
loading, caching, protocol and thinking settings differ.

## Evidence and failure behavior

Before each question, the runtime retrieves snapshot bounds and confirmed tags.
Query timestamps must lie within that snapshot; unknown tags are rejected.
Long raw time-series requests are downsampled when no interval is specified.
Steam-temperature plots always show the normal range. Statistics and retrieved
time series trigger an explicit host-generated warning if TE_8332A leaves 530-545 C.

MCP domain responses preserve their original JSON text and also provide structured
`result` and `audit_step` fields. Failures set MCP `isError` and mark the audit entry
unsuccessful. The model receives explicit tool errors and may correct arguments.
Reasoning may reference only actual successful evidence steps from the current run.
Previously recorded reasoning may also be cited when it resolves transitively to
successful data evidence; bare hypotheses do not constitute evidence.

For diagnosis, data calls require a prior logged hypothesis and subsequent observation.
Use `[hypothesis:H1]` IDs, resolve them as confirmed/rejected/inconclusive, and record
the control-loop, sensor-reliability and upstream-disturbance assessment with
`[cause-versus-symptom]`. Completion requires a final conclusion with confidence and
a corrective action citing an actually retrieved document's title, `Rev` and step.
Hypotheses without IDs are rejected before persistence, and exact duplicate reasoning
entries are rejected with the remaining completion requirements. This prevents
unrepairable completion requirements and repeated conclusions from consuming context.
Diagnostic tool availability is phase-gated: reasoning before a query, data tools after
a hypothesis, and only observation/rejection/conclusion immediately after the result.
The outcome must reference that result and explicitly resolve its hypothesis as
confirmed, rejected or inconclusive. This is runtime enforcement, not just prompting.
An uncertain candidate is allowed; unsupported certainty is not.
When no anomaly is confirmed, the agent can close with a grounded no-anomaly
conclusion rather than fabricating a cause or an unnecessary corrective action.
Substantive diagnostic completion requires consulting all four data-source families.

Large responses are exposed to the model as explicitly partial context views with
omitted-item counts. Their complete outputs remain in the sealed raw trace. Older
tool messages may be compacted further; the current evidence catalog is injected on
every model round. Omitted events must not be interpreted as absent events.

The host can validate structure and provenance, not prove physical causality or the
truth of every model sentence. Operator review and independent measurements remain
essential. If the model cannot satisfy the evidence protocol, context/output limits
are hit, inference fails, or reports fail to render, the command fails explicitly
instead of claiming successful diagnosis. Partial traces are retained.

Every turn attempts finalization in `finally`. A completed run contains:

- `trace.jsonl`, raw tool outputs and sealed `session_meta.json`;
- existing `summary.md` and `report.md` with the trace hash;
- `conversation.json` with model messages, endpoint, status and failure details;
- `answer.md` after successful completion, including clickable audit links.

Conversation/answer files are convenience artifacts, not part of the sealed evidence
hash. The runtime verifies every returned plot/report path before reporting success
and appends exact, URL-encoded file links independently of the model's prose. Missing
generated files fail explicitly. Reports and plots are not opened automatically.
Do not edit a sealed trace.
`/audit-trail` re-renders a finalized trace without running a model.
New finalized metadata distinguishes `complete`, `failed` and `interrupted` runs;
reports from failed/interrupted runs display a prominent incomplete-investigation
banner. Ollama model identity is included in audit provenance.

## Setup and index maintenance

`python setup_local.py` creates the project environment when absent, installs
`requirements.txt`, checks the local services, and checks/builds the index.
`--skip-index` skips the index operation; `--rebuild-index` explicitly recreates it.
No Claude installation, account, generated Claude configuration or external embedding
download is needed by the runtime. Existing ML dependencies are retained for analytics.
The supplied static data is not a live plant connection.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s .\tests -v
```

Offline tests cover model protocol failures, tool validation, evidence/citation guards,
private endpoint restrictions, embedding contracts, audit closure on success/failure,
snapshot bounds, temperature warnings and plot settings.
The MCP integration test checks historian, alarms, graph, document listing/retrieval,
plot generation, error signaling and sealed reports against the shipped databases.
It does not require an Ollama model call. Live RAG, analytic pipelines and LLM quality
are separate integration checks; their availability is not implied by offline tests.

Migration validation additionally exercised BGE-M3 RAG retrieval through MCP and both
analytic tools over the full dataset at one-minute resolution (7,200 observations).
MSPC retained eight components with cumulative explained variance 0.8539; both
pipelines produced all reported plots and fault reports. A Windows MCP-specific hang
was fixed by assigning analytic subprocesses `stdin=DEVNULL` rather than inheriting
the MCP protocol stream. Timeout and cancellation tests verify process-tree cleanup,
including the Windows virtual-environment interpreter redirector.

The regression suite currently contains 40 passing tests. A live Spanish statistics
query and a resumed follow-up both completed through Qwen 3.5 35B, re-queried the
requested snapshot interval and generated sealed reports. Syntax compilation,
`git diff --check` and `pip check` also passed. No repository commits were created,
and existing user backup/inspection scripts were not deleted or overwritten.

Live model tests also exposed invalid evidence references, context exhaustion and a
Qwen 3.8 27B timeout at 65,536 context with native thinking. These are not treated as
successful diagnoses. The runtime rejects invalid references, exposes a current
evidence catalog and preserves failed traces. A synthetic protocol pass alone does
not establish that an alternative model can complete an industrial investigation.

The final JSON profile (`qwen3.5:35b`, native thinking disabled, 32,768 context)
completed an audited four-source diagnostic run, sealed both reports and explicitly
returned low confidence (0.3) rather than claiming a proven physical cause. This checks
orchestration and closure, not the correctness of every process interpretation.
English and Spanish outcome labels are accepted; structured outcome fields avoid
depending on translated words for protocol correctness.

Concurrent CLI inference attempts also produced connection resets while Ollama's
HTTP service remained reachable and its runner unloaded. The client does not restart
the shared server, modify GPU settings or hide these failures. Use sequential
inference sessions until server-side parallelism and memory capacity are validated.

Both slash commands subsequently completed sequentially through the final CLI,
including deterministic cross-references, generated-file verification and sealed audit
links. The full-dataset ML run reported 153 fault periods and a 9.32% isolation anomaly
rate; MSPC reported 234 anomalous periods. These algorithm outputs are reproducible
pipeline results, not proof of the model's physical interpretation.

### Remaining interpretation limits

Live summaries still occasionally overstated causality, inferred advance warning from
feature importance, or described topology too confidently despite explicit instructions.
The host labels routine analytic interpretation as unvalidated and appends independently
verified file links; it does not pretend to semantically certify every generated sentence.
Evidence-reference validation proves that cited evidence exists, not that every claim is
entailed by it. Hypothesis and cause-versus-symptom checks enforce the recorded workflow,
not the truth of a physical diagnosis. Engineering review and event-specific confirming
measurements remain necessary before acting on model recommendations.

No controlled, ground-truth industrial accuracy comparison across all installed models,
server concurrency certification, full clean-machine bootstrap, or legacy index rebuild
was performed. The runtime no longer requires Claude Code, but this migration is not
production safety certification of an LLM.

## Organized package and latest real-model findings

The CLI remains `local_agent.py`; its implementation now lives in `local_runtime/`.
Shared client/analysis modules moved with it, docs live under `docs/`, and old manual
scripts/backups are isolated under `archive/migration_experiments/`. All active imports
and project-root resolution were updated without changing the user-facing command.

Seven chat models were compared under one synthetic JSON profile; see
[the evaluation report](MODEL_EVALUATION.md). Qwen 3.8 also completed a real statistics
plus topology request. Gemma 12B did not satisfy the real graph/citation contract,
despite passing the synthetic cases. Three identical rejected answers now terminate
explicitly rather than looping until the maximum round count.

The real checks exposed legacy kPa labels for `PTCA_8322A` and `PTCA_8324` in both the
historian metadata and graph. The historian read layer and analytical metadata loaders
now use their known stored measurement unit, MPa. Numeric values and databases are
unchanged. Graph source labels remain intact and generate an explicit host warning
when they conflict with the confirmed measurement units.
