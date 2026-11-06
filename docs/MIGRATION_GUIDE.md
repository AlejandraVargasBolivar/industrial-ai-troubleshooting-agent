# Claude Code Replacement: Architecture, Inventory and Usage

## 1. What has actually been replaced?

The local Python application replaces the **Claude Code agent runtime**: conversation,
model requests, tool selection, validation, audit lifecycle and report delivery.
It does not replace the boiler's existing query engines or invent a new dataset.

No Claude executable, Anthropic account, Claude subscription or external MCP registration
is required. The retained `CLAUDE.md` and `.claude/commands/` files are plain-text domain
references, not a running Claude dependency. Do not remove them: the local application
still loads their applicable operating rules and analytical workflows.

The application runs on the computer holding this repository. Inference and embeddings
run on the private Ollama server, which is currently `http://172.19.120.10:11434`.
"Local" therefore means local/private infrastructure, not necessarily one computer.
There is no separately configured NVIDIA NIM endpoint. GPU execution is managed by
Ollama; the original statistical scripts still run in the project's Python environment.

## 2. Current layout and ownership

Run commands from the **repository root**, not from inside these subdirectories.

```text
local_agent.py                      Stable user-facing CLI launcher
setup_local.py                      Environment/index bootstrap
local_runtime/
    __init__.py
    agent.py                        Conversation, JSON actions, evidence and audits
    ollama_client.py                 Private endpoint and shared embedding contract
    analysis_tools.py                Allowlisted ML/MSPC process adapter and follow-ups
scripts/
    __init__.py
    evaluate_models.py              Repeatable sequential model comparison
tests/
    test_local_agent.py              Runtime, evidence and JSON contracts
    test_analysis_tools.py           Analytics, timeout and process cleanup
    test_audit_status.py             Audit completion/failure handling
    test_industrial_integration.py   Real databases and MCP, without LLM inference
    test_model_evaluation.py         Evaluation scoring and temporal checks
docs/
    MIGRATION_GUIDE.md               This complete explanatory guide
    LOCAL_AGENT.md                   Detailed operating contract and migration findings
    MODEL_EVALUATION.md              Measured model results and selection limits
reports/
    model_evaluation.json           Machine-readable synthetic evaluation results
archive/migration_experiments/
    README.md                       Historical-file warning
    scripts/                        Five old manual inspection/prototype scripts
    backups/                        Three original backup files
plots/                              Generated plots and analytic reports
traces/                             Generated audit evidence and conversations
```

### New or replaced application files

| File | Origin and purpose |
|---|---|
| [local_agent.py](../local_agent.py) | Replaces the incomplete prototype's entry point. Now deliberately small; importing the package does not start the agent. |
| [local_runtime/agent.py](../local_runtime/agent.py) | The migrated controller, formerly the large root-level local agent. Owns the CLI implementation, conversation, tool loop, validation and finalization. |
| [local_runtime/ollama_client.py](../local_runtime/ollama_client.py) | New shared client/configuration helpers, moved into the runtime package. Used by both index building and document search. |
| [local_runtime/analysis_tools.py](../local_runtime/analysis_tools.py) | New adapter, moved into the package. Runs only the two existing analytical scripts; defines their schemas and deterministic follow-up queries. |
| [setup_local.py](../setup_local.py) | New independent bootstrap; creates the virtual environment, installs declared dependencies and checks local services/index. |
| [scripts/evaluate_models.py](../scripts/evaluate_models.py) | New supported evaluation command. Uses the production JSON adapter with synthetic data. |
| [tests/](../tests/) | New automated coverage; not production runtime code. |
| [docs/](./) | Supported operating and migration documentation. |

### Existing files modified, rather than replaced

| File | Change |
|---|---|
| [historian_mcp_server.py](../historian_mcp_server.py) | Added analytical tool routing, structured evidence-step transport and explicit failure signaling. The server remains the central tool interface. |
| [audit_trace.py](../audit_trace.py) | Added completion/failure status, local model provenance and analytical result summaries; exposed rendering failures. |
| [render_audit.py](../render_audit.py) | Added incomplete-investigation banners and analytical tool labels. |
| [rag_tools.py](../rag_tools.py) | Integrated shared BGE-M3 embeddings, index checks and exact metadata filtering while retaining the user's prior RAG migration. |
| [build_docs_db.py](../build_docs_db.py) | Integrated shared embeddings and provenance metadata for new/rebuilt indexes. |
| [requirements.txt](../requirements.txt) | Declared the MCP/HTTP/schema/process dependencies used by the local runtime. |
| [README.md](../README.md) | Changed the supported startup path to the local agent and linked this documentation. |
| [how_it_works.md](../how_it_works.md) | Marked its Claude-specific architecture as historical and linked the current guide. |
| [tag_metadata.py](../tag_metadata.py) and [historian_tools.py](../historian_tools.py) | Corrected the legacy labels of the two main steam-pressure tags to match stored MPa readings, without changing numbers or databases. |
| [ml_fault_detection.py](../ml_fault_detection.py) and [mspc_analysis.py](../mspc_analysis.py) | Reused that unit-label correction in metadata loading only; statistical algorithms remain unchanged. |

### Existing industrial implementation preserved

The historian and alarm SQL queries, graph traversal algorithms, statistical algorithms,
databases and plant documents were not rewritten. The two steam-pressure unit labels
were corrected at the read/metadata layer; no measurements were multiplied or divided.
In particular, the active originals remain:

- [historian_tools.py](../historian_tools.py), [alarm_tools.py](../alarm_tools.py);
- [kg_tools.py](../kg_tools.py), [knowledge_graph.py](../knowledge_graph.py);
- [ml_fault_detection.py](../ml_fault_detection.py), [mspc_analysis.py](../mspc_analysis.py);
- the shipped DuckDB databases, topology JSON and plant document library;
- [CLAUDE.md](../CLAUDE.md) and its original command references.

The archived files are not newly invented production components. They were earlier
experiments or backups already present in the worktree:

- `scripts/inspect_audit.py`, `scripts/inspect_local_agent.py`;
- `scripts/test_mcp_connection.py`, `scripts/close_audit_test.py`;
- `scripts/patch_local_agent.py`;
- `backups/build_docs_db.py.bak`, `backups/rag_tools.py.bak`;
- `backups/local_agent.py.backup`.

All eight were moved without editing their contents. The old patch script is not a
supported updater, and the old scripts assume the former layout. Do not run them
against the reorganized application. Nothing was deleted and no commit was created.

## 3. Architecture: who does what?

```text
User question
    |
local_agent.py -> local_runtime.agent
    |-- HTTP messages/actions -> private Ollama chat model
    |-- validates schema, tags, timestamps, evidence and diagnostic phase
    |
    +-- MCP stdio -> historian_mcp_server.py
                       |-- historian_tools -> historian DuckDB
                       |-- alarm_tools     -> alarm DuckDB
                       |-- kg_tools        -> NetworkX process topology
                       |-- rag_tools       -> Chroma + private BGE-M3 embeddings
                       |-- analysis adapter -> existing ML/MSPC scripts
                       +-- audit_trace     -> sealed evidence and rendered reports
```

The **LLM interprets language and proposes actions**. It does not get unrestricted
Python, PowerShell, filesystem access or a raw SQL console. The **host application**
decides which actions are currently allowed and validates them. The **MCP server**
executes the corresponding predefined function. MCP is the tool communication protocol,
not another model and not an additional reasoning engine.

The server has 25 tools: 20 original data tools, two analytical tools and three audit
tools. The host owns session start/end and never delegates those lifecycle operations
to the model. The exposed tool subset changes with the diagnostic phase.

### A normal question, step by step

1. The host starts the MCP server with the project virtual-environment interpreter.
2. It creates a new audit, obtains the snapshot bounds and confirms available tags.
3. It combines domain instructions, applicable workflows, conversational context and
   currently permitted tool schemas.
4. Ollama returns one schema-constrained JSON action: a tool request or a final answer.
5. The host validates the request. A data call is executed through MCP and recorded
   with its actual input, complete output and evidence-step number.
6. A bounded view of the result goes to the model. Large results are explicitly marked
   partial; complete raw evidence stays in the audit.
7. The loop continues until the answer passes the applicable completion checks, or an
   explicit timeout, error or maximum-rounds limit ends the run.
8. The host verifies generated files, finalizes the audit, renders reports and appends
   exact artifact links. It saves the conversation and successful answer.

The production default is JSON actions rather than native Ollama function calling.
This made required fields and diagnostic phases enforceable instead of merely asking
the model to remember them. `--protocol native` remains an experimental alternative.

## 4. The four kinds of evidence

| Source | Answers | Does not establish by itself |
|---|---|---|
| Historian | What changed, when, by how much, and the measured trend/statistics. | Why it changed. |
| Alarm log | Which thresholds/lifecycles were active around an event. | A physical root cause, or stability merely because no alarm appeared. |
| Knowledge graph | Which equipment and sensors connect physically or through defined relationships. | Current measurements, time order, failure propagation or causality. |
| Document search | Which documented procedure/specification may apply, with title and revision. | That the documented failure mode actually occurred in this event. |

The supplied historian is a **static five-day snapshot in March/April 2022**, not a
live feed. "Last hour" is anchored to its latest available timestamp, not today's
calendar date. A report folder's creation timestamp is unrelated to the historian's
process time.

## 5. The graph, in detail

### It is a process knowledge graph, not LangGraph

[boiler_kg.json](../boiler_kg.json) stores topology. [knowledge_graph.py](../knowledge_graph.py)
loads it into a NetworkX **directed multigraph**: connections have direction, and
multiple distinct relationships may connect the same nodes.
[kg_tools.py](../kg_tools.py) lazily loads the graph once per MCP server process and
exposes its queries. There is no Neo4j service, LangGraph orchestration framework or
new graph-training pipeline.

Nodes represent equipment, sensors, systems, control loops and process boundaries.
Relationships include physical `FLOW`, membership `PART_OF`, `SYMMETRIC_PAIR`,
control feedback/actuation and KPI relationships.

Examples:

```text
TE_8332A --PART_OF--> steam_outlet
TV_8329ZC --PART_OF--> primary_desuperheater
YJJWSLL   --PART_OF--> primary_desuperheater

steam_drum
    --FLOW(Steam)--> low_temp_superheater
    --FLOW(Steam)--> primary_desuperheater
    --FLOW(Steam)--> high_temp_superheater
    --FLOW(Steam)--> steam_outlet
```

The process chain above was verified through the real MCP graph tools. Sensor
membership and flow direction must not be confused: the spray-water sensor is on
the desuperheater, which is downstream of the drum along the steam path.

### The six graph tools

| Tool | Purpose and implementation |
|---|---|
| `kg_trace_stream` | Returns a predefined ordered stream path, including parallel branches and sensors at each stop. Streams include Steam, FeedWater, FlueGas, air streams, Coal and Solids. |
| `kg_query_equipment` | Resolves equipment names/aliases and returns attached sensors, streams and neighboring equipment. |
| `kg_get_upstream_sensors` | Maps a sensor to its equipment, reverses physical FLOW connections and performs breadth-first traversal. Returns unique upstream sensors, closest equipment first, grouped by hop count. |
| `kg_find_process_path` | Finds a shortest path using FLOW edges rather than unrelated metadata links. It can fall back to an upstream traversal when no forward path exists. |
| `kg_get_related_sensors` | Finds co-located sensors, symmetric counterparts, same-system sensors and available control-loop context. |
| `kg_get_system_sensors` | Groups equipment and sensors within a named process system. |

For `TE_8332A`, the real upstream query returned **24 sensors**. Near the outlet it
returned the desuperheater's `TV_8329ZC` and `YJJWSLL` at two equipment-flow hops, then
drum pressure `PTCA_8322A` at four hops. These are topological distances, not seconds
of delay or measures of causal strength.

The fan lookup resolved `induced_draft_fan` and its sensors `YFJ3_AI`, `YFJ3_ZD1` and
`YFJ3_ZD2`. The path from steam drum to the outlet temperature sensor resolved the
same five-equipment chain shown above. All six graph tool surfaces were checked
through the reorganized MCP server.

### Important graph limits

- Legacy pressure labels in both the graph and shipped historian metadata were kPa,
  while the two main steam-pressure measurements are stored in MPa. The read layer now
  reports the known stored units, and the host surfaces graph/measurement unit conflicts.
  Graph source labels are preserved rather than silently rewriting topology or evidence.
- The graph is static and contains no live readings. Query the historian for values.
- Upstream lookup excludes sensors on the starting equipment; use related sensors too.
- The default upstream traversal combines physical-flow streams, with a 15-hop bound.
  Returned sensors can therefore span several process subsystems.
- Name matching can select a first match. Resolve ambiguous names explicitly.
- A reverse path is not a claim that the physical flow direction reversed.
- Connectivity is a candidate search map, not proof of failure or causality.
- Document RAG and graph queries are complementary. This is not a claim that a
  graph-based semantic retrieval or learned causal-discovery system was added.

## 6. Analytics and audited diagnosis are different

### `/ml-fault-detection`

The host runs the existing Isolation Forest script over the historian, using its
normal-operation training subset. The existing XGBoost stage ranks features associated
with the target temperature classification. It generates monitoring, importance and
fault-report artifacts.

Without relying on the model to remember the workflow, the host then retrieves alarm
context and active alarms for the five worst fault periods, upstream graph evidence
and aggregate alarms for the three top features, and a relevant troubleshooting guide.
It also queries actual outlet-temperature statistics for the out-of-range warning.

Feature importance is **not** a demonstrated advance-warning horizon or physical
cause. Those require temporal validation and engineering evidence.

### `/mspc-analysis`

The host runs the existing PCA monitoring script, which computes T-squared and SPE,
control limits, anomalous periods and contribution plots. It then retrieves alarm
context for the top three events of each statistic and upstream graph context for
their leading contributors. Repeated requests are deduplicated without discarding
different contributors at the same timestamp.

These are statistical event summaries, not automatically completed physical
root-cause investigations.

### Diagnostic questions

Diagnostic intent, or `--diagnostic`, activates stricter sequencing:
hypothesis before a query; observation/rejection afterward; actual evidence references;
resolution of named hypotheses; a cause-versus-symptom check; source-family coverage;
confidence-bearing conclusion; and document-grounded corrective actions when justified.
Insufficient or contradictory evidence must not become a fabricated successful cause.

The host checks these recorded contracts. It cannot prove that every natural-language
interpretation is correct or that a mentioned engineering check was physically done.

## 7. How to use the application

The current environment is already installed. No repeated setup is needed merely to
start the application.

```powershell
.\.venv\Scripts\python.exe .\local_agent.py --check
.\.venv\Scripts\python.exe .\local_agent.py
```

Inside the interactive session, ask normal questions or use:

| Input | Effect |
|---|---|
| `/mspc-analysis` | Full statistical monitoring pipeline and cross-references. |
| `/ml-fault-detection` | Full ML pipeline and cross-references. |
| `/audit-trail` | Re-render the most recent finalized audit without model inference. |
| `/audit-trail RUN_ID` | Re-render a specific finalized audit. |
| `/new` | Clear conversational context. |
| `/exit` | Exit cleanly. |

One-shot requests and configuration:

```powershell
.\.venv\Scripts\python.exe .\local_agent.py "Trace the Steam process path."
.\.venv\Scripts\python.exe .\local_agent.py --diagnostic "Investigate the steam temperature drop on March 29."
.\.venv\Scripts\python.exe .\local_agent.py "/mspc-analysis"
.\.venv\Scripts\python.exe .\local_agent.py --model qwen3.8:27b "Summarize steam temperature in the last hour."
.\.venv\Scripts\python.exe .\local_agent.py --resume RUN_ID
.\.venv\Scripts\python.exe -m scripts.evaluate_models
```

`RUN_ID` is the directory name of a completed run under `traces`, not a literal input.
Resume restores conversational history, but evidence is queried afresh for each turn.

See [Local Agent Operations](LOCAL_AGENT.md) for all configuration flags and variables.
Change the chat model with `--model` or `OLLAMA_MODEL`; that does not change BGE-M3.
Changing `OLLAMA_EMBED_MODEL` is a different operation and requires a compatible
document index, normally an explicit rebuild.

For a new installation or a moved/broken virtual environment, use `python setup_local.py`.
It creates/uses `.venv`, installs declared dependencies and checks the services/index.
It does not install Ollama models. `--rebuild-index` is explicit maintenance, not a
routine startup step; it was not run during this migration.

## 8. Where results go

Under `traces/<RUN_ID>/`:

- `trace.jsonl` and complete tool-output evidence;
- `session_meta.json`, including status, model provenance and evidence hash;
- `summary.md` and `report.md`;
- `conversation.json`, including status/history and model messages;
- `answer.md` after successful completion.

Under `plots/`, analytical runs have their own output folders. The host checks returned
paths and appends exact, URL-encoded links. It does not open operating-system viewers.

The SHA-256 protects the recorded evidence against undetected edits relative to the
stored digest; it does not certify the physical conclusion, protect against an attacker
replacing both evidence and digest, or make convenience conversation files sealed.
Failed/interrupted investigations remain identifiable; do not delete or edit sealed
traces merely to make the output directory look cleaner.

## 9. Problems fixed during migration

| Problem | Resolution |
|---|---|
| One-question prototype with hard-coded backend and short loop. | Configurable CLI, interactive history, resume and explicit round/time limits. |
| Native tool calls skipped diagnostic phases or invented references. | JSON action schemas, phase-specific tools and authoritative evidence catalog. |
| MCP analytic subprocesses blocked on inherited protocol input. | Isolated stdin, bounded execution, cancellation and process-tree cleanup. |
| Error-shaped results could look successful. | MCP error signaling, explicit host errors and failed audit status. |
| BGE-M3 queries still used the old BGE-small prefix. | Shared index/query embedding implementation and empty BGE-M3 prefix. |
| Index compatibility was implicit. | Model/dimension checks and provenance metadata for newly built indexes. |
| Large payloads exhausted model context. | Explicitly partial model-facing views, while retaining complete raw audit evidence. |
| Models omitted some analytical cross-references. | Mandatory follow-ups executed by the host rather than remembered by the LLM. |
| Models invented artifact URLs or claimed plots opened. | Host-verified artifact links; no automatic viewers. |
| An alternative model repeated the same rejected answer. | Explicit failure after three identical incomplete answers, without fabricated success. |
| Legacy pressure-unit labels disagreed with the stored numeric values. | Shared measurement-unit correction and explicit graph/unit warnings; no numeric rescaling or database mutation. |
| Manual stdio inputs and separate-process audit closing caused confusion. | One host-managed MCP/audit connection and supported report commands. |

## 10. Readiness: honest boundary

**Ready for controlled interactive use with the supplied dataset:** querying all four
sources, graph traversal, document retrieval, analytical pipelines, conversation,
audit reporting and model selection through the CLI.

**Not certified for unattended industrial operation:** live plant integration,
automatic DCS writes/control, guaranteed physical diagnosis, exhaustive multilingual
intent detection, server concurrency, or a full clean-machine installation.

The legacy document index retrieves relevant results but lacks recorded embedding-model
provenance. A matching vector dimension alone cannot certify its original model.
Rebuild deliberately before relying on provenance.

Model summaries can still overstate causality, lead time or topology. Source citations
and formal workflow checks are necessary, not sufficient, for engineering correctness.
Keep human review and event-specific confirming measurements before operational action.

After reorganization, all six graph tool surfaces and real BGE-M3 retrieval were checked.
Both full analytical scripts again processed 7,200 observations and produced 15 verified
files in total. The automated suite contains 40 passing tests, including pressure units,
repeated-answer termination, evaluation scoring and the relocated runtime's contracts.

See [the measured model comparison](MODEL_EVALUATION.md) and [operational details](LOCAL_AGENT.md).
