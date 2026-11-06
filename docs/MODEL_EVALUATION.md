# Local Model Evaluation

## What was measured

On 2026-10-28, all seven installed chat models were evaluated sequentially through
the **same production JSON action adapter**. BGE-M3 was excluded from chat selection:
the server advertises it as an embedding-only model.

Conditions:

- Ollama `0.33.3`, private endpoint `http://172.19.120.10:11434`;
- JSON action protocol, temperature zero, 32,768 requested context, thinking disabled;
- three synthetic cases per model, no plant data;
- identical prompts and validators, no parallel inference and no automatic retries;
- one recorded final run per model, not repeated statistical sampling;
- model digests, actual server metadata, raw answers/errors and durations preserved in
  [the machine-readable report](../reports/model_evaluation.json).

Cases:

1. **Tool round trip:** request exactly `lookup_sensor(SYNTHETIC_42)`, then report the
   returned value `17.25` rather than guessing it.
2. **Snapshot window:** first request snapshot bounds, calculate the last two hours
   across midnight, request the exact confirmed tag/window and report the returned count.
3. **Interpretation caution:** distinguish feature importance and absence of alarms
   from demonstrated physical causality, advance warning and sensor stability. The
   three verdicts are checked; the Spanish explanation is retained for human review.

The evaluator originally rejected equivalent newline/comma separators between correct
verdict markers. That scoring defect was corrected and regression-tested, and the
entire seven-model evaluation was rerun. The published report uses the corrected
whitespace-independent scoring, not the initial false formatting failures.

## Results

Times are wall-clock seconds per case. The first case includes potential model
loading; later cases can use a warm runner. They are **not tokens/second**, VRAM
measurements, or controlled latency rankings.

| Model | Tool round trip | Exact snapshot window | Caution verdicts | Passed | Model storage (GiB) |
|---|---:|---:|---:|---:|---:|
| `qwen3.5:35b` | Pass, 12.323 s | Pass, 3.012 s | Pass, 1.660 s | 3/3 | 22.23 |
| `qwen3.6:27b` | Pass, 12.144 s | Pass, 5.342 s | Pass, 6.167 s | 3/3 | 16.55 |
| `qwen3.8:27b` | Pass, 11.531 s | Pass, 4.272 s | Pass, 7.570 s | 3/3 | 16.52 |
| `ornith-1.5:35b` | Pass, 11.202 s | Fail, 5.940 s | Pass, 2.808 s | 2/3 | 21.06 |
| `gemma4:12b` | Pass, 9.094 s | Pass, 5.805 s | Pass, 4.234 s | 3/3 | 7.04 |
| `gemma4:31b` | Pass, 16.564 s | Pass, 11.924 s | Pass, 9.145 s | 3/3 | 18.50 |
| `qwen3.5:122b` | Pass, 28.185 s | Pass, 6.177 s | Pass, 4.968 s | 3/3 | 75.78 |

This is **20 of 21 successful model/case combinations**. The command exited with
status 1 because it intentionally reports any failed case as a failed evaluation;
it did not crash or omit the remaining models.

For Ornith's window case, the response claimed the supplied tools were unavailable
and did not request `snapshot_bounds`, despite their schemas being provided. It also
mistakenly characterized the explicitly synthetic prompt as real operational data.
This is a demonstrated failure in that scenario, not a claim that Ornith cannot
ever use tools. Its other two cases passed.

All chat models advertised tool and thinking support and used `Q4_K_M` quantization.
Reported families were:

- `qwen35moe`: Qwen 3.5 35B/122B and Ornith;
- `qwen35`: the locally named Qwen 3.6/3.8 27B tags;
- `gemma4`: both Gemma models.

Local tag names alone do not prove architecture, upstream release or training quality.
The report stores the model digests and server details for reproducibility.
Model storage is not the memory needed for inference: KV cache, runtime overhead,
offloading and actual GPU capacity are additional factors.

## Selection decision

**Keep `qwen3.5:35b` as the supported default for now.** It passed these checks and
already completed the migration's real four-source audited diagnostic and both
analytical CLI workflows. A larger parameter count did not establish better
industrial conclusions, and the 122B model's first-case latency/storage cost was
higher in this observation.

`qwen3.6:27b` and `qwen3.8:27b` are plausible alternatives. The smaller `gemma4:12b`
passed the synthetic cases but failed the real combined request described below;
it is not recommended as a drop-in replacement for the complete agent contract.
All require task-specific validation before assuming equivalence to the supported default. Do not select
solely from three synthetic passes or the lowest number in the latency table.

Ornith is not recommended as the default with the current JSON contract until its
observed tool-availability refusal is understood. No model is silently substituted
when another fails.

## Additional real-data checks

The same Spanish, read-only request was sent sequentially to Gemma 12B and Qwen 3.8:
report the last snapshot hour's steam-temperature minimum/maximum/mean and query the
Steam graph path, without inferring causes or recommending actions.

| Model | Actual outcome |
|---|---|
| `gemma4:12b` | Queried historian statistics, but invented a graph description without calling the graph tool and repeatedly used unsupported citation syntax. The host refused completion and sealed a failed run after the configured 12-round limit. |
| `qwen3.8:27b` | Called both `historian_get_statistics` and `kg_trace_stream`, returned the correct five-equipment path and completed the run. It initially copied legacy pressure-unit labels, so completion is not a blanket correctness claim. |

Evidence: [Gemma failed audit](../traces/20261028_090103_e0ab0b/report.md) and
[Qwen 3.8 completed audit](../traces/20261028_090454_9ae9ad/report.md).
These links reference local validation artifacts, not files shipped with a fresh clone.

This exposed two tightly related issues that were then addressed:

1. Repeating an identical rejected answer now fails explicitly after three repetitions,
   instead of spending the entire configured round budget. It does not silently accept
   the bad answer or switch models.
2. Both the legacy historian metadata and graph labeled the two main steam pressures
   in kPa although the numeric historian readings are already MPa. The historian read
   layer and metadata loaders now report MPa without rescaling numbers or changing
   database files. Graph kPa labels remain visible as source metadata; the host flags
   the conflict and tells the model which units apply to measurements.

A subsequent default-model graph query displayed the host-generated unit warning
and distinguished both units correctly:
[post-correction audit](../traces/20261028_091547_447ba9/report.md).
Both analytical pipelines were rerun after reorganization/correction and retained
their numerical results; all 15 reported generated files were verified.

## What these results do not prove

The caution test explicitly asks for three structured verdicts. Passing it does not
prove that a long free-form industrial summary will consistently avoid causal
overstatement; live summaries have shown otherwise. Likewise, a synthetic time-window
test is not a full root-cause investigation, complete MCP workflow or process-expert
assessment.

No ground-truth industrial accuracy ranking, repeated multi-seed study, memory
capacity/concurrency certification, or exhaustive full-RCA comparison across the
seven models was performed. The successful baseline diagnosis had low confidence,
not a scientifically confirmed physical cause.

## Reproduce or select a model

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m scripts.evaluate_models
.\.venv\Scripts\python.exe -m scripts.evaluate_models --models qwen3.5:35b gemma4:12b --output .\reports\model_comparison_subset.json
.\.venv\Scripts\python.exe .\local_agent.py --model qwen3.8:27b
.\.venv\Scripts\python.exe .\local_agent.py --model gemma4:12b
```

The existing `local_agent.py --benchmark MODEL ...` is still the smaller two-turn
protocol probe. The new evaluation command covers all three cases and writes a
persistent report after each case, including explicit failures.

Changing the chat model does not replace BGE-M3 or require rebuilding its index.
