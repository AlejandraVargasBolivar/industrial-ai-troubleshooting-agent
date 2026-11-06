"""Allowlisted analytics subprocesses; never execute model-provided commands."""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
import psutil

PROJECT_DIR = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)

COMMON_PROPERTIES = {
    "downsample_minutes": {"type": "integer", "minimum": 1, "maximum": 60},
    "start_time": {"type": "string"},
    "end_time": {"type": "string"},
    "top_n_events": {"type": "integer", "minimum": 1, "maximum": 10},
    "noc_tag": {"type": "string"},
    "noc_min": {"type": "number"},
    "noc_max": {"type": "number"},
}
ANALYSIS_SCHEMAS = {
    "analysis_run_mspc": {
        "type": "object",
        "properties": {
            **COMMON_PROPERTIES,
            "cpv_threshold": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
            "n_components": {"type": "integer", "minimum": 1, "maximum": 30},
            "exclude_tags": {"type": "string"},
            "top_n_contributors": {"type": "integer", "minimum": 1, "maximum": 30},
        },
        "additionalProperties": False,
    },
    "analysis_run_fault_detection": {
        "type": "object",
        "properties": {
            **COMMON_PROPERTIES,
            "contamination": {"type": "number", "exclusiveMinimum": 0, "maximum": 0.5},
            "n_estimators": {"type": "integer", "minimum": 1, "maximum": 1000},
            "no_xgb": {"type": "boolean"},
        },
        "additionalProperties": False,
    },
}


def analysis_followup_calls(name: str, result: dict, data_range: dict) -> list[tuple[str, dict]]:
    """Translate the existing analytic command workflows into exact evidence queries."""
    calls: list[tuple[str, dict]] = []
    if "TE_8332A" in result["model"]["tags_used"]:
        calls.append(("historian_get_statistics", {
            "tag_names": ["TE_8332A"],
            "start_time": data_range["earliest_timestamp"],
            "end_time": data_range["latest_timestamp"],
        }))
    if name == "analysis_run_mspc":
        for statistic in ("T2", "SPE"):
            for event in result["worst_events"][statistic][:3]:
                timestamp = event["timestamp"]
                calls.extend([
                    ("alarm_search_context", {"timestamp": timestamp, "window_minutes": 30}),
                    ("kg_get_upstream_sensors", {"node_id": event["top_contributors"][0]["tag"]}),
                ])
    elif name == "analysis_run_fault_detection":
        for event in result["worst_events"][:5]:
            calls.extend([
                ("alarm_search_context", {"timestamp": event["start"], "window_minutes": 30}),
                ("alarm_get_active_at", {"timestamp": event["start"]}),
            ])
        features = result["feature_importance"]["top_features"][:3]
        for feature in features:
            tag = feature["tag"]
            calls.extend([
                ("kg_get_upstream_sensors", {"node_id": tag}),
                ("alarm_get_statistics", {
                    "start_time": data_range["earliest_timestamp"],
                    "end_time": data_range["latest_timestamp"], "tag_names": [tag],
                }),
            ])
        if features:
            calls.append(("docs_search", {
                "query": "Troubleshooting boiler steam temperature excursions and early-warning sensors",
                "sensor_tag": features[0]["tag"], "doc_type": "troubleshooting",
            }))
    else:
        raise ValueError(f"Unknown analytic workflow: {name}")
    unique = []
    for call in calls:
        if call not in unique:
            unique.append(call)
    return unique


def stop_process_tree(pid: int) -> None:
    """Include the Windows virtual-environment interpreter redirector's children."""
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    children = parent.children(recursive=True)
    for process in [*reversed(children), parent]:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            continue
    _, alive = psutil.wait_procs(children, timeout=5)
    if alive:
        raise RuntimeError(f"Could not stop analysis child process IDs: {[p.pid for p in alive]}")


async def run_analysis(name: str, arguments: dict, timeout: float = 600) -> dict:
    from jsonschema import validate

    validate(arguments, ANALYSIS_SCHEMAS[name])
    script = {
        "analysis_run_mspc": "mspc_analysis.py",
        "analysis_run_fault_detection": "ml_fault_detection.py",
    }[name]
    command = [sys.executable, str(PROJECT_DIR / script)]
    for key, value in arguments.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                command.append(flag)
        else:
            command.extend([flag, str(value)])
    process = await asyncio.create_subprocess_exec(
        *command, cwd=PROJECT_DIR,
        env={**os.environ, "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg"},
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    communication = asyncio.create_task(process.communicate())
    try:
        stdout, stderr = await asyncio.wait_for(asyncio.shield(communication), timeout)
    except TimeoutError as exc:
        if process.returncode is None:
            await asyncio.to_thread(stop_process_tree, process.pid)
        _, stderr = await communication
        raise RuntimeError(
            f"{script} exceeded {timeout} seconds:\n"
            f"{stderr.decode('utf-8', errors='replace')}"
        ) from exc
    except asyncio.CancelledError:
        if process.returncode is None:
            await asyncio.to_thread(stop_process_tree, process.pid)
        await communication
        raise
    progress = stderr.decode("utf-8", errors="replace")
    if process.returncode:
        raise RuntimeError(f"{script} exited with {process.returncode}:\n{progress}")
    result = json.loads(stdout.decode("utf-8"))
    if not isinstance(result, dict) or not {"model", "results", "plots"} <= result.keys():
        raise ValueError(f"{script} returned an invalid analysis summary.")
    logger.info("%s progress:\n%s", script, progress)
    return {**result, "execution_log": progress}
