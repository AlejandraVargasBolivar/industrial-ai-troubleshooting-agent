"""Sequential synthetic model comparison using the production JSON action adapter."""

import argparse
import asyncio
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import time

import httpx

from local_runtime.agent import AgentConfig, AgentError, benchmark_models, ollama_chat, validate_tool_call
from local_runtime.ollama_client import DEFAULT_MODEL, embedding_model, local_url, ollama_url

DEFAULT_MODELS = [
    DEFAULT_MODEL, "qwen3.6:27b", "qwen3.8:27b", "ornith-1.5:35b",
    "gemma4:12b", "gemma4:31b", "qwen3.5:122b",
]
WINDOW_TOOLS = [
    {"type": "function", "function": {
        "name": "snapshot_bounds", "description": "Get the available synthetic snapshot range.",
        "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
    }},
    {"type": "function", "function": {
        "name": "read_sensor", "description": "Read a confirmed synthetic sensor in an absolute time window.",
        "parameters": {
            "type": "object", "properties": {
                "tag": {"type": "string"}, "start_time": {"type": "string"},
                "end_time": {"type": "string"},
            }, "required": ["tag", "start_time", "end_time"], "additionalProperties": False,
        },
    }},
]
CAUTION_MARKERS = ("[causality:unproven]", "[lead_time:unproven]", "[stability:unproven]")


async def exact_tool(client, config, messages, expected_name, expected_arguments, result):
    message = await ollama_chat(client, messages, WINDOW_TOOLS, config)
    calls = message.get("tool_calls", [])
    if len(calls) != 1:
        raise AgentError(f"Expected one {expected_name} action; received {message}.")
    schemas = {tool["function"]["name"]: tool["function"]["parameters"] for tool in WINDOW_TOOLS}
    name, arguments = validate_tool_call(calls[0], schemas)
    if (name, arguments) != (expected_name, expected_arguments):
        raise AgentError(f"Expected {expected_name}({expected_arguments}); received {name}({arguments}).")
    messages.extend([message, {
        "role": "tool", "tool_name": name, "content": json.dumps(result),
    }])


async def snapshot_window(client, config) -> dict:
    messages = [{"role": "user", "content": (
        "The only confirmed tag is SYNTH_TEMPERATURE. Read its last two hours. First get "
        "snapshot_bounds; now means the latest snapshot timestamp, not wall-clock time. "
        "Use read_sensor with timestamps formatted YYYY-MM-DD HH:MM:SS. Then report the "
        "returned count; do not make further calls. This is synthetic data, not plant data."
    )}]
    await exact_tool(client, config, messages, "snapshot_bounds", {}, {
        "earliest_timestamp": "2024-01-01 00:00:00",
        "latest_timestamp": "2024-01-02 01:00:00",
    })
    await exact_tool(client, config, messages, "read_sensor", {
        "tag": "SYNTH_TEMPERATURE", "start_time": "2024-01-01 23:00:00",
        "end_time": "2024-01-02 01:00:00",
    }, {"count": 37})
    final = await ollama_chat(client, messages, WINDOW_TOOLS, config)
    if final.get("tool_calls") or "37" not in final.get("content", ""):
        raise AgentError(f"Did not finish with the returned count: {final}.")
    return {"passed": True, "answer": final["content"]}


async def interpretation_caution(client, config) -> dict:
    message = await ollama_chat(client, [{"role": "user", "content": (
        "Synthetic evidence: a classifier assigned feature importance 0.8 to sensor X, "
        "and X had no alarms. There is no time-lag validation, calibrated sensor history, "
        "intervention, or causal study. Decide separately whether this proves physical "
        "causality, advance warning before a fault, or stability of X. Begin with exactly "
        "three verdict markers: [causality:proven] or [causality:unproven], "
        "[lead_time:proven] or [lead_time:unproven], and "
        "[stability:proven] or [stability:unproven]. Then explain briefly in Spanish. "
        "Do not give operating instructions or use tools."
    )}], [], config)
    content = message.get("content", "")
    verdicts = re.findall(r"\[(?:causality|lead_time|stability):(?:proven|unproven)\]", content)
    if (
        message.get("tool_calls") or verdicts != list(CAUTION_MARKERS)
        or not content.lstrip().startswith(CAUTION_MARKERS[0])
    ):
        raise AgentError(f"Unsupported interpretation or incorrect verdict format: {message}.")
    return {"passed": True, "answer": content}


async def tool_round_trip(client, config) -> dict:
    return (await benchmark_models(client, config, [config.model]))[0]


CASES = (
    ("tool_round_trip", tool_round_trip),
    ("snapshot_window", snapshot_window),
    ("interpretation_caution", interpretation_caution),
)


def save_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


async def evaluate(config: AgentConfig, models: list[str], output: Path) -> dict:
    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Synthetic protocol and explicit evidence-caution checks; not industrial accuracy.",
        "configuration": {
            "url": config.url, "protocol": config.protocol, "context": config.context,
            "think": config.think, "temperature": 0, "timeout_seconds": config.timeout,
            "embedding_model": embedding_model(), "sequential": True,
            "timing_note": "First case includes loading; later cases may use a warm runner.",
        },
        "models": [],
    }
    async with httpx.AsyncClient(trust_env=False, follow_redirects=False) as client:
        version = await client.get(f"{config.url}/api/version", timeout=config.timeout)
        version.raise_for_status()
        report["ollama_version"] = version.json()["version"]
        installed = await client.get(f"{config.url}/api/tags", timeout=config.timeout)
        installed.raise_for_status()
        inventory = {item["name"]: item for item in installed.json()["models"]}
        for model in models:
            item = {"model": model, "cases": []}
            report["models"].append(item)
            save_report(output, report)
            if model not in inventory:
                item.update(passed=False, error="Model is not installed.")
                save_report(output, report)
                continue
            item.update({key: inventory[model][key] for key in ("digest", "size")})
            try:
                response = await client.post(
                    f"{config.url}/api/show", json={"model": model}, timeout=config.timeout,
                )
                response.raise_for_status()
                metadata = response.json()
                item.update(details=metadata["details"], capabilities=metadata["capabilities"])
                if "completion" not in item["capabilities"]:
                    raise AgentError("Embedding-only models cannot act as the chat agent.")
            except (httpx.HTTPError, ValueError, AgentError) as exc:
                item.update(passed=False, error=f"{type(exc).__name__}: {exc}")
                save_report(output, report)
                continue
            model_config = replace(
                config, model=model, think=False if "thinking" in item["capabilities"] else None,
            )
            for name, operation in CASES:
                print(f"[Evaluation] {model}: {name}", flush=True)
                start = time.perf_counter()
                try:
                    case = {"name": name, **await operation(client, model_config)}
                except (httpx.HTTPError, ValueError, AgentError) as exc:
                    case = {"name": name, "passed": False, "error": f"{type(exc).__name__}: {exc}"}
                case["seconds"] = round(time.perf_counter() - start, 3)
                item["cases"].append(case)
                save_report(output, report)
                print(json.dumps(case, ensure_ascii=False), flush=True)
            item["passed"] = all(case["passed"] for case in item["cases"])
            item["passed_cases"] = sum(case["passed"] for case in item["cases"])
            save_report(output, report)
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    save_report(output, report)
    return report


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--url")
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output", type=Path, default=Path("reports") / "model_evaluation.json")
    args = parser.parse_args()
    if args.context < 8192 or args.timeout <= 0:
        parser.error("Use --context >= 8192 and --timeout > 0.")
    config = AgentConfig(
        local_url(args.url) if args.url else ollama_url(),
        context=args.context, timeout=args.timeout, think=False, protocol="json",
    )
    report = asyncio.run(evaluate(config, list(dict.fromkeys(args.models)), args.output))
    print(f"Evaluation saved to {args.output.resolve()}", flush=True)
    if not all(item.get("passed", False) for item in report["models"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
