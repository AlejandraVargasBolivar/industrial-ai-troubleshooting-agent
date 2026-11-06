"""Offline contract tests for the local runtime and industrial workflow guards."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from mcp import types

from local_runtime import agent
from local_runtime.ollama_client import embed_texts, local_url, validate_index


def response(payload, step=None, error=False):
    structured = {"result": payload, "audit_step": step} if step is not None else None
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(payload))],
        structuredContent=structured, isError=error,
    )


class ToolContractTests(unittest.TestCase):
    def test_graph_pressure_units_are_flagged_without_rewriting_source_evidence(self):
        sensor = {"tag_name": "PTCA_8324", "units": "kPa", "normal_min": 9400}
        result = {"equipment": [{"sensors": [sensor]}], "duplicate": sensor}
        metadata = {"PTCA_8324": {"units": "MPa"}}
        notice = agent.unit_metadata_notice("kg_trace_stream", result, metadata)
        self.assertIn("historian readings MPa", notice)
        self.assertEqual(notice.count("PTCA_8324"), 1)
        self.assertEqual(sensor["units"], "kPa")
        self.assertEqual(sensor["normal_min"], 9400)
        self.assertEqual(agent.unit_metadata_notice("historian_list_tags", result, metadata), "")
        self.assertEqual(agent.unit_metadata_notice("kg_trace_stream", result, {}), "")

    def test_artifact_links_are_verified_encoded_and_deduplicated(self):
        with tempfile.TemporaryDirectory() as directory:
            plot = Path(directory) / "event chart.png"
            plot.write_bytes(b"synthetic plot")
            state = agent.EvidenceState()
            state.steps[1] = ("historian_plot_tags", {"plot_path": str(plot)})
            state.steps[2] = ("analysis_run_mspc", {"plots": {
                "monitoring": str(plot), "contributions": [str(plot)], "optional": None,
            }})
            footer = agent.artifact_footer(state)
            self.assertEqual(footer.count("]("), 1)
            self.assertIn(plot.as_uri(), footer)
            self.assertIn("%20", footer)
            state.steps[3] = ("historian_plot_tags", {"plot_path": str(plot.with_name("missing.png"))})
            with self.assertRaisesRegex(agent.AgentError, "does not exist"):
                agent.artifact_footer(state)

    def test_structured_observation_schema_requires_exact_evidence_and_outcome(self):
        tool = SimpleNamespace(
            name="audit_log_reasoning", description="", inputSchema={
                "type": "object", "properties": {
                    "reasoning_type": {"type": "string", "enum": [
                        "hypothesis", "observation", "rejection", "conclusion", "corrective_action",
                    ]},
                    "text": {"type": "string"},
                    "evidence_steps": {"type": "array", "items": {"type": "integer"}},
                    "confidence": {"type": "number"},
                }, "required": ["reasoning_type", "text"],
            },
        )
        tools = agent.convert_mcp_tools([tool])
        state = agent.EvidenceState()
        state.reasoning.append({"reasoning_type": "hypothesis", "text": "[hypothesis:H1] Test."})
        state.needs_observation = 8
        schema = agent.action_schema(agent.diagnostic_tools(tools, state, True))
        arguments = {
            "reasoning_type": "observation", "text": "Synthetic observation.",
            "hypothesis_id": "H1", "hypothesis_status": "inconclusive", "evidence_steps": [8],
        }
        action = {"kind": "tool", "name": "audit_log_reasoning", "arguments": arguments}
        agent.validate(action, schema)
        for key in ("evidence_steps", "hypothesis_id", "hypothesis_status"):
            invalid = {**arguments}
            invalid.pop(key)
            with self.assertRaises(agent.ValidationError):
                agent.validate({**action, "arguments": invalid}, schema)
        with self.assertRaises(agent.ValidationError):
            agent.validate({**action, "arguments": {**arguments, "evidence_steps": [7]}}, schema)
        normalized = agent.normalize_reasoning(arguments)
        self.assertIn("[hypothesis:H1]", normalized["text"])
        self.assertIn("inconclusive", normalized["text"])
        self.assertNotIn("hypothesis_id", normalized)

    def test_domain_prompt_preserves_operating_rules_without_duplicate_api_tables(self):
        full = (agent.PROJECT_DIR / "CLAUDE.md").read_text(encoding="utf-8")
        runtime = agent.domain_instructions()
        self.assertIn("## Critical Rules", runtime)
        self.assertIn("## Root Cause Analysis Methodology", runtime)
        self.assertIn("### Distinguish Cause from Symptom", runtime)
        self.assertLess(len(runtime), len(full) * 0.8)
        self.assertNotIn("## Codebase Structure", runtime)

    def test_resume_restores_complete_history_and_rejects_failed_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root / "traces" / "synthetic_run"
            trace.mkdir(parents=True)
            history = [
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "first answer"},
                {"role": "user", "content": "follow-up"},
                {"role": "assistant", "content": "second answer"},
            ]
            path = trace / "conversation.json"
            path.write_text(json.dumps({"status": "complete", "history": history}), encoding="utf-8")
            with patch.object(agent, "PROJECT_DIR", root):
                self.assertEqual(agent.resume_history("synthetic_run"), history)
                path.write_text(json.dumps({"status": "failed", "history": history}), encoding="utf-8")
                with self.assertRaises(agent.AgentError):
                    agent.resume_history("synthetic_run")
                with self.assertRaises(agent.AgentError):
                    agent.resume_history("..")

    def test_explicit_diagnostic_intents_do_not_include_routine_statistics(self):
        self.assertTrue(agent.is_diagnostic("Por qué bajó la temperatura?"))
        self.assertTrue(agent.is_diagnostic("Investiga la causa raíz"))
        self.assertFalse(agent.is_diagnostic("Dime el promedio y no hagas un diagnostico causal."))

    def test_only_advertised_tools_accept_valid_arguments(self):
        schemas = {"lookup": {"type": "object", "properties": {"tag": {"type": "string"}},
                              "required": ["tag"]}}
        call = {"function": {"name": "lookup", "arguments": '{"tag":"known"}'}}
        self.assertEqual(agent.validate_tool_call(call, schemas), ("lookup", {"tag": "known"}))
        for invalid in (
            {"function": {"name": "audit_end_session", "arguments": {}}},
            {"function": {"name": "lookup", "arguments": {"tag": "known", "shell": "cmd"}}},
            {"function": {"name": "lookup", "arguments": "not-json"}},
        ):
            with self.assertRaises((ValueError, agent.ValidationError)):
                agent.validate_tool_call(invalid, schemas)

    def test_runtime_lifecycle_tools_are_not_advertised(self):
        tools = [SimpleNamespace(name=name, description="", inputSchema={})
                 for name in ("historian_list_tags", "audit_start_session", "audit_end_session")]
        self.assertEqual(len(agent.convert_mcp_tools(tools)), 1)

    def test_snapshot_bounds_confirmed_tags_and_temperature_plot(self):
        data_range = {"earliest_timestamp": "2022-03-27 14:28:54",
                      "latest_timestamp": "2022-04-01 14:28:49"}
        arguments = {"tag_names": ["TE_8332A"], "start_time": "2022-04-01 13:28:49",
                     "end_time": "2022-04-01 14:28:49", "show_normal_range": False}
        agent.guard_query("historian_plot_tags", arguments, {"TE_8332A"}, data_range)
        self.assertTrue(arguments["show_normal_range"])
        for invalid in (
            {**arguments, "tag_names": ["invented"]},
            {**arguments, "end_time": "2026-10-27 14:28:49"},
            {**arguments, "start_time": "2022-04-01 14:28:49", "end_time": "2022-04-01 13:28:49"},
        ):
            with self.assertRaises(ValueError):
                agent.guard_query("historian_get_statistics", invalid, {"TE_8332A"}, data_range)

    def test_anomalies_are_not_suppressed_by_model(self):
        notice = agent.anomaly_notice("historian_get_statistics", {
            "statistics": {"TE_8332A": {"min": 529.9, "max": 545.1}}})
        self.assertIn("outside normal", notice)
        self.assertEqual(agent.anomaly_notice("historian_get_statistics", {
            "statistics": {"TE_8332A": {"min": 530, "max": 545}}}), "")

    def test_reasoning_requires_real_evidence_and_retrieved_citations(self):
        state = agent.EvidenceState()
        state.steps[2] = ("docs_search", [{"title": "Synthetic Guide", "revision": "1.2"}])
        action = {"reasoning_type": "corrective_action",
                  "text": "Inspect the test component. Per Synthetic Guide, Rev 1.2.",
                  "evidence_steps": [2]}
        state.validate_reasoning(action)
        for invalid in (
            {**action, "evidence_steps": [999]},
            {**action, "text": "Per Invented Guide, Rev 1.2."},
            {"reasoning_type": "conclusion", "text": "Cause", "evidence_steps": [2]},
        ):
            with self.assertRaises(ValueError):
                state.validate_reasoning(invalid)

    def test_incomplete_diagnostic_is_rejected(self):
        state = agent.EvidenceState()
        state.reasoning.append({"reasoning_type": "hypothesis", "text": "[hypothesis:H1] Test."})
        reasons = " ".join(state.incomplete_reasons())
        self.assertIn("Resolve hypotheses", reasons)
        self.assertIn("cause-versus-symptom", reasons)
        self.assertIn("corrective action", reasons)

    def test_duplicate_reasoning_is_not_logged_as_new_evidence(self):
        state = agent.EvidenceState()
        entry = {"reasoning_type": "hypothesis", "text": "[hypothesis:H1] Test."}
        state.reasoning.append(entry)
        with self.assertRaisesRegex(ValueError, "already recorded"):
            state.validate_reasoning(entry)

    def test_diagnostic_tool_phases_prevent_skipping_reasoning(self):
        state = agent.EvidenceState()
        tools = [
            {"function": {"name": "historian_get_statistics", "parameters": {}}},
            {"function": {"name": "audit_log_reasoning", "parameters": {
                "properties": {"reasoning_type": {"type": "string"}},
            }}},
        ]
        self.assertEqual(agent.diagnostic_tools(tools, state, True)[0]["function"]["name"],
                         "audit_log_reasoning")
        state.reasoning.append({"reasoning_type": "hypothesis", "text": "[hypothesis:H1] Test."})
        self.assertEqual(agent.diagnostic_tools(tools, state, True)[0]["function"]["name"],
                         "historian_get_statistics")
        state.needs_observation = 8
        selected = agent.diagnostic_tools(tools, state, True)
        self.assertEqual(selected[0]["function"]["parameters"]["properties"]["reasoning_type"]["enum"],
                         ["observation", "rejection", "conclusion"])
        self.assertNotIn("enum", tools[1]["function"]["parameters"]["properties"]["reasoning_type"])

    def test_spanish_audit_outcomes_are_not_rejected_as_missing_english_words(self):
        state = agent.EvidenceState()
        state.reasoning.append({"reasoning_type": "hypothesis", "text": "[hypothesis:H1] Test."})
        state.steps[8] = ("historian_get_statistics", {})
        state.needs_observation = 8
        for outcome in ("Confirmado parcialmente", "Rechazada", "Descartada", "Inconclusa"):
            state.validate_reasoning({
                "reasoning_type": "observation", "text": f"[hypothesis:H1] {outcome}.",
                "evidence_steps": [8],
            })
        self.assertTrue(agent.is_final_conclusion("La causa raíz candidata es una condición sintética."))

    def test_grounded_reasoning_references_resolve_to_actual_data(self):
        state = agent.EvidenceState()
        state.steps[2] = ("historian_get_statistics", {})
        state.reasoning_steps.update({3, 4})
        state.reasoning_sources.update({3: [2], 4: []})
        state.validate_reasoning({
            "reasoning_type": "conclusion", "text": "Test observation",
            "evidence_steps": [3], "confidence": 0.6,
        })
        self.assertEqual(state.data_sources([3]), {2})
        with self.assertRaises(ValueError):
            state.validate_reasoning({
                "reasoning_type": "observation", "text": "Unsupported",
                "evidence_steps": [4],
            })

    def test_large_tool_results_are_explicit_partial_context_views(self):
        payload = {"audit_step": 8, "result": {"alarms": [
            {"tag": f"synthetic_{index}", "message": "x" * 100} for index in range(1000)
        ]}}
        view = json.loads(agent.context_payload(payload))
        self.assertEqual(view["context_view"]["audit_step"], 8)
        self.assertEqual(view["omitted_from_context"][0]["total"], 1000)
        self.assertIn("complete result is preserved", view["notice"])
        self.assertLess(len(json.dumps(view)), 8000)
        self.assertEqual(len(payload["result"]["alarms"]), 1000)

    def test_no_anomaly_closure_does_not_fabricate_corrective_actions(self):
        state = agent.EvidenceState()
        state.steps[2] = ("historian_get_statistics", {
            "statistics": {"TE_8332A": {"min": 530, "max": 545}}})
        state.reasoning = [
            {"reasoning_type": "hypothesis", "text": "[hypothesis:H1] Check for excursions."},
            {"reasoning_type": "conclusion", "text":
             "No anomaly was confirmed. [hypothesis:H1] rejected: no temperature excursion."},
        ]
        self.assertEqual(state.incomplete_reasons(), [])
        state.steps[2][1]["statistics"]["TE_8332A"]["min"] = 520
        self.assertIn("contradicts", " ".join(state.incomplete_reasons()))

    def test_public_model_endpoint_is_rejected(self):
        with self.assertRaises(ValueError):
            local_url("https://8.8.8.8")
        self.assertEqual(local_url("http://127.0.0.1:11434/"), "http://127.0.0.1:11434")

    def test_index_model_change_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_index(SimpleNamespace(metadata={"embedding_model": "different"}))

    @patch("local_runtime.ollama_client.httpx.Client")
    def test_embeddings_reject_invalid_count_dimensions_and_values(self, client):
        result = client.return_value.__enter__.return_value.post.return_value
        for vectors in ([], [[1], [1]], [[]], [[float("nan")]], [[True]]):
            result.json.return_value = {"embeddings": vectors}
            with self.assertRaises(ValueError):
                embed_texts(["synthetic"])


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_json_actions_are_schema_enforced_and_converted_to_mcp_calls(self):
        tools = [{"function": {
            "name": "synthetic_lookup", "description": "Synthetic lookup.",
            "parameters": {"type": "object", "properties": {"tag": {"type": "string"}},
                           "required": ["tag"]},
        }}]

        def handler(request):
            body = json.loads(request.content)
            self.assertIn("format", body)
            self.assertNotIn("tools", body)
            action = {"kind": "tool", "name": "synthetic_lookup", "arguments": {"tag": "test"}}
            return httpx.Response(200, json={
                "message": {"role": "assistant", "content": json.dumps(action)},
            })

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            result = await agent.ollama_chat(
                client, [], tools, agent.AgentConfig("http://127.0.0.1"),
            )
        self.assertEqual(result["tool_calls"][0]["function"], {
            "name": "synthetic_lookup", "arguments": {"tag": "test"},
        })

    async def test_complete_diagnostic_protocol_produces_answer_and_reports(self):
        def tool(name, arguments):
            return {"role": "assistant", "tool_calls": [
                {"function": {"name": name, "arguments": arguments}}]}

        def reason(kind, text, refs=None, confidence=None):
            arguments = {"reasoning_type": kind, "text": text}
            if refs is not None:
                arguments["evidence_steps"] = refs
            if confidence is not None:
                arguments["confidence"] = confidence
            identifier = agent.re.search(r"\[hypothesis:(H\d+)\]", text)
            if identifier and kind == "observation" and "confirmed" in text:
                arguments["hypothesis_id"] = identifier[1]
                arguments["hypothesis_status"] = "confirmed"
            return tool("audit_log_reasoning", arguments)

        scripted = [reason("hypothesis", "Unidentified hypotheses must not reach MCP.")]
        for index, (name, arguments, evidence) in enumerate((
            ("historian_get_statistics", {
                "tag_names": ["TE_8332A"], "start_time": "2022-03-29 13:30:00",
                "end_time": "2022-03-29 14:30:00"}, 8),
            ("alarm_search_context", {"timestamp": "2022-03-29 14:00:00"}, 11),
            ("kg_get_upstream_sensors", {"node_id": "TE_8332A"}, 14),
            ("docs_get_document", {"doc_id": "synthetic_guide"}, 17),
        ), 1):
            scripted.extend([
                reason("hypothesis", f"[hypothesis:H{index}] Check {name}."),
                tool(name, arguments),
                reason("observation", f"[hypothesis:H{index}] confirmed: evidence retrieved.", [evidence]),
            ])
        scripted.extend([
            reason("observation", "[cause-versus-symptom] Control response, sensor reliability "
                   "and upstream disturbances remain unverified in this synthetic test.", [8, 11, 14]),
            reason("conclusion", "The candidate root cause is a synthetic test condition. "
                   "1. Synthetic input changes the test output. Additional verification: "
                   "check the synthetic sensor independently.", [8, 11, 14, 17], 0.6),
            reason("corrective_action", "Inspect the synthetic component. Per Synthetic Guide, Rev 1.2.",
                   [8, 17]),
            {"role": "assistant", "content": "Synthetic test conclusion, not plant advice [step 8][step 17]."},
        ])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("summary.md", "report.md"):
                (root / name).write_text("synthetic report", encoding="utf-8")
            counter = 0
            closed_arguments = None

            async def call(name, arguments):
                nonlocal counter, closed_arguments
                if name == "audit_start_session":
                    return response({"trace_dir": directory})
                if name == "audit_end_session":
                    closed_arguments = arguments
                    return response({"summary": str(root / "summary.md"),
                                     "report": str(root / "report.md"), "trace_sha256": "abc"})
                counter += 1
                if name == "audit_log_reasoning":
                    if arguments["reasoning_type"] == "hypothesis":
                        self.assertRegex(arguments["text"], r"\[hypothesis:H\d+\]")
                    return response({"step": counter})
                data = {
                    "historian_get_data_range": {"earliest_timestamp": "2022-03-27 14:28:54",
                                                 "latest_timestamp": "2022-04-01 14:28:49"},
                    "historian_list_tags": [{"tag_name": "TE_8332A"}],
                    "historian_get_statistics": {"statistics": {"TE_8332A": {"min": 520, "max": 540}}},
                    "alarm_search_context": {"alarm_count": 1},
                    "kg_get_upstream_sensors": {"sensors": []},
                    "docs_get_document": {"title": "Synthetic Guide", "revision": "1.2"},
                }[name]
                return response(data, counter)

            properties = {
                "reasoning_type": {"type": "string"}, "text": {"type": "string"},
                "evidence_steps": {"type": "array", "items": {"type": "integer"}},
                "confidence": {"type": "number"}, "tag_names": {"type": "array"},
                "hypothesis_id": {"type": "string"}, "hypothesis_status": {"type": "string"},
                "start_time": {"type": "string"}, "end_time": {"type": "string"},
                "timestamp": {"type": "string"}, "node_id": {"type": "string"},
                "doc_id": {"type": "string"},
            }
            tools = [{"function": {"name": name, "parameters": {
                "type": "object", "properties": properties,
            }}} for name in ("audit_log_reasoning", "historian_get_statistics",
                             "alarm_search_context", "kg_get_upstream_sensors", "docs_get_document")]
            with patch.object(agent, "ollama_chat", AsyncMock(side_effect=scripted)):
                answer = await agent.run_turn(
                    SimpleNamespace(call_tool=call), None, tools,
                    agent.AgentConfig("http://127.0.0.1"), "Synthetic diagnostic test",
                    diagnostic=True,
                )
            self.assertIn("outside normal", answer)
            self.assertIn("Synthetic test conclusion", answer)
            self.assertEqual(closed_arguments["status"], "complete")
            self.assertTrue((root / "answer.md").is_file())

    async def test_mcp_error_flag_and_error_payload_are_both_failures(self):
        session = SimpleNamespace(call_tool=AsyncMock())
        for result in (response({"error": "failed"}), response({}, error=True)):
            session.call_tool.return_value = result
            with self.assertRaises(agent.AgentError):
                await agent.checked_call(session, "test", {})

    async def test_ollama_errors_empty_and_truncated_messages_fail(self):
        for payload in (
            {"message": {"role": "assistant", "content": ""}},
            {"message": {"role": "assistant", "content": "partial"}, "done_reason": "length"},
            {"error": "missing model"},
        ):
            async with httpx.AsyncClient(transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json=payload)
            )) as client:
                with self.assertRaises(agent.AgentError):
                    await agent.ollama_chat(client, [], [], agent.AgentConfig("http://127.0.0.1"))

    async def test_analysis_command_finishes_required_queries_before_model_interpretation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plot = root / "synthetic plot.png"
            plot.write_bytes(b"synthetic plot")
            for name in ("summary.md", "report.md"):
                (root / name).write_text("synthetic report", encoding="utf-8")
            calls = []
            timestamp = "2022-03-28 19:25:00"
            analysis = {
                "model": {"tags_used": ["TE_8332A"]}, "results": {},
                "plots": {"monitoring": str(plot)},
                "worst_events": {
                    "T2": [{"timestamp": timestamp, "top_contributors": [{"tag": "AIR_8301A"}]}],
                    "SPE": [{"timestamp": timestamp, "top_contributors": [{"tag": "YFJ3_ZD2"}]}],
                },
            }

            async def call(name, arguments):
                calls.append(name)
                if name == "audit_start_session":
                    return response({"trace_dir": directory})
                if name == "audit_end_session":
                    self.assertEqual(arguments["status"], "complete")
                    return response({"summary": str(root / "summary.md"),
                                     "report": str(root / "report.md"), "trace_sha256": "abc"})
                payloads = {
                    "historian_get_data_range": {"earliest_timestamp": "2022-03-27 14:28:54",
                                                 "latest_timestamp": "2022-04-01 14:28:49"},
                    "historian_list_tags": [{"tag_name": tag} for tag in
                                            ("TE_8332A", "AIR_8301A", "YFJ3_ZD2")],
                    "analysis_run_mspc": analysis,
                    "historian_get_statistics": {"statistics": {"TE_8332A": {"min": 529, "max": 546}}},
                    "alarm_search_context": {"alarms": []},
                    "kg_get_upstream_sensors": {"sensors": []},
                }
                return response(payloads[name], len(calls) - 1)

            async def interpret(*args):
                self.assertEqual(calls, [
                    "audit_start_session", "historian_get_data_range", "historian_list_tags",
                    "analysis_run_mspc", "historian_get_statistics", "alarm_search_context",
                    "kg_get_upstream_sensors", "kg_get_upstream_sensors",
                ])
                return {"role": "assistant", "content": "Analysis complete [step 3]."}

            with patch.object(agent, "ollama_chat", side_effect=interpret):
                answer = await agent.run_turn(
                    SimpleNamespace(call_tool=call), None, [],
                    agent.AgentConfig("http://127.0.0.1"), "/mspc-analysis",
                )
            self.assertIn("outside normal", answer)
            self.assertIn("ANALYSIS LIMITATION", answer)
            self.assertIn(plot.as_uri(), answer)
            self.assertEqual(calls[-1], "audit_end_session")

    async def test_preflight_answer_reports_and_transcript_survive_success_and_failure(self):
        for failure in (False, True, "repeated"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for name in ("summary.md", "report.md"):
                    (root / name).write_text("synthetic report", encoding="utf-8")
                calls = []

                async def call(name, arguments):
                    calls.append(name)
                    if name == "audit_start_session":
                        return response({"trace_dir": directory})
                    if name == "historian_get_data_range":
                        return response({"earliest_timestamp": "2022-03-27 14:28:54",
                                         "latest_timestamp": "2022-04-01 14:28:49"}, 1)
                    if name == "historian_list_tags":
                        return response([{"tag_name": "TE_8332A"}], 2)
                    if name == "audit_end_session":
                        return response({"summary": str(root / "summary.md"),
                                         "report": str(root / "report.md"), "trace_sha256": "abc"})
                    raise AssertionError(name)

                session = SimpleNamespace(call_tool=call)
                if failure == "repeated":
                    model = AsyncMock(return_value={"role": "assistant", "content": "Uncited answer."})
                else:
                    model = AsyncMock(side_effect=agent.AgentError("model unavailable")) if failure else AsyncMock(
                        return_value={"role": "assistant", "content": "Snapshot confirmed [step 1]."})
                with patch.object(agent, "ollama_chat", model):
                    if failure:
                        with self.assertRaises(agent.AgentError):
                            await agent.run_turn(session, None, [], agent.AgentConfig("http://127.0.0.1"), "test")
                        if failure == "repeated":
                            self.assertEqual(model.await_count, 3)
                    else:
                        answer = await agent.run_turn(
                            session, None, [], agent.AgentConfig("http://127.0.0.1"), "test")
                        self.assertIn("file:///", answer)
                        self.assertTrue((root / "answer.md").is_file())
                self.assertEqual(calls[-1], "audit_end_session")
                self.assertTrue((root / "conversation.json").is_file())
                transcript = json.loads((root / "conversation.json").read_text(encoding="utf-8"))
                self.assertEqual(transcript["status"], "failed" if failure else "complete")

    async def test_voluntary_lookup_hypothesis_does_not_require_a_root_cause(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("summary.md", "report.md"):
                (root / name).write_text("report", encoding="utf-8")

            async def call(name, arguments):
                if name == "audit_start_session":
                    return response({"trace_dir": directory})
                if name == "historian_get_data_range":
                    return response({"earliest_timestamp": "2022-03-27 14:28:54",
                                     "latest_timestamp": "2022-04-01 14:28:49"}, 1)
                if name == "historian_list_tags":
                    return response([{"tag_name": "TE_8332A"}], 2)
                if name == "audit_log_reasoning":
                    return response({"step": 3})
                if name == "audit_end_session":
                    return response({"summary": str(root / "summary.md"),
                                     "report": str(root / "report.md"), "trace_sha256": "abc"})
                raise AssertionError(name)

            tools = [{"function": {"name": "audit_log_reasoning", "parameters": {
                "type": "object", "properties": {
                    "reasoning_type": {"type": "string"}, "text": {"type": "string"},
                }, "required": ["reasoning_type", "text"],
            }}}]
            model = AsyncMock(side_effect=[
                {"role": "assistant", "tool_calls": [{"function": {
                    "name": "audit_log_reasoning", "arguments": {
                        "reasoning_type": "hypothesis", "text": "Confirm snapshot.",
                    }}}]},
                {"role": "assistant", "content": "Snapshot scope confirmed [step 1]."},
            ])
            with patch.object(agent, "ollama_chat", model):
                answer = await agent.run_turn(
                    SimpleNamespace(call_tool=call), None, tools,
                    agent.AgentConfig("http://127.0.0.1"), "Show data range")
            self.assertIn("Snapshot scope confirmed", answer)
            self.assertEqual(model.await_count, 2)


if __name__ == "__main__":
    unittest.main()
