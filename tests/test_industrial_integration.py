"""Real MCP integration tests; no LLM is required and plant files are read through tools."""

import json
import os
from pathlib import Path
import unittest

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from local_runtime.agent import PROJECT_DIR, PYTHON_EXE, checked_call


class IndustrialIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_existing_sources_and_sealed_reports_through_real_mcp(self):
        params = StdioServerParameters(
            command=str(PYTHON_EXE), args=[str(PROJECT_DIR / "historian_mcp_server.py")],
            cwd=str(PROJECT_DIR),
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "MPLBACKEND": "Agg",
                 "ANONYMIZED_TELEMETRY": "False"},
        )
        async with stdio_client(params) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                tools = {tool.name for tool in (await session.list_tools()).tools}
                self.assertTrue({"analysis_run_mspc", "analysis_run_fault_detection"} <= tools)
                await checked_call(session, "audit_start_session", {"question": "Local runtime integration test"})
                try:
                    data_range = await checked_call(session, "historian_get_data_range", {})
                    self.assertEqual(data_range["result"]["total_rows"], 86400)
                    tags = await checked_call(session, "historian_list_tags", {})
                    self.assertEqual(len(tags["result"]), 30)
                    pressure_tags = {"PTCA_8322A", "PTCA_8324"}
                    self.assertTrue(all(
                        tag["units"] == "MPa" for tag in tags["result"]
                        if tag["tag_name"] in pressure_tags
                    ))
                    for tag in sorted(pressure_tags):
                        searched = await checked_call(session, "historian_search_tags", {"query": tag})
                        self.assertEqual(searched["result"][0]["units"], "MPa")
                    pressure = await checked_call(session, "historian_get_statistics", {
                        "tag_names": sorted(pressure_tags), "start_time": "2022-04-01 13:28:49",
                        "end_time": "2022-04-01 14:28:49",
                    })
                    for tag in pressure_tags:
                        values = pressure["result"]["statistics"][tag]
                        self.assertEqual(values["units"], "MPa")
                        self.assertGreater(values["mean"], 0)
                        self.assertLess(values["max"], 12)
                    stats = await checked_call(session, "historian_get_statistics", {
                        "tag_names": ["TE_8332A"], "start_time": "2022-04-01 13:28:49",
                        "end_time": "2022-04-01 14:28:49"})
                    self.assertGreater(stats["result"]["statistics"]["TE_8332A"]["count"], 0)
                    for name, arguments in (
                        ("alarm_search_context", {"timestamp": "2022-03-29 14:00:00"}),
                        ("alarm_get_active_at", {"timestamp": "2022-03-29 14:00:00"}),
                        ("alarm_get_statistics", {"start_time": "2022-03-29 13:00:00",
                                                  "end_time": "2022-03-29 15:00:00"}),
                        ("alarm_detect_flood", {"start_time": "2022-03-29 13:00:00",
                                                "end_time": "2022-03-29 15:00:00"}),
                        ("kg_get_upstream_sensors", {"node_id": "TE_8332A"}),
                        ("kg_trace_stream", {"stream_name": "Steam"}),
                    ):
                        result = await checked_call(session, name, arguments)
                        self.assertIsInstance(result["audit_step"], int)
                    documents = await checked_call(session, "docs_list_documents", {})
                    self.assertEqual(len(documents["result"]), 62)
                    document = await checked_call(session, "docs_get_document", {
                        "doc_id": documents["result"][0]["doc_id"]})
                    self.assertTrue(document["result"]["revision"])
                    plot = await checked_call(session, "historian_plot_tags", {
                        "tag_names": ["TE_8332A"], "start_time": "2022-04-01 13:28:49",
                        "end_time": "2022-04-01 14:28:49", "show_normal_range": True})
                    self.assertTrue(Path(plot["result"]["plot_path"]).is_file())
                    error = await session.call_tool("historian_get_statistics", arguments={
                        "tag_names": ["INVALID_TAG"], "start_time": "2022-04-01 13:28:49",
                        "end_time": "2022-04-01 14:28:49"})
                    self.assertTrue(error.isError)
                finally:
                    closed = await checked_call(session, "audit_end_session", {})
                self.assertEqual(len(closed["trace_sha256"]), 64)
                for key in ("report", "summary"):
                    self.assertTrue(Path(closed[key]).is_file())
                entries = [json.loads(line) for line in (
                    Path(closed["trace_dir"]) / "trace.jsonl"
                ).read_text(encoding="utf-8").splitlines()]
                self.assertFalse(entries[-1]["success"])


if __name__ == "__main__":
    unittest.main()
