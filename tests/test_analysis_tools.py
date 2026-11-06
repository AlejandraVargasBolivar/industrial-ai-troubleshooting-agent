"""Regression tests for isolated, bounded analytics subprocess execution."""

import asyncio
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import psutil

from local_runtime import analysis_tools


class AnalysisWorkflowTests(unittest.TestCase):
    data_range = {
        "earliest_timestamp": "2022-03-27 14:28:54",
        "latest_timestamp": "2022-04-01 14:28:49",
    }

    def test_mspc_cross_references_both_statistics_without_losing_distinct_contributors(self):
        events = {
            "T2": [{"timestamp": "2022-03-28 19:25:00",
                    "top_contributors": [{"tag": "AIR_8301A"}]}],
            "SPE": [{"timestamp": "2022-03-28 19:25:00",
                     "top_contributors": [{"tag": "YFJ3_ZD2"}]}],
        }
        calls = analysis_tools.analysis_followup_calls(
            "analysis_run_mspc", {
                "model": {"tags_used": ["TE_8332A"]}, "worst_events": events,
            }, self.data_range,
        )
        self.assertEqual(len([c for c in calls if c[0] == "alarm_search_context"]), 1)
        self.assertEqual(
            [args["node_id"] for name, args in calls if name == "kg_get_upstream_sensors"],
            ["AIR_8301A", "YFJ3_ZD2"],
        )
        self.assertEqual(calls[0], ("historian_get_statistics", {
            "tag_names": ["TE_8332A"], "start_time": self.data_range["earliest_timestamp"],
            "end_time": self.data_range["latest_timestamp"],
        }))

    def test_ml_cross_references_each_event_and_the_three_highest_ranked_features(self):
        features = [{"tag": tag} for tag in ("YFJ3_ZD2", "AIR_8301A", "FT_8301", "TE_8319A")]
        calls = analysis_tools.analysis_followup_calls(
            "analysis_run_fault_detection", {
                "model": {"tags_used": ["TE_8332A"]},
                "worst_events": [{"start": "2022-03-28 19:25:00"}],
                "feature_importance": {"top_features": features},
            }, self.data_range,
        )
        self.assertEqual(len(calls), 10)
        self.assertEqual(len([c for c in calls if c[0] == "alarm_get_active_at"]), 1)
        self.assertEqual(
            [args["tag_names"] for name, args in calls if name == "alarm_get_statistics"],
            [[feature["tag"]] for feature in features[:3]],
        )
        self.assertEqual(calls[-1][0], "docs_search")
        self.assertEqual(calls[-1][1]["sensor_tag"], features[0]["tag"])
        for name, args in calls:
            if name == "alarm_get_statistics":
                self.assertEqual(args["end_time"], self.data_range["latest_timestamp"])

    def test_no_events_or_feature_ranking_does_not_invent_queries(self):
        calls = analysis_tools.analysis_followup_calls(
            "analysis_run_fault_detection", {
                "model": {"tags_used": []}, "worst_events": [],
                "feature_importance": {"top_features": []},
            }, self.data_range,
        )
        self.assertEqual(calls, [])


class AnalysisProcessTests(unittest.IsolatedAsyncioTestCase):
    async def test_stdin_is_isolated_and_valid_json_is_returned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "mspc_analysis.py").write_text(
                "import json, sys\n"
                "assert sys.stdin.read() == ''\n"
                "print(json.dumps({'model': {}, 'results': {}, 'plots': {}}))\n",
                encoding="utf-8",
            )
            with patch.object(analysis_tools, "PROJECT_DIR", root):
                result = await analysis_tools.run_analysis("analysis_run_mspc", {}, timeout=10)
            self.assertEqual(result["plots"], {})

    async def test_timeout_reports_progress_and_kills_redirector_descendants(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pid_path = root / "child.pid"
            (root / "mspc_analysis.py").write_text(
                "import subprocess, sys, time\n"
                "from pathlib import Path\n"
                "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
                f"Path({str(pid_path)!r}).write_text(str(child.pid))\n"
                "print('Synthetic progress before timeout', file=sys.stderr, flush=True)\n"
                "time.sleep(60)\n", encoding="utf-8",
            )
            with patch.object(analysis_tools, "PROJECT_DIR", root):
                with self.assertRaisesRegex(RuntimeError, "Synthetic progress before timeout"):
                    await analysis_tools.run_analysis("analysis_run_mspc", {}, timeout=2)
            self.assertTrue(pid_path.is_file())
            self.assertFalse(psutil.pid_exists(int(pid_path.read_text())))

    async def test_cancellation_cleans_up_analytic_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "mspc_analysis.py").write_text(
                "import os, time\nfrom pathlib import Path\n"
                "Path('running.pid').write_text(str(os.getpid()))\ntime.sleep(60)\n",
                encoding="utf-8",
            )
            with patch.object(analysis_tools, "PROJECT_DIR", root):
                task = asyncio.create_task(analysis_tools.run_analysis("analysis_run_mspc", {}, timeout=30))
                for _ in range(100):
                    if (root / "running.pid").exists():
                        break
                    await asyncio.sleep(0.05)
                self.assertTrue((root / "running.pid").exists())
                pid = int((root / "running.pid").read_text())
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertFalse(psutil.pid_exists(pid))


if __name__ == "__main__":
    unittest.main()
