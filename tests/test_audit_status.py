"""Audit status must distinguish sealed partial evidence from completed diagnosis."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from audit_trace import TraceSession, summarize_result
from render_audit import render_report, render_summary


class AuditStatusTests(unittest.TestCase):
    def test_partial_run_banner_and_hash_are_preserved_on_rerender(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = TraceSession("synthetic", "Test failure reporting", root)
            session.record_tool_call(
                "historian_get_data_range", {}, {"total_rows": 86400}, 1.0, True,
            )
            meta = session.finalize("failed", "Synthetic inference failure")
            for renderer in (render_report, render_summary):
                text = Path(renderer(directory)).read_text(encoding="utf-8")
                self.assertIn("INCOMPLETE RUN (failed)", text)
                self.assertIn("Synthetic inference failure", text)
            checksum = hashlib.sha256((root / "trace.jsonl").read_bytes())
            for path in sorted((root / "raw").glob("step_*.json")):
                checksum.update(hashlib.sha256(path.read_bytes()).hexdigest().encode("utf-8"))
            self.assertEqual(meta["trace_sha256"], checksum.hexdigest())
            self.assertEqual(meta, json.loads((root / "session_meta.json").read_text(encoding="utf-8")))
            self.assertEqual(session.finalize(), meta)

    def test_analytical_results_have_useful_audit_summaries(self):
        summary = summarize_result("analysis_run_mspc", {
            "model": {"total_rows": 7200, "n_components": 8, "cumulative_variance": 0.8539}})
        self.assertIn("8 PCA components", summary["summary_text"])
        self.assertEqual(summary["metrics"]["total_rows"], 7200)


if __name__ == "__main__":
    unittest.main()
