"""Offline tests for the repeatable synthetic model evaluation."""

import unittest
from unittest.mock import AsyncMock, patch

from local_runtime.agent import AgentConfig, AgentError
from scripts import evaluate_models as evaluation


class ModelEvaluationTests(unittest.IsolatedAsyncioTestCase):
    async def test_snapshot_window_requires_bounds_then_exact_cross_midnight_window(self):
        config = AgentConfig("http://127.0.0.1")
        messages = [
            {"tool_calls": [{"function": {"name": "snapshot_bounds", "arguments": {}}}]},
            {"tool_calls": [{"function": {"name": "read_sensor", "arguments": {
                "tag": "SYNTH_TEMPERATURE", "start_time": "2024-01-01 23:00:00",
                "end_time": "2024-01-02 01:00:00",
            }}}]},
            {"content": "There are 37 observations."},
        ]
        with patch.object(evaluation, "ollama_chat", AsyncMock(side_effect=messages)) as chat:
            result = await evaluation.snapshot_window(None, config)
        self.assertTrue(result["passed"])
        self.assertEqual(chat.await_count, 3)

    async def test_wrong_window_is_a_failure_not_a_partial_pass(self):
        with patch.object(evaluation, "ollama_chat", AsyncMock(return_value={
            "tool_calls": [{"function": {"name": "read_sensor", "arguments": {
                "tag": "SYNTH_TEMPERATURE", "start_time": "2024-01-02 00:00:00",
                "end_time": "2024-01-02 01:00:00",
            }}}],
        })):
            with self.assertRaises(AgentError):
                await evaluation.snapshot_window(None, AgentConfig("http://127.0.0.1"))

    async def test_caution_does_not_accept_claims_of_proven_causality(self):
        for separator, correct in ((" ", True), ("\n", True), (", ", True), (" ", False)):
            with self.subTest(separator=separator, correct=correct):
                content = separator.join(evaluation.CAUTION_MARKERS) + " Insufficient evidence."
                if not correct:
                    content = content.replace("[causality:unproven]", "[causality:proven]")
                with patch.object(evaluation, "ollama_chat", AsyncMock(return_value={"content": content})):
                    if correct:
                        self.assertTrue((await evaluation.interpretation_caution(
                            None, AgentConfig("http://127.0.0.1"),
                        ))["passed"])
                    else:
                        with self.assertRaises(AgentError):
                            await evaluation.interpretation_caution(None, AgentConfig("http://127.0.0.1"))


if __name__ == "__main__":
    unittest.main()
