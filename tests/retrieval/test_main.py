"""Regression tests for the manual retrieval command-line entrypoint."""

from __future__ import annotations

import contextlib
import io
import unittest
from unittest.mock import patch

from retrieval.main import main
from retrieval.models import FinalAnswerResult, MetricsSnapshot, UserQuestionInput


class RetrievalMainTests(unittest.TestCase):
    """Protect CLI feedback and error handling for manual retrieval runs."""

    def _build_result(self) -> FinalAnswerResult:
        """
        Build one minimal final answer result for CLI tests.

        Returns
        -------
        FinalAnswerResult
            Completed retrieval result used by patched main runs.
        """

        return FinalAnswerResult(
            question=UserQuestionInput(question_text="Pergunta de teste?"),
            status="completed",
            answer_text="Resposta de teste.",
            grounded=True,
            metrics_snapshot=MetricsSnapshot(
                stage_latency_ms={
                    "query_embedding": 10300.0,
                    "retrieval": 1600.0,
                    "answer_generation": 15100.0,
                },
                total_latency_ms=27000.0,
            ),
        )

    def test_main_prints_progress_by_default(self) -> None:
        """
        Verify the manual CLI reports progress before printing the result.
        """

        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        with patch("retrieval.main.RetrievalService") as service_class:
            service_class.return_value.answer_question.return_value = (
                self._build_result()
            )

            with contextlib.redirect_stdout(stdout_buffer):
                with contextlib.redirect_stderr(stderr_buffer):
                    exit_code = main(["--question", "Pergunta de teste?"])

        self.assertEqual(exit_code, 0)
        self.assertIn("[INFO] +", stderr_buffer.getvalue())
        self.assertIn("Running retrieval pipeline.", stderr_buffer.getvalue())
        self.assertIn(
            "Retrieval pipeline completed. Printing final result.",
            stderr_buffer.getvalue(),
        )
        self.assertIn(
            "Latency: total=27.0s; query_embedding=10.3s",
            stdout_buffer.getvalue(),
        )
        self.assertIn("[INFO] Answer:", stdout_buffer.getvalue())
        self.assertIn("Resposta de teste.", stdout_buffer.getvalue())

    def test_main_quiet_suppresses_progress(self) -> None:
        """
        Verify --quiet keeps progress messages out of standard error.
        """

        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        with patch("retrieval.main.RetrievalService") as service_class:
            service_class.return_value.answer_question.return_value = (
                self._build_result()
            )

            with contextlib.redirect_stdout(stdout_buffer):
                with contextlib.redirect_stderr(stderr_buffer):
                    exit_code = main(
                        ["--quiet", "--question", "Pergunta de teste?"]
                    )

        self.assertEqual(exit_code, 0)
        self.assertEqual(stderr_buffer.getvalue(), "")
        self.assertIn("Resposta de teste.", stdout_buffer.getvalue())

    def test_main_reports_unexpected_errors(self) -> None:
        """
        Verify unexpected exceptions become actionable CLI errors.
        """

        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()

        with patch("retrieval.main.RetrievalService") as service_class:
            service_class.side_effect = TypeError("broken dependency")

            with contextlib.redirect_stdout(stdout_buffer):
                with contextlib.redirect_stderr(stderr_buffer):
                    exit_code = main(["--question", "Pergunta de teste?"])

        self.assertEqual(exit_code, 1)
        self.assertEqual(stdout_buffer.getvalue(), "")
        self.assertIn(
            "[ERROR] Unexpected TypeError: broken dependency",
            stderr_buffer.getvalue(),
        )


if __name__ == "__main__":
    unittest.main()
