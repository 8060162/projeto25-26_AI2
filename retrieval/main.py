from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence

# Ensure the project root is available in sys.path so imports such as
# "from Chunking.config.settings import PipelineSettings" work when running
# "python retrieval/main.py" from the project root.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Chunking.config.settings import PipelineSettings
from retrieval.models import FinalAnswerResult, GuardrailDecision
from retrieval.service import RetrievalService


def run_retrieval_main(
    question_text: str,
    settings: Optional[PipelineSettings] = None,
) -> FinalAnswerResult:
    """
    Execute the retrieval flow for one user question.

    Parameters
    ----------
    question_text : str
        User question submitted through the manual retrieval entrypoint.

    settings : Optional[PipelineSettings]
        Shared runtime settings. Default settings are loaded when omitted.

    Returns
    -------
    FinalAnswerResult
        Final retrieval result returned by the orchestration service.
    """

    normalized_question_text = question_text.strip()
    if not normalized_question_text:
        raise ValueError("Question text cannot be empty.")

    resolved_settings = settings or PipelineSettings()
    retrieval_service = RetrievalService(settings=resolved_settings)
    return retrieval_service.answer_question(normalized_question_text)


def _build_argument_parser() -> argparse.ArgumentParser:
    """
    Build the command-line parser for the retrieval entrypoint.

    Returns
    -------
    argparse.ArgumentParser
        Parser configured for manual retrieval execution.
    """

    parser = argparse.ArgumentParser(
        description="Run one manual question through the retrieval flow."
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Question text to submit to the retrieval service.",
    )
    parser.add_argument(
        "--question",
        dest="question_option",
        help="Explicit question text to submit to the retrieval service.",
    )
    return parser


def _resolve_question_text(arguments: argparse.Namespace) -> str:
    """
    Resolve the effective question text from parsed CLI arguments.

    Parameters
    ----------
    arguments : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    str
        Final question text that should be executed.
    """

    if isinstance(arguments.question_option, str) and arguments.question_option.strip():
        return arguments.question_option.strip()

    if isinstance(arguments.question, str) and arguments.question.strip():
        return arguments.question.strip()

    raise ValueError(
        "A question must be provided through the positional argument or "
        "'--question'."
    )


def _print_guardrail_summary(
    label: str,
    decision: Optional[GuardrailDecision],
) -> None:
    """
    Print one concise guardrail summary when a decision is available.

    Parameters
    ----------
    label : str
        Human-readable label describing the guardrail stage.

    decision : Optional[GuardrailDecision]
        Guardrail decision to summarize.
    """

    if decision is None:
        return

    print(
        f"[INFO] {label}: allowed={decision.allowed}; "
        f"action={decision.action}; category={decision.category or '<none>'}"
    )
    if decision.reason:
        print(f"[INFO] {label} reason: {decision.reason}")
    if decision.matched_rules:
        print(
            f"[INFO] {label} matched rules: "
            + ", ".join(decision.matched_rules)
        )


def _print_result_summary(result: FinalAnswerResult) -> None:
    """
    Print a concise summary of the completed retrieval execution.

    Parameters
    ----------
    result : FinalAnswerResult
        Final retrieval result returned by the service.
    """

    print(f"[INFO] Retrieval status: {result.status}")
    print(f"[INFO] Grounded: {result.grounded}")

    if result.retrieval_context is not None:
        print(
            "[INFO] Retrieval context: "
            f"{result.retrieval_context.chunk_count} chunks, "
            f"{result.retrieval_context.character_count} characters"
        )

    if result.citations:
        print(f"[INFO] Citations: {', '.join(result.citations)}")

    _print_guardrail_summary("Pre-guardrail", result.pre_guardrail)
    _print_guardrail_summary("Post-guardrail", result.post_guardrail)

    print("[INFO] Answer:")
    print(result.answer_text)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    Run the standalone retrieval entrypoint and print the final result.

    Parameters
    ----------
    argv : Optional[Sequence[str]]
        Optional argument list used instead of `sys.argv`.

    Returns
    -------
    int
        Process exit code for shell execution.
    """

    parser = _build_argument_parser()
    arguments = parser.parse_args(argv)

    try:
        question_text = _resolve_question_text(arguments)
        result = run_retrieval_main(
            question_text=question_text,
            settings=PipelineSettings(),
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    _print_result_summary(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
