from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import traceback
from typing import Any, Optional, Sequence

# Ensure the project root is available in sys.path so imports such as
# "from Chunking.config.settings import PipelineSettings" work when running
# "python retrieval/main.py" from the project root.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Chunking.config.settings import PipelineSettings
from retrieval.models import FinalAnswerResult, GuardrailDecision, MetricsSnapshot
from retrieval.service import RetrievalService


def _configure_cli_logging(enabled: bool) -> None:
    """
    Configure warning logs for manual retrieval executions.

    Parameters
    ----------
    enabled : bool
        Whether warning logs should be written to standard error.
    """

    if not enabled:
        return

    logging.basicConfig(
        level=logging.WARNING,
        format="[%(levelname)s] %(name)s: %(message)s",
    )


class RetrievalCliProgress:
    """
    Emit concise progress messages for manual retrieval executions.
    """

    def __init__(self, enabled: bool = True) -> None:
        """
        Initialize the progress reporter.

        Parameters
        ----------
        enabled : bool
            Whether progress messages should be written to standard error.
        """

        self.enabled = enabled
        self.started_at = time.perf_counter()

    def info(self, message: str) -> None:
        """
        Write one informational progress message when reporting is enabled.

        Parameters
        ----------
        message : str
            Human-readable progress message.
        """

        if not self.enabled:
            return

        elapsed_seconds = time.perf_counter() - self.started_at
        print(
            f"[INFO] +{elapsed_seconds:.1f}s {message}",
            file=sys.stderr,
            flush=True,
        )


class _ProgressGuardrails:
    """
    Wrap guardrail checks with CLI progress reporting.
    """

    def __init__(self, delegate: Any, progress: RetrievalCliProgress) -> None:
        """
        Initialize the guardrail progress wrapper.

        Parameters
        ----------
        delegate : Any
            Guardrail implementation used by the retrieval service.

        progress : RetrievalCliProgress
            CLI progress reporter.
        """

        self.delegate = delegate
        self.progress = progress

    def evaluate_pre_request(self, question: Any) -> Any:
        """
        Evaluate pre-request guardrails with progress messages.

        Parameters
        ----------
        question : Any
            Normalized question passed by the retrieval service.

        Returns
        -------
        Any
            Guardrail decision returned by the wrapped implementation.
        """

        self.progress.info("Checking pre-request guardrails.")
        decision = self.delegate.evaluate_pre_request(question)
        self.progress.info("Pre-request guardrails completed.")
        return decision

    def evaluate_post_response(
        self,
        *,
        answer_text: str,
        context: Any,
        grounded: bool,
    ) -> Any:
        """
        Evaluate post-response guardrails with progress messages.

        Parameters
        ----------
        answer_text : str
            Generated answer text.

        context : Any
            Retrieval context used for generation.

        grounded : bool
            Whether the answer generator marked the answer as grounded.

        Returns
        -------
        Any
            Guardrail decision returned by the wrapped implementation.
        """

        self.progress.info("Checking post-response guardrails.")
        decision = self.delegate.evaluate_post_response(
            answer_text=answer_text,
            context=context,
            grounded=grounded,
        )
        self.progress.info("Post-response guardrails completed.")
        return decision


class _ProgressRetrievalRouter:
    """
    Wrap retrieval routing with CLI progress reporting.
    """

    def __init__(self, delegate: Any, progress: RetrievalCliProgress) -> None:
        """
        Initialize the routing progress wrapper.

        Parameters
        ----------
        delegate : Any
            Retrieval router implementation.

        progress : RetrievalCliProgress
            CLI progress reporter.
        """

        self.delegate = delegate
        self.progress = progress

    def route(self, question: Any) -> Any:
        """
        Route one question with progress messages.

        Parameters
        ----------
        question : Any
            Normalized question passed by the retrieval service.

        Returns
        -------
        Any
            Route decision returned by the wrapped implementation.
        """

        self.progress.info("Routing retrieval request.")
        decision = self.delegate.route(question)
        self.progress.info("Retrieval routing completed.")
        return decision


class _ProgressEmbeddingProvider:
    """
    Wrap query embedding with CLI progress reporting.
    """

    def __init__(self, delegate: Any, progress: RetrievalCliProgress) -> None:
        """
        Initialize the embedding progress wrapper.

        Parameters
        ----------
        delegate : Any
            Embedding provider implementation.

        progress : RetrievalCliProgress
            CLI progress reporter.
        """

        self.delegate = delegate
        self.progress = progress

    def embed_texts(self, texts: Sequence[str]) -> Any:
        """
        Generate embeddings with progress messages.

        Parameters
        ----------
        texts : Sequence[str]
            Text batch passed by the retrieval service.

        Returns
        -------
        Any
            Embedding vectors returned by the wrapped implementation.
        """

        self.progress.info("Generating query embedding.")
        vectors = self.delegate.embed_texts(texts)
        self.progress.info("Query embedding completed.")
        return vectors


class _ProgressStorage:
    """
    Wrap vector-store retrieval with CLI progress reporting.
    """

    def __init__(self, delegate: Any, progress: RetrievalCliProgress) -> None:
        """
        Initialize the storage progress wrapper.

        Parameters
        ----------
        delegate : Any
            Embedding storage implementation.

        progress : RetrievalCliProgress
            CLI progress reporter.
        """

        self.delegate = delegate
        self.progress = progress

    def query_similar_chunks(self, **kwargs: Any) -> Any:
        """
        Query similar chunks with progress messages.

        Parameters
        ----------
        kwargs : Any
            Storage query arguments passed by the retrieval service.

        Returns
        -------
        Any
            Retrieved chunks returned by the wrapped implementation.
        """

        top_k = kwargs.get("top_k", "<default>")
        self.progress.info(f"Querying ChromaDB for up to {top_k} candidate chunks.")
        results = self.delegate.query_similar_chunks(**kwargs)
        self.progress.info(f"ChromaDB returned {len(results)} candidate chunks.")
        return results


class _ProgressContextBuilder:
    """
    Wrap context building with CLI progress reporting.
    """

    def __init__(self, delegate: Any, progress: RetrievalCliProgress) -> None:
        """
        Initialize the context-builder progress wrapper.

        Parameters
        ----------
        delegate : Any
            Context builder implementation.

        progress : RetrievalCliProgress
            CLI progress reporter.
        """

        self.delegate = delegate
        self.progress = progress

    def build_context(self, retrieved_chunks: Any, **kwargs: Any) -> Any:
        """
        Build retrieval context with progress messages.

        Parameters
        ----------
        retrieved_chunks : Any
            Retrieved candidate chunks.

        kwargs : Any
            Context-builder arguments passed by the retrieval service.

        Returns
        -------
        Any
            Retrieval context returned by the wrapped implementation.
        """

        self.progress.info("Building grounded retrieval context.")
        context = self.delegate.build_context(retrieved_chunks, **kwargs)
        self.progress.info(
            "Context building completed with "
            f"{context.chunk_count} selected chunks."
        )
        return context


class _ProgressAnswerGenerator:
    """
    Wrap answer generation with CLI progress reporting.
    """

    def __init__(self, delegate: Any, progress: RetrievalCliProgress) -> None:
        """
        Initialize the answer-generator progress wrapper.

        Parameters
        ----------
        delegate : Any
            Answer generator implementation.

        progress : RetrievalCliProgress
            CLI progress reporter.
        """

        self.delegate = delegate
        self.progress = progress

    def generate_answer(self, generation_input: Any) -> Any:
        """
        Generate a grounded answer with progress messages.

        Parameters
        ----------
        generation_input : Any
            Answer-generation input built by the retrieval service.

        Returns
        -------
        Any
            Generated answer returned by the wrapped implementation.
        """

        self.progress.info("Generating grounded answer.")
        answer = self.delegate.generate_answer(generation_input)
        self.progress.info("Answer generation completed.")
        return answer


class _ProgressGroundingValidator:
    """
    Wrap grounding validation with CLI progress reporting.
    """

    def __init__(self, delegate: Any, progress: RetrievalCliProgress) -> None:
        """
        Initialize the grounding-validator progress wrapper.

        Parameters
        ----------
        delegate : Any
            Grounding validator implementation.

        progress : RetrievalCliProgress
            CLI progress reporter.
        """

        self.delegate = delegate
        self.progress = progress

    def validate(self, **kwargs: Any) -> Any:
        """
        Validate grounding with progress messages.

        Parameters
        ----------
        kwargs : Any
            Grounding-validation arguments passed by the retrieval service.

        Returns
        -------
        Any
            Grounding result returned by the wrapped implementation.
        """

        self.progress.info("Validating answer grounding.")
        result = self.delegate.validate(**kwargs)
        self.progress.info("Grounding validation completed.")
        return result


def run_retrieval_main(
    question_text: str,
    settings: Optional[PipelineSettings] = None,
    progress: Optional[RetrievalCliProgress] = None,
) -> FinalAnswerResult:
    """
    Execute the retrieval flow for one user question.

    Parameters
    ----------
    question_text : str
        User question submitted through the manual retrieval entrypoint.

    settings : Optional[PipelineSettings]
        Shared runtime settings. Default settings are loaded when omitted.

    progress : Optional[RetrievalCliProgress]
        Optional CLI progress reporter used by the manual entrypoint.

    Returns
    -------
    FinalAnswerResult
        Final retrieval result returned by the orchestration service.
    """

    normalized_question_text = question_text.strip()
    if not normalized_question_text:
        raise ValueError("Question text cannot be empty.")

    progress_reporter = progress or RetrievalCliProgress(enabled=False)
    progress_reporter.info("Loading settings and initializing retrieval service.")
    resolved_settings = settings or PipelineSettings()
    retrieval_service = RetrievalService(settings=resolved_settings)
    _attach_progress_wrappers(retrieval_service, progress_reporter)
    progress_reporter.info("Running retrieval pipeline.")
    return retrieval_service.answer_question(normalized_question_text)


def _attach_progress_wrappers(
    retrieval_service: RetrievalService,
    progress: RetrievalCliProgress,
) -> None:
    """
    Attach CLI-only progress wrappers around slow retrieval dependencies.

    Parameters
    ----------
    retrieval_service : RetrievalService
        Service instance used by the manual entrypoint.

    progress : RetrievalCliProgress
        Progress reporter that should receive stage updates.
    """

    if not progress.enabled:
        return

    retrieval_service.guardrails = _ProgressGuardrails(
        retrieval_service.guardrails,
        progress,
    )
    retrieval_service.retrieval_router = _ProgressRetrievalRouter(
        retrieval_service.retrieval_router,
        progress,
    )
    retrieval_service.embedding_provider = _ProgressEmbeddingProvider(
        retrieval_service.embedding_provider,
        progress,
    )
    retrieval_service.storage = _ProgressStorage(retrieval_service.storage, progress)
    retrieval_service.context_builder = _ProgressContextBuilder(
        retrieval_service.context_builder,
        progress,
    )
    retrieval_service.answer_generator = _ProgressAnswerGenerator(
        retrieval_service.answer_generator,
        progress,
    )
    retrieval_service.grounding_validator = _ProgressGroundingValidator(
        retrieval_service.grounding_validator,
        progress,
    )


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
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages and print only the final result.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print a traceback when an unexpected error occurs.",
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
        f"action={decision.action}; category={decision.category or '<none>'}",
        flush=True,
    )
    if decision.reason:
        print(f"[INFO] {label} reason: {decision.reason}", flush=True)
    if decision.matched_rules:
        print(
            f"[INFO] {label} matched rules: "
            + ", ".join(decision.matched_rules),
            flush=True,
        )


def _format_latency_seconds(elapsed_ms: float) -> str:
    """
    Format one latency value from milliseconds into seconds.

    Parameters
    ----------
    elapsed_ms : float
        Elapsed stage time in milliseconds.

    Returns
    -------
    str
        Human-readable latency value in seconds.
    """

    return f"{elapsed_ms / 1000.0:.1f}s"


def _print_latency_summary(metrics_snapshot: Optional[MetricsSnapshot]) -> None:
    """
    Print stage-latency metrics when the retrieval flow collected them.

    Parameters
    ----------
    metrics_snapshot : Optional[MetricsSnapshot]
        Metrics snapshot attached to the final retrieval result.
    """

    if metrics_snapshot is None or not metrics_snapshot.stage_latency_ms:
        return

    stage_order = [
        "pre_guardrails",
        "retrieval_routing",
        "query_embedding",
        "retrieval",
        "context_builder",
        "answer_generation",
        "grounding_validation",
        "post_guardrails",
    ]
    formatted_stages = []

    for stage_name in stage_order:
        elapsed_ms = metrics_snapshot.stage_latency_ms.get(stage_name)
        if elapsed_ms is not None:
            formatted_stages.append(
                f"{stage_name}={_format_latency_seconds(elapsed_ms)}"
            )

    for stage_name, elapsed_ms in metrics_snapshot.stage_latency_ms.items():
        if stage_name not in stage_order:
            formatted_stages.append(
                f"{stage_name}={_format_latency_seconds(elapsed_ms)}"
            )

    print(
        "[INFO] Latency: "
        f"total={_format_latency_seconds(metrics_snapshot.total_latency_ms)}; "
        + "; ".join(formatted_stages),
        flush=True,
    )


def _print_result_summary(result: FinalAnswerResult) -> None:
    """
    Print a concise summary of the completed retrieval execution.

    Parameters
    ----------
    result : FinalAnswerResult
        Final retrieval result returned by the service.
    """

    print(f"[INFO] Retrieval status: {result.status}", flush=True)
    print(f"[INFO] Grounded: {result.grounded}", flush=True)

    if result.retrieval_context is not None:
        print(
            "[INFO] Retrieval context: "
            f"{result.retrieval_context.chunk_count} chunks, "
            f"{result.retrieval_context.character_count} characters",
            flush=True,
        )

    if result.citations:
        print(f"[INFO] Citations: {', '.join(result.citations)}", flush=True)

    _print_latency_summary(result.metrics_snapshot)
    _print_guardrail_summary("Pre-guardrail", result.pre_guardrail)
    _print_guardrail_summary("Post-guardrail", result.post_guardrail)

    print("[INFO] Answer:", flush=True)
    print(result.answer_text or "<empty answer>", flush=True)


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
    progress = RetrievalCliProgress(enabled=not arguments.quiet)
    _configure_cli_logging(enabled=not arguments.quiet)

    try:
        question_text = _resolve_question_text(arguments)
        result = run_retrieval_main(
            question_text=question_text,
            settings=None,
            progress=progress,
        )
        progress.info("Retrieval pipeline completed. Printing final result.")
        _print_result_summary(result)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(
            f"[ERROR] Unexpected {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        if arguments.debug:
            traceback.print_exc(file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
