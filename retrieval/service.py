from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from Chunking.config.settings import PipelineSettings
from embedding.provider_factory import EmbeddingProvider, create_embedding_provider
from embedding.storage import ChromaEmbeddingStorage
from retrieval.answer_generator import (
    AnswerGenerationError,
    AnswerGenerator,
    GeneratedAnswer,
    create_answer_generator,
)
from retrieval.context_builder import RetrievalContextBuilder
from retrieval.grounding_validator import GroundingValidator
from retrieval.guardrails import DeterministicGuardrails
from retrieval.metrics import RetrievalMetricsCollector
from retrieval.models import (
    AnswerGenerationInput,
    DiagnosticSignal,
    FinalAnswerResult,
    GroundingVerificationResult,
    GuardrailDecision,
    RetrievalRouteDecision,
    RetrievalRouteMetadata,
    RetrievalContext,
    RetrievedChunkResult,
    UserQuestionInput,
)
from retrieval.query_normalizer import SemanticQueryNormalizer
from retrieval.retrieval_router import RetrievalRouter


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class _RetrievalPassResult:
    """
    Store the runtime output of one retrieval and context-building pass.
    """

    pass_name: str
    route_decision: RetrievalRouteDecision
    retrieved_chunks: List[RetrievedChunkResult]
    retrieval_context: RetrievalContext
    retrieval_breadth: int
    retrieval_filter: Optional[Dict[str, Any]]


@dataclass(slots=True)
class RetrievalService:
    """
    Orchestrate the end-to-end retrieval and grounded answer flow.

    Design goals
    ------------
    - keep orchestration centralized in one service layer
    - reuse the existing embedding, storage, context, guardrail, and metrics
      modules without duplicating their responsibilities
    - expose dependency injection points that make the service testable
    """

    settings: PipelineSettings
    embedding_provider: EmbeddingProvider
    storage: ChromaEmbeddingStorage
    context_builder: RetrievalContextBuilder
    guardrails: DeterministicGuardrails
    answer_generator: AnswerGenerator
    metrics_collector: RetrievalMetricsCollector
    query_normalizer: SemanticQueryNormalizer
    retrieval_router: RetrievalRouter
    grounding_validator: GroundingValidator

    def __init__(
        self,
        settings: Optional[PipelineSettings] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        storage: Optional[ChromaEmbeddingStorage] = None,
        context_builder: Optional[RetrievalContextBuilder] = None,
        guardrails: Optional[DeterministicGuardrails] = None,
        answer_generator: Optional[AnswerGenerator] = None,
        metrics_collector: Optional[RetrievalMetricsCollector] = None,
        query_normalizer: Optional[SemanticQueryNormalizer] = None,
        retrieval_router: Optional[RetrievalRouter] = None,
        grounding_validator: Optional[GroundingValidator] = None,
    ) -> None:
        """
        Initialize the retrieval orchestration service.

        Parameters
        ----------
        settings : Optional[PipelineSettings]
            Shared runtime settings. Default settings are loaded when omitted.

        embedding_provider : Optional[EmbeddingProvider]
            Optional embedding provider override used for query embeddings.

        storage : Optional[ChromaEmbeddingStorage]
            Optional storage-layer override used for ChromaDB retrieval.

        context_builder : Optional[RetrievalContextBuilder]
            Optional context-builder override.

        guardrails : Optional[DeterministicGuardrails]
            Optional deterministic guardrails override.

        answer_generator : Optional[AnswerGenerator]
            Optional grounded answer-generator override.

        metrics_collector : Optional[RetrievalMetricsCollector]
            Optional metrics collector override.

        query_normalizer : Optional[SemanticQueryNormalizer]
            Optional deterministic semantic-query normalizer override.

        retrieval_router : Optional[RetrievalRouter]
            Optional deterministic retrieval-router override.

        grounding_validator : Optional[GroundingValidator]
            Optional post-generation grounding validator override.
        """

        resolved_settings = settings or PipelineSettings()

        self.settings = resolved_settings
        self.embedding_provider = embedding_provider or create_embedding_provider(
            resolved_settings
        )
        self.storage = storage or ChromaEmbeddingStorage(resolved_settings)
        self.context_builder = context_builder or RetrievalContextBuilder(
            resolved_settings
        )
        self.guardrails = guardrails or DeterministicGuardrails(resolved_settings)
        self.answer_generator = answer_generator or create_answer_generator(
            resolved_settings
        )
        self.metrics_collector = metrics_collector or RetrievalMetricsCollector(
            resolved_settings
        )
        self.query_normalizer = query_normalizer or SemanticQueryNormalizer(
            resolved_settings
        )
        self.retrieval_router = retrieval_router or RetrievalRouter(resolved_settings)
        self.grounding_validator = grounding_validator or GroundingValidator(
            resolved_settings
        )

    def answer_question(
        self,
        question: UserQuestionInput | str,
    ) -> FinalAnswerResult:
        """
        Execute the full retrieval flow for one user question.

        Parameters
        ----------
        question : UserQuestionInput | str
            User question contract or plain string supplied by the caller.

        Returns
        -------
        FinalAnswerResult
            Final blocked, deflected, or answered retrieval result.
        """

        normalized_question = self.query_normalizer.normalize(
            self._normalize_question(question)
        )
        self._ensure_retrieval_enabled()

        self.metrics_collector.record_request_started()

        with self.metrics_collector.measure_stage("pre_guardrails"):
            pre_guardrail_decision = self.guardrails.evaluate_pre_request(
                normalized_question
            )

        self._record_guardrail_metrics(
            decision=pre_guardrail_decision,
            question=normalized_question,
        )

        if not pre_guardrail_decision.allowed:
            self.metrics_collector.record_request_outcome(
                blocked=True,
                deflected=pre_guardrail_decision.action == "deflect",
            )
            return self._build_guardrail_result(
                question=normalized_question,
                status=self._resolve_result_status(pre_guardrail_decision),
                answer_text=self._build_guardrail_message(pre_guardrail_decision),
                pre_guardrail=pre_guardrail_decision,
                post_guardrail=None,
                retrieval_context=None,
                grounded=False,
                answer_metadata={
                    "flow_stage": "pre_guardrails",
                    "guardrail_action": pre_guardrail_decision.action,
                },
            )

        with self.metrics_collector.measure_stage("retrieval_routing"):
            route_decision = self.retrieval_router.route(normalized_question)

        with self.metrics_collector.measure_stage("query_embedding"):
            query_vector = self._build_query_embedding(normalized_question)

        first_pass_result = self._execute_retrieval_pass(
            pass_name="first_pass",
            question=normalized_question,
            query_vector=query_vector,
            route_decision=route_decision,
        )
        retrieval_pass_result = self._select_retrieval_pass_result(
            first_pass_result=first_pass_result,
            second_pass_result=self._execute_second_retrieval_pass(
                question=normalized_question,
                query_vector=query_vector,
                first_pass_result=first_pass_result,
            ),
        )
        retrieved_chunks = retrieval_pass_result.retrieved_chunks
        retrieval_context = retrieval_pass_result.retrieval_context
        route_decision = retrieval_pass_result.route_decision
        retrieval_pass_metadata = self._build_retrieval_pass_metadata(
            selected_pass_result=retrieval_pass_result,
            first_pass_result=first_pass_result,
        )

        self.metrics_collector.record_retrieval_context(
            retrieval_context,
            expected_chunk_ids=self._read_optional_string_list(
                normalized_question.metadata,
                "expected_chunk_ids",
            ),
        )

        route_metadata = self._build_route_metadata(
            route_decision=route_decision,
            retrieval_context=retrieval_context,
            grounding_verification=None,
            question=normalized_question,
            retrieval_pass_metadata=retrieval_pass_metadata,
        )

        if not self._evidence_allows_generation(retrieval_context):
            self.metrics_collector.record_request_outcome(deflected=True)
            return self._build_guardrail_result(
                question=normalized_question,
                status="deflected",
                answer_text=self._build_evidence_deflection_message(
                    retrieval_context,
                ),
                pre_guardrail=pre_guardrail_decision,
                post_guardrail=None,
                retrieval_context=retrieval_context,
                grounded=False,
                answer_metadata=self._build_answer_metadata(
                    question=normalized_question,
                    generated_answer=None,
                    retrieved_chunks=retrieved_chunks,
                    retrieval_context=retrieval_context,
                    flow_stage="evidence_routing",
                    route_metadata=route_metadata,
                    response_mode="deflection",
                ),
                route_metadata=route_metadata,
            )

        with self.metrics_collector.measure_stage("answer_generation"):
            try:
                generated_answer = self.answer_generator.generate_answer(
                    self._build_answer_generation_input(
                        question=normalized_question,
                        retrieval_context=retrieval_context,
                        route_metadata=route_metadata,
                    )
                )
            except AnswerGenerationError as exc:
                LOGGER.warning(
                    "Answer generation failed and the retrieval service will "
                    "return a deflection. Provider='%s', error_type='%s', error='%s'.",
                    self.settings.response_generation_provider,
                    exc.__class__.__name__,
                    exc,
                )
                self.metrics_collector.record_request_outcome(deflected=True)
                return self._build_guardrail_result(
                    question=normalized_question,
                    status="deflected",
                    answer_text=self._build_answer_generation_deflection_message(),
                    pre_guardrail=pre_guardrail_decision,
                    post_guardrail=None,
                    retrieval_context=retrieval_context,
                    grounded=False,
                    answer_metadata=self._build_answer_metadata(
                        question=normalized_question,
                        generated_answer=None,
                        retrieved_chunks=retrieved_chunks,
                        retrieval_context=retrieval_context,
                        flow_stage="answer_generation",
                        route_metadata=route_metadata,
                        response_mode="deflection",
                    )
                    | {
                        "answer_generation_error": str(exc),
                        "answer_generation_error_type": exc.__class__.__name__,
                    },
                    route_metadata=route_metadata,
                )

        citations = self._build_citations(retrieval_context)

        with self.metrics_collector.measure_stage("grounding_validation"):
            grounding_verification = self.grounding_validator.validate(
                answer_text=generated_answer.answer_text,
                context=retrieval_context,
                citations=citations,
            )

        route_metadata = self._build_route_metadata(
            route_decision=route_decision,
            retrieval_context=retrieval_context,
            grounding_verification=grounding_verification,
            question=normalized_question,
            retrieval_pass_metadata=retrieval_pass_metadata,
        )

        if not grounding_verification.accepted:
            self.metrics_collector.record_request_outcome(deflected=True)
            return self._build_guardrail_result(
                question=normalized_question,
                status="deflected",
                answer_text=self._build_grounding_deflection_message(
                    grounding_verification,
                ),
                pre_guardrail=pre_guardrail_decision,
                post_guardrail=None,
                retrieval_context=retrieval_context,
                grounded=False,
                answer_metadata=self._build_answer_metadata(
                    question=normalized_question,
                    generated_answer=generated_answer,
                    retrieved_chunks=retrieved_chunks,
                    retrieval_context=retrieval_context,
                    flow_stage="grounding_validation",
                    route_metadata=route_metadata,
                    response_mode="deflection",
                ),
                route_metadata=route_metadata,
            )

        with self.metrics_collector.measure_stage("post_guardrails"):
            post_guardrail_decision = self.guardrails.evaluate_post_response(
                answer_text=generated_answer.answer_text,
                context=retrieval_context,
                grounded=generated_answer.grounded,
            )

        self._record_guardrail_metrics(
            decision=post_guardrail_decision,
            question=normalized_question,
        )

        if not post_guardrail_decision.allowed:
            self.metrics_collector.record_request_outcome(
                blocked=post_guardrail_decision.action == "block",
                deflected=post_guardrail_decision.action == "deflect",
            )
            return self._build_guardrail_result(
                question=normalized_question,
                status=self._resolve_result_status(post_guardrail_decision),
                answer_text=self._build_guardrail_message(post_guardrail_decision),
                pre_guardrail=pre_guardrail_decision,
                post_guardrail=post_guardrail_decision,
                retrieval_context=retrieval_context,
                grounded=False,
                answer_metadata=self._build_answer_metadata(
                    question=normalized_question,
                    generated_answer=generated_answer,
                    retrieved_chunks=retrieved_chunks,
                    retrieval_context=retrieval_context,
                    flow_stage="post_guardrails",
                    route_metadata=route_metadata,
                    response_mode="deflection",
                ),
                route_metadata=route_metadata,
            )

        response_mode = self._resolve_response_mode(retrieval_context)
        answer_text = self._apply_cautious_answer_policy(
            answer_text=generated_answer.answer_text,
            retrieval_context=retrieval_context,
            response_mode=response_mode,
        )
        self.metrics_collector.record_request_outcome(successful=True)

        return FinalAnswerResult(
            question=normalized_question,
            status="completed",
            answer_text=answer_text,
            grounded=generated_answer.grounded,
            retrieval_context=retrieval_context,
            pre_guardrail=pre_guardrail_decision,
            post_guardrail=post_guardrail_decision,
            citations=citations,
            answer_metadata=self._build_answer_metadata(
                question=normalized_question,
                generated_answer=generated_answer,
                retrieved_chunks=retrieved_chunks,
                retrieval_context=retrieval_context,
                flow_stage="completed",
                route_metadata=route_metadata,
                response_mode=response_mode,
            ),
            route_metadata=route_metadata,
            diagnostic_stage=route_metadata.diagnostic_stage,
            diagnostic_category=route_metadata.diagnostic_category,
            diagnostic_signals=list(route_metadata.diagnostic_signals),
            metrics_snapshot=self.metrics_collector.build_snapshot(),
        )

    def _normalize_question(
        self,
        question: UserQuestionInput | str,
    ) -> UserQuestionInput:
        """
        Normalize the public question input into the retrieval model contract.

        Parameters
        ----------
        question : UserQuestionInput | str
            Public question input supplied by the caller.

        Returns
        -------
        UserQuestionInput
            Normalized question contract used internally by the service.
        """

        if isinstance(question, UserQuestionInput):
            normalized_question = question
        elif isinstance(question, str):
            normalized_question = UserQuestionInput(question_text=question)
        else:
            raise ValueError("Question must be a UserQuestionInput instance or string.")

        if not normalized_question.question_text:
            raise ValueError("Question text cannot be empty.")

        return normalized_question

    def _ensure_retrieval_enabled(self) -> None:
        """
        Ensure retrieval execution is enabled in shared runtime settings.
        """

        if not self.settings.retrieval_enabled:
            raise ValueError("Retrieval is disabled in runtime settings.")

    def _build_query_embedding(
        self,
        question: UserQuestionInput,
    ) -> List[float]:
        """
        Generate one query embedding from the configured embedding provider.

        Parameters
        ----------
        question : UserQuestionInput
            Normalized question contract.

        Returns
        -------
        List[float]
            Numeric query vector used by the storage layer.
        """

        query_vectors = self.embedding_provider.embed_texts(
            [question.normalized_query_text]
        )
        if not query_vectors:
            raise ValueError("Query embedding provider returned no vectors.")

        query_vector = query_vectors[0]
        if not query_vector:
            raise ValueError("Query embedding provider returned an empty vector.")

        return [float(value) for value in query_vector]

    def _resolve_retrieval_breadth(
        self,
        route_decision: Optional[RetrievalRouteDecision] = None,
    ) -> int:
        """
        Resolve the storage-query breadth used before context selection.

        Returns
        -------
        int
            Positive retrieval breadth that keeps the storage query at least as
            broad as the configured candidate-pool size.
        """

        retrieval_top_k = int(self.settings.retrieval_top_k)
        candidate_pool_size = self._resolve_route_candidate_pool_size(route_decision)

        if retrieval_top_k <= 0:
            raise ValueError("retrieval_top_k must be greater than zero.")
        if candidate_pool_size <= 0:
            raise ValueError("retrieval_candidate_pool_size must be greater than zero.")

        return max(retrieval_top_k, candidate_pool_size)

    def _resolve_route_candidate_pool_size(
        self,
        route_decision: Optional[RetrievalRouteDecision],
    ) -> int:
        """
        Resolve candidate-pool breadth from routing metadata when available.

        Parameters
        ----------
        route_decision : Optional[RetrievalRouteDecision]
            Deterministic route emitted before vector retrieval.

        Returns
        -------
        int
            Positive candidate-pool breadth used for vector retrieval.
        """

        if isinstance(route_decision, RetrievalRouteDecision):
            candidate_pool_size = route_decision.metadata.get("candidate_pool_size")
            try:
                return int(candidate_pool_size)
            except (TypeError, ValueError):
                pass

        return int(self.settings.retrieval_candidate_pool_size)

    def _build_retrieval_filter(
        self,
        route_decision: Optional[RetrievalRouteDecision],
    ) -> Optional[Dict[str, Any]]:
        """
        Build a storage-layer metadata filter from scoped route targets.

        Parameters
        ----------
        route_decision : Optional[RetrievalRouteDecision]
            Deterministic route emitted before vector retrieval.

        Returns
        -------
        Optional[Dict[str, Any]]
            ChromaDB metadata filter when exact document ids are available.
        """

        if not isinstance(route_decision, RetrievalRouteDecision):
            return None
        if route_decision.retrieval_scope != "scoped":
            return None

        target_doc_ids = [
            doc_id.strip()
            for doc_id in route_decision.target_doc_ids
            if isinstance(doc_id, str) and doc_id.strip()
        ]
        if not target_doc_ids:
            return None
        if len(target_doc_ids) == 1:
            return {"doc_id": target_doc_ids[0]}
        return {"doc_id": {"$in": target_doc_ids}}

    def _query_similar_chunks(
        self,
        *,
        query_vector: List[float],
        top_k: int,
        retrieval_filter: Optional[Dict[str, Any]],
    ) -> List[RetrievedChunkResult]:
        """
        Query storage with an optional deterministic retrieval filter.

        Parameters
        ----------
        query_vector : List[float]
            Query embedding vector.

        top_k : int
            Number of candidates requested from storage.

        retrieval_filter : Optional[Dict[str, Any]]
            Optional storage metadata filter derived from routing.

        Returns
        -------
        List[RetrievedChunkResult]
            Retrieved candidate chunks.
        """

        if retrieval_filter:
            return self.storage.query_similar_chunks(
                query_vector=query_vector,
                top_k=top_k,
                where=retrieval_filter,
            )

        return self.storage.query_similar_chunks(
            query_vector=query_vector,
            top_k=top_k,
        )

    def _execute_retrieval_pass(
        self,
        *,
        pass_name: str,
        question: UserQuestionInput,
        query_vector: List[float],
        route_decision: RetrievalRouteDecision,
    ) -> _RetrievalPassResult:
        """
        Execute one vector retrieval pass and build its grounded context.

        Parameters
        ----------
        pass_name : str
            Stable pass identifier used for metadata and stage latency labels.

        question : UserQuestionInput
            Normalized user question used by context selection.

        query_vector : List[float]
            Query embedding vector reused across first and second passes.

        route_decision : RetrievalRouteDecision
            Route decision controlling breadth and optional storage filtering.

        Returns
        -------
        _RetrievalPassResult
            Retrieved candidates and selected context for this pass.
        """

        retrieval_breadth = self._resolve_retrieval_breadth(route_decision)
        retrieval_filter = self._build_retrieval_filter(route_decision)
        retrieval_stage_name = (
            "retrieval" if pass_name == "first_pass" else f"{pass_name}_retrieval"
        )
        context_stage_name = (
            "context_builder"
            if pass_name == "first_pass"
            else f"{pass_name}_context_builder"
        )

        with self.metrics_collector.measure_stage(retrieval_stage_name):
            retrieved_chunks = self._query_similar_chunks(
                query_vector=query_vector,
                top_k=retrieval_breadth,
                retrieval_filter=retrieval_filter,
            )

        with self.metrics_collector.measure_stage(context_stage_name):
            retrieval_context = self.context_builder.build_context(
                retrieved_chunks,
                top_k=retrieval_breadth,
                query_text=question.normalized_query_text,
                query_metadata=question.query_metadata,
                route_decision=route_decision,
            )

        return _RetrievalPassResult(
            pass_name=pass_name,
            route_decision=route_decision,
            retrieved_chunks=retrieved_chunks,
            retrieval_context=retrieval_context,
            retrieval_breadth=retrieval_breadth,
            retrieval_filter=retrieval_filter,
        )

    def _execute_second_retrieval_pass(
        self,
        *,
        question: UserQuestionInput,
        query_vector: List[float],
        first_pass_result: _RetrievalPassResult,
    ) -> Optional[_RetrievalPassResult]:
        """
        Execute a document-focused retry when first-pass evidence is weak.

        Parameters
        ----------
        question : UserQuestionInput
            Normalized user question.

        query_vector : List[float]
            Query embedding vector reused for retry retrieval.

        first_pass_result : _RetrievalPassResult
            First-pass retrieval output used to decide retry eligibility.

        Returns
        -------
        Optional[_RetrievalPassResult]
            Second-pass result when retry is justified, otherwise `None`.
        """

        if not self._should_attempt_second_pass(first_pass_result):
            return None

        target_doc_id = self._infer_second_pass_target_doc_id(first_pass_result)
        if not target_doc_id:
            return None

        second_pass_route = self._build_second_pass_route_decision(
            first_pass_result=first_pass_result,
            target_doc_id=target_doc_id,
        )

        return self._execute_retrieval_pass(
            pass_name="second_pass",
            question=question,
            query_vector=query_vector,
            route_decision=second_pass_route,
        )

    def _should_attempt_second_pass(
        self,
        first_pass_result: _RetrievalPassResult,
    ) -> bool:
        """
        Decide whether the service should retry retrieval before deflection.

        Parameters
        ----------
        first_pass_result : _RetrievalPassResult
            First-pass retrieval output.

        Returns
        -------
        bool
            `True` when settings, routing, and evidence state justify retry.
        """

        if not self.settings.retrieval_second_pass_retry_enabled:
            return False

        route_decision = first_pass_result.route_decision
        if not route_decision.allow_second_pass:
            return False

        retryable_scopes = {
            "broad",
            "broad_expanded",
            "retry_candidate_document_scoped",
        }
        if route_decision.retrieval_scope not in retryable_scopes:
            return False

        return self._context_needs_second_pass(first_pass_result.retrieval_context)

    def _context_needs_second_pass(
        self,
        retrieval_context: RetrievalContext,
    ) -> bool:
        """
        Determine whether selected evidence needs retry before final deflection.

        Parameters
        ----------
        retrieval_context : RetrievalContext
            First-pass context emitted by the context builder.

        Returns
        -------
        bool
            `True` for insufficient, empty, weak, or conflicting evidence.
        """

        if not self._evidence_allows_generation(retrieval_context):
            return True

        evidence_quality = retrieval_context.evidence_quality
        if evidence_quality is None:
            return not bool(retrieval_context.context_text.strip())

        if evidence_quality.strength in {"empty", "weak", "unknown"}:
            return True

        return evidence_quality.conflict == "conflicting"

    def _infer_second_pass_target_doc_id(
        self,
        first_pass_result: _RetrievalPassResult,
    ) -> str:
        """
        Infer a document target for second-pass retrieval without hardcoding.

        Parameters
        ----------
        first_pass_result : _RetrievalPassResult
            First-pass retrieval output.

        Returns
        -------
        str
            Dominant or routed document id when available, otherwise an empty
            string.
        """

        route_doc_id = self._read_first_string(
            first_pass_result.route_decision.metadata.get("inferred_target_doc_ids")
        )
        if route_doc_id:
            return route_doc_id

        route_doc_id = self._read_first_string(
            first_pass_result.route_decision.target_doc_ids
        )
        if route_doc_id:
            return route_doc_id

        primary_doc_id = self._infer_primary_anchor_doc_id(
            first_pass_result.retrieval_context
        )
        if primary_doc_id:
            return primary_doc_id

        return self._infer_dominant_candidate_doc_id(
            first_pass_result.retrieved_chunks
        )

    def _infer_primary_anchor_doc_id(
        self,
        retrieval_context: RetrievalContext,
    ) -> str:
        """
        Infer the document id associated with the selected primary anchor.

        Parameters
        ----------
        retrieval_context : RetrievalContext
            Context that may carry primary-anchor chunk metadata.

        Returns
        -------
        str
            Primary-anchor document id when recoverable.
        """

        primary_chunk_ids = self._read_string_list(
            retrieval_context.metadata.get("primary_anchor_chunk_ids")
        )
        if not primary_chunk_ids:
            return ""

        primary_chunk_id_set = set(primary_chunk_ids)
        for chunk in retrieval_context.chunks:
            if chunk.chunk_id in primary_chunk_id_set and chunk.doc_id:
                return chunk.doc_id

        return ""

    def _infer_dominant_candidate_doc_id(
        self,
        retrieved_chunks: List[RetrievedChunkResult],
    ) -> str:
        """
        Infer the dominant document id from first-pass candidate distribution.

        Parameters
        ----------
        retrieved_chunks : List[RetrievedChunkResult]
            Raw first-pass candidates returned by storage.

        Returns
        -------
        str
            Dominant document id when its share satisfies settings.
        """

        doc_counts: Dict[str, int] = {}
        for chunk in retrieved_chunks:
            if chunk.doc_id:
                doc_counts[chunk.doc_id] = doc_counts.get(chunk.doc_id, 0) + 1

        if not doc_counts:
            return ""

        ordered_doc_counts = sorted(
            doc_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
        top_doc_id, top_count = ordered_doc_counts[0]
        total_count = sum(doc_counts.values())
        dominant_share = top_count / total_count if total_count else 0.0

        if dominant_share >= float(
            self.settings.retrieval_second_pass_dominant_document_min_share
        ):
            return top_doc_id

        return ""

    def _build_second_pass_route_decision(
        self,
        *,
        first_pass_result: _RetrievalPassResult,
        target_doc_id: str,
    ) -> RetrievalRouteDecision:
        """
        Build a scoped route decision for document-focused retry retrieval.

        Parameters
        ----------
        first_pass_result : _RetrievalPassResult
            First-pass retrieval output.

        target_doc_id : str
            Document id inferred from route, context, or candidate dominance.

        Returns
        -------
        RetrievalRouteDecision
            Scoped route decision for the second retrieval pass.
        """

        first_route = first_pass_result.route_decision
        candidate_pool_size = max(
            int(self.settings.retrieval_top_k),
            int(self.settings.retrieval_second_pass_retry_candidate_pool_size),
        )

        return RetrievalRouteDecision(
            route_name="second_pass_document_scoped",
            retrieval_profile="second_pass_document_scoped",
            retrieval_scope="scoped",
            target_doc_ids=[target_doc_id],
            target_document_titles=list(first_route.target_document_titles),
            target_article_numbers=list(first_route.target_article_numbers),
            target_article_titles=list(first_route.target_article_titles),
            comparative=first_route.comparative,
            allow_second_pass=False,
            reasons=[
                "first_pass_evidence_retry_triggered",
                "second_pass_document_scope_selected",
            ],
            metadata={
                "candidate_pool_size": candidate_pool_size,
                "first_pass_route_name": first_route.route_name,
                "first_pass_retrieval_scope": first_route.retrieval_scope,
                "retry_target_doc_id": target_doc_id,
                "routing_enabled": True,
            },
        )

    def _select_retrieval_pass_result(
        self,
        *,
        first_pass_result: _RetrievalPassResult,
        second_pass_result: Optional[_RetrievalPassResult],
    ) -> _RetrievalPassResult:
        """
        Select the retrieval pass that should feed generation or deflection.

        Parameters
        ----------
        first_pass_result : _RetrievalPassResult
            Original retrieval output.

        second_pass_result : Optional[_RetrievalPassResult]
            Retry output when a retry was executed.

        Returns
        -------
        _RetrievalPassResult
            Selected pass result for the remaining service flow.
        """

        if second_pass_result is None:
            return first_pass_result

        if self._evidence_allows_generation(second_pass_result.retrieval_context):
            return second_pass_result

        if not self._evidence_allows_generation(first_pass_result.retrieval_context):
            return second_pass_result

        return first_pass_result

    def _build_retrieval_pass_metadata(
        self,
        *,
        selected_pass_result: _RetrievalPassResult,
        first_pass_result: _RetrievalPassResult,
    ) -> Dict[str, Any]:
        """
        Build detached metadata describing first-pass and selected-pass state.

        Parameters
        ----------
        selected_pass_result : _RetrievalPassResult
            Pass selected for the rest of the retrieval flow.

        first_pass_result : _RetrievalPassResult
            Original first-pass output.

        Returns
        -------
        Dict[str, Any]
            Serializable metadata for route and answer metadata payloads.
        """

        return {
            "selected_pass": selected_pass_result.pass_name,
            "second_pass_triggered": selected_pass_result.pass_name == "second_pass",
            "first_pass": self._summarize_retrieval_pass(first_pass_result),
            "selected": self._summarize_retrieval_pass(selected_pass_result),
        }

    def _summarize_retrieval_pass(
        self,
        retrieval_pass_result: _RetrievalPassResult,
    ) -> Dict[str, Any]:
        """
        Summarize one retrieval pass for deterministic metadata output.

        Parameters
        ----------
        retrieval_pass_result : _RetrievalPassResult
            Retrieval pass output to summarize.

        Returns
        -------
        Dict[str, Any]
            Compact serializable pass summary.
        """

        evidence_quality = retrieval_pass_result.retrieval_context.evidence_quality
        summary: Dict[str, Any] = {
            "pass_name": retrieval_pass_result.pass_name,
            "route_name": retrieval_pass_result.route_decision.route_name,
            "retrieval_scope": retrieval_pass_result.route_decision.retrieval_scope,
            "retrieval_breadth": retrieval_pass_result.retrieval_breadth,
            "retrieval_filter": dict(retrieval_pass_result.retrieval_filter or {}),
            "retrieved_chunk_count": len(retrieval_pass_result.retrieved_chunks),
            "context_chunk_count": retrieval_pass_result.retrieval_context.chunk_count,
        }

        if evidence_quality is not None:
            summary["evidence_strength"] = evidence_quality.strength
            summary["evidence_conflict"] = evidence_quality.conflict
            summary["sufficient_for_answer"] = evidence_quality.sufficient_for_answer

        return summary

    def _read_first_string(
        self,
        value: Any,
    ) -> str:
        """
        Read the first non-empty string from a value or list-like payload.

        Parameters
        ----------
        value : Any
            Candidate string or list of strings.

        Returns
        -------
        str
            First clean string value, otherwise an empty string.
        """

        if isinstance(value, str):
            return value.strip()

        for item in self._read_string_list(value):
            return item

        return ""

    def _read_string_list(
        self,
        value: Any,
    ) -> List[str]:
        """
        Normalize a list-like payload into clean string values.

        Parameters
        ----------
        value : Any
            Candidate list-like payload.

        Returns
        -------
        List[str]
            Ordered non-empty string values.
        """

        if not isinstance(value, list):
            return []

        normalized_values: List[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                normalized_values.append(item.strip())

        return normalized_values

    def _build_answer_generation_input(
        self,
        *,
        question: UserQuestionInput,
        retrieval_context: RetrievalContext,
        route_metadata: Optional[RetrievalRouteMetadata] = None,
    ) -> AnswerGenerationInput:
        """
        Build the answer-generation payload from question and grounded context.

        Parameters
        ----------
        question : UserQuestionInput
            Normalized user question.

        retrieval_context : RetrievalContext
            Context selected from retrieved chunks.

        route_metadata : Optional[RetrievalRouteMetadata]
            Route and evidence metadata attached to the generation request.

        Returns
        -------
        AnswerGenerationInput
            Grounded answer-generation payload.
        """

        return AnswerGenerationInput(
            question=question,
            context=retrieval_context,
            metadata={
                "request_id": question.request_id,
                "conversation_id": question.conversation_id,
                "normalized_query_text": question.normalized_query_text,
                "formatting_instructions": list(question.formatting_instructions),
                "query_metadata": dict(question.query_metadata),
            },
            grounding_instruction=self._build_grounding_instruction(
                retrieval_context,
            ),
            route_metadata=route_metadata,
        )

    def _build_route_metadata(
        self,
        *,
        route_decision: Optional[RetrievalRouteDecision],
        retrieval_context: Optional[RetrievalContext],
        grounding_verification: Optional[GroundingVerificationResult],
        question: UserQuestionInput,
        retrieval_pass_metadata: Optional[Dict[str, Any]] = None,
    ) -> RetrievalRouteMetadata:
        """
        Build the service-level routing metadata envelope.

        Parameters
        ----------
        route_decision : Optional[RetrievalRouteDecision]
            Pre-retrieval routing decision.

        retrieval_context : Optional[RetrievalContext]
            Context selected after retrieval.

        grounding_verification : Optional[GroundingVerificationResult]
            Post-generation grounding validation result.

        question : UserQuestionInput
            Normalized user question carrying optional benchmark metadata.

        retrieval_pass_metadata : Optional[Dict[str, Any]]
            Optional first-pass and selected-pass metadata.

        Returns
        -------
        RetrievalRouteMetadata
            Typed routing metadata attached to generation and final results.
        """

        benchmark_case_id = ""
        raw_benchmark_case_id = question.metadata.get("benchmark_case_id")
        if isinstance(raw_benchmark_case_id, str):
            benchmark_case_id = raw_benchmark_case_id.strip()

        diagnostic_stage, diagnostic_category, diagnostic_signals = (
            self._resolve_runtime_diagnostics(
                route_decision=route_decision,
                retrieval_context=retrieval_context,
                grounding_verification=grounding_verification,
            )
        )

        return RetrievalRouteMetadata(
            route_decision=route_decision,
            evidence_quality=(
                retrieval_context.evidence_quality
                if isinstance(retrieval_context, RetrievalContext)
                else None
            ),
            grounding_verification=grounding_verification,
            diagnostic_stage=diagnostic_stage,
            diagnostic_category=diagnostic_category,
            diagnostic_signals=diagnostic_signals,
            benchmark_case_id=benchmark_case_id,
            metadata={
                "request_id": question.request_id,
                "conversation_id": question.conversation_id,
                "retrieval_passes": dict(retrieval_pass_metadata or {}),
            },
        )

    def _evidence_allows_generation(
        self,
        retrieval_context: RetrievalContext,
    ) -> bool:
        """
        Decide whether selected evidence is sufficient for answer generation.

        Parameters
        ----------
        retrieval_context : RetrievalContext
            Context selected after vector retrieval and context building.

        Returns
        -------
        bool
            `True` when generation may proceed, otherwise `False`.
        """

        evidence_quality = retrieval_context.evidence_quality
        if evidence_quality is None:
            return bool(retrieval_context.context_text.strip())

        if not evidence_quality.sufficient_for_answer:
            return False

        return evidence_quality.conflict != "conflicting"

    def _build_grounding_instruction(
        self,
        retrieval_context: RetrievalContext,
    ) -> str:
        """
        Build a compact caution instruction from evidence-quality signals.

        Parameters
        ----------
        retrieval_context : RetrievalContext
            Selected context carrying evidence-quality classification.

        Returns
        -------
        str
            Additional grounding instruction for ambiguous but usable evidence.
        """

        evidence_quality = retrieval_context.evidence_quality
        if evidence_quality is None:
            return ""

        if evidence_quality.ambiguity == "ambiguous":
            return (
                "State uncertainty clearly when close legal competitors are present "
                "and cite the selected article or document explicitly."
            )

        return ""

    def _resolve_response_mode(
        self,
        retrieval_context: RetrievalContext,
    ) -> str:
        """
        Resolve the explicit response mode for the selected grounded context.

        Parameters
        ----------
        retrieval_context : RetrievalContext
            Final context selected for answer generation.

        Returns
        -------
        str
            Stable response-mode label used by the service result metadata.
        """

        evidence_quality = retrieval_context.evidence_quality
        if (
            evidence_quality is not None
            and evidence_quality.ambiguity == "ambiguous"
            and self.settings.retrieval_response_policy_cautious_answer_enabled
        ):
            return "cautious"
        return "confident"

    def _apply_cautious_answer_policy(
        self,
        *,
        answer_text: str,
        retrieval_context: RetrievalContext,
        response_mode: str,
    ) -> str:
        """
        Apply the service-level cautious-answer policy when ambiguity remains.

        Parameters
        ----------
        answer_text : str
            Answer text returned by the configured generator.

        retrieval_context : RetrievalContext
            Context used to ground the generated answer.

        response_mode : str
            Explicit service response mode derived from the selected evidence.

        Returns
        -------
        str
            Final answer text returned by the service.
        """

        if response_mode != "cautious":
            return answer_text

        clarification_prefix = self._build_cautious_answer_prefix(retrieval_context)
        if not clarification_prefix:
            return answer_text

        normalized_answer_text = answer_text.strip()
        if not normalized_answer_text:
            return clarification_prefix
        if normalized_answer_text.startswith(clarification_prefix):
            return normalized_answer_text
        return f"{clarification_prefix} {normalized_answer_text}"

    def _build_cautious_answer_prefix(
        self,
        retrieval_context: RetrievalContext,
    ) -> str:
        """
        Build one explicit caution prefix for ambiguous but usable evidence.

        Parameters
        ----------
        retrieval_context : RetrievalContext
            Context that still carries usable but ambiguous evidence.

        Returns
        -------
        str
            Prefix describing the cautious-answer policy for the current
            response.
        """

        evidence_quality = retrieval_context.evidence_quality
        if evidence_quality is None or evidence_quality.ambiguity != "ambiguous":
            return ""

        primary_anchor = ""
        metadata_primary_anchor = retrieval_context.metadata.get("primary_anchor")
        if isinstance(metadata_primary_anchor, str):
            primary_anchor = metadata_primary_anchor.strip()

        if primary_anchor:
            return (
                "The answer below follows the most likely governing legal anchor "
                f"({primary_anchor}), but close legal competitors remain."
            )

        return (
            "The answer below follows the most likely governing legal anchor, "
            "but close legal competitors remain."
        )

    def _resolve_runtime_diagnostics(
        self,
        *,
        route_decision: Optional[RetrievalRouteDecision],
        retrieval_context: Optional[RetrievalContext],
        grounding_verification: Optional[GroundingVerificationResult],
    ) -> tuple[str, str, List[DiagnosticSignal]]:
        """
        Resolve the shared diagnostic taxonomy for the current runtime state.

        Parameters
        ----------
        route_decision : Optional[RetrievalRouteDecision]
            Routing decision chosen before retrieval.

        retrieval_context : Optional[RetrievalContext]
            Context selected after retrieval and context building.

        grounding_verification : Optional[GroundingVerificationResult]
            Final grounding-validation result when available.

        Returns
        -------
        tuple[str, str, List[DiagnosticSignal]]
            Diagnostic stage, category, and explicit signals for downstream
            consumers.
        """

        if isinstance(grounding_verification, GroundingVerificationResult):
            return (
                grounding_verification.diagnostic_stage,
                grounding_verification.diagnostic_category,
                list(grounding_verification.diagnostic_signals),
            )

        if not isinstance(retrieval_context, RetrievalContext):
            return "", "", []

        evidence_quality = retrieval_context.evidence_quality
        if evidence_quality is None:
            if retrieval_context.context_text.strip():
                return "", "", []
            return (
                "retrieval",
                "evidence_insufficiency",
                [
                    DiagnosticSignal(
                        stage="retrieval",
                        category="evidence_insufficiency",
                        code="no_context_selected",
                        detail="No grounded context could be assembled.",
                    )
                ],
            )

        if evidence_quality.diagnostic_stage or evidence_quality.diagnostic_category:
            return (
                evidence_quality.diagnostic_stage,
                evidence_quality.diagnostic_category,
                list(evidence_quality.diagnostic_signals),
            )

        return self._build_evidence_diagnostics(
            route_decision=route_decision,
            retrieval_context=retrieval_context,
        )

    def _build_evidence_diagnostics(
        self,
        *,
        route_decision: Optional[RetrievalRouteDecision],
        retrieval_context: RetrievalContext,
    ) -> tuple[str, str, List[DiagnosticSignal]]:
        """
        Build diagnostic taxonomy from evidence-quality and routing signals.

        Parameters
        ----------
        route_decision : Optional[RetrievalRouteDecision]
            Routing decision used for the selected retrieval pass.

        retrieval_context : RetrievalContext
            Selected context carrying evidence-quality classification.

        Returns
        -------
        tuple[str, str, List[DiagnosticSignal]]
            Diagnostic stage, category, and explicit signals for the current
            retrieval state.
        """

        evidence_quality = retrieval_context.evidence_quality
        if evidence_quality is None:
            return "", "", []

        diagnostic_stage = "context_builder"
        diagnostic_signals: List[DiagnosticSignal] = []
        route_name = (
            route_decision.route_name
            if isinstance(route_decision, RetrievalRouteDecision)
            else ""
        )
        route_scope = (
            route_decision.retrieval_scope
            if isinstance(route_decision, RetrievalRouteDecision)
            else ""
        )

        if evidence_quality.conflict == "conflicting":
            diagnostic_signals.append(
                DiagnosticSignal(
                    stage=diagnostic_stage,
                    category="retrieval_failure",
                    code="wrong_primary_anchor_selected",
                    detail="Conflicting legal anchors remain in the selected context.",
                    chunk_ids=list(evidence_quality.conflicting_chunk_ids),
                    metadata={
                        "route_name": route_name,
                        "retrieval_scope": route_scope,
                        "primary_anchor": retrieval_context.metadata.get(
                            "primary_anchor",
                            "",
                        ),
                    },
                )
            )
            return diagnostic_stage, "retrieval_failure", diagnostic_signals

        if not evidence_quality.sufficient_for_answer:
            diagnostic_signals.append(
                DiagnosticSignal(
                    stage=diagnostic_stage,
                    category="evidence_insufficiency",
                    code="insufficient_legal_evidence",
                    detail="Selected evidence is not specific enough for a safe answer.",
                    chunk_ids=[
                        chunk.chunk_id
                        for chunk in retrieval_context.chunks
                        if chunk.chunk_id
                    ],
                    metadata={
                        "route_name": route_name,
                        "retrieval_scope": route_scope,
                        "evidence_strength": evidence_quality.strength,
                    },
                )
            )
            return diagnostic_stage, "evidence_insufficiency", diagnostic_signals

        if evidence_quality.ambiguity == "ambiguous":
            diagnostic_signals.append(
                DiagnosticSignal(
                    stage=diagnostic_stage,
                    category="cautious_answer",
                    code="close_legal_competitors_remain",
                    detail="Selected evidence is usable but still legally close to competitors.",
                    chunk_ids=list(evidence_quality.close_competitor_chunk_ids),
                    metadata={
                        "route_name": route_name,
                        "retrieval_scope": route_scope,
                    },
                )
            )
            return diagnostic_stage, "cautious_answer", diagnostic_signals

        diagnostic_signals.append(
            DiagnosticSignal(
                stage=diagnostic_stage,
                category="grounded_answer",
                code="primary_anchor_supported",
                detail="Selected evidence supports one governing legal anchor.",
                chunk_ids=self._read_string_list(
                    retrieval_context.metadata.get("primary_anchor_chunk_ids")
                ),
                metadata={
                    "route_name": route_name,
                    "retrieval_scope": route_scope,
                    "primary_anchor": retrieval_context.metadata.get(
                        "primary_anchor",
                        "",
                    ),
                },
            )
        )
        return diagnostic_stage, "grounded_answer", diagnostic_signals

    def _record_guardrail_metrics(
        self,
        *,
        decision: GuardrailDecision,
        question: UserQuestionInput,
    ) -> None:
        """
        Feed optional labeled guardrail information into the metrics collector.

        Parameters
        ----------
        decision : GuardrailDecision
            Guardrail decision produced by the current stage.

        question : UserQuestionInput
            Question carrying optional evaluation labels in metadata.
        """

        expected_safe = self._read_optional_bool(question.metadata, "expected_safe")
        expected_jailbreak = self._read_optional_bool(
            question.metadata,
            "expected_jailbreak",
        )
        self.metrics_collector.record_guardrail_decision(
            decision,
            expected_safe=expected_safe,
            expected_jailbreak=expected_jailbreak,
        )

    def _read_optional_bool(
        self,
        metadata: Dict[str, Any],
        key: str,
    ) -> Optional[bool]:
        """
        Read one optional boolean label from question metadata.

        Parameters
        ----------
        metadata : Dict[str, Any]
            Question metadata mapping.

        key : str
            Metadata key to read.

        Returns
        -------
        Optional[bool]
            Boolean label when explicitly provided, otherwise `None`.
        """

        value = metadata.get(key)
        if isinstance(value, bool):
            return value
        return None

    def _read_optional_string_list(
        self,
        metadata: Dict[str, Any],
        key: str,
    ) -> Optional[List[str]]:
        """
        Read one optional list of string labels from question metadata.

        Parameters
        ----------
        metadata : Dict[str, Any]
            Question metadata mapping.

        key : str
            Metadata key to read.

        Returns
        -------
        Optional[List[str]]
            Clean string labels when explicitly provided, otherwise `None`.
        """

        value = metadata.get(key)
        if not isinstance(value, list):
            return None

        normalized_values: List[str] = []

        for raw_value in value:
            if not isinstance(raw_value, str):
                continue

            normalized_value = raw_value.strip()
            if normalized_value:
                normalized_values.append(normalized_value)

        if not normalized_values:
            return None

        return normalized_values

    def _build_guardrail_result(
        self,
        *,
        question: UserQuestionInput,
        status: str,
        answer_text: str,
        pre_guardrail: Optional[GuardrailDecision],
        post_guardrail: Optional[GuardrailDecision],
        retrieval_context: Optional[RetrievalContext],
        grounded: bool,
        answer_metadata: Dict[str, Any],
        route_metadata: Optional[RetrievalRouteMetadata] = None,
    ) -> FinalAnswerResult:
        """
        Build one blocked or deflected final result.

        Parameters
        ----------
        question : UserQuestionInput
            Original normalized question.

        status : str
            Final result status.

        answer_text : str
            User-facing message returned by the service.

        pre_guardrail : Optional[GuardrailDecision]
            Pre-request guardrail decision, when available.

        post_guardrail : Optional[GuardrailDecision]
            Post-response guardrail decision, when available.

        retrieval_context : Optional[RetrievalContext]
            Retrieval context built before deflection, when available.

        grounded : bool
            Grounding flag attached to the final result.

        answer_metadata : Dict[str, Any]
            Result metadata emitted by the service.

        route_metadata : Optional[RetrievalRouteMetadata]
            Route, evidence, and grounding metadata attached to the result.

        Returns
        -------
        FinalAnswerResult
            Final blocked or deflected result.
        """

        diagnostic_stage = ""
        diagnostic_category = ""
        diagnostic_signals: List[DiagnosticSignal] = []

        if isinstance(route_metadata, RetrievalRouteMetadata):
            diagnostic_stage = route_metadata.diagnostic_stage
            diagnostic_category = route_metadata.diagnostic_category
            diagnostic_signals = list(route_metadata.diagnostic_signals)

        return FinalAnswerResult(
            question=question,
            status=status,
            answer_text=answer_text,
            grounded=grounded,
            retrieval_context=retrieval_context,
            pre_guardrail=pre_guardrail,
            post_guardrail=post_guardrail,
            citations=self._build_citations(retrieval_context),
            diagnostic_stage=diagnostic_stage,
            diagnostic_category=diagnostic_category,
            diagnostic_signals=diagnostic_signals,
            answer_metadata=answer_metadata,
            route_metadata=route_metadata,
            metrics_snapshot=self.metrics_collector.build_snapshot(),
        )

    def _resolve_result_status(
        self,
        decision: GuardrailDecision,
    ) -> str:
        """
        Map one guardrail decision to the final result status string.

        Parameters
        ----------
        decision : GuardrailDecision
            Guardrail decision emitted by one pipeline stage.

        Returns
        -------
        str
            Final status string stored in `FinalAnswerResult`.
        """

        if decision.action == "deflect":
            return "deflected"
        return "blocked"

    def _build_guardrail_message(
        self,
        decision: GuardrailDecision,
    ) -> str:
        """
        Build one user-facing message from a guardrail decision.

        Parameters
        ----------
        decision : GuardrailDecision
            Guardrail decision emitted by the service.

        Returns
        -------
        str
            User-facing block or deflection message.
        """

        if decision.action == "deflect":
            return (
                "A supported answer could not be returned safely. "
                f"{decision.reason}"
            ).strip()

        return (
            "The request or response was blocked by deterministic safety checks. "
            f"{decision.reason}"
        ).strip()

    def _build_answer_metadata(
        self,
        *,
        question: UserQuestionInput,
        generated_answer: Optional[GeneratedAnswer],
        retrieved_chunks: List[RetrievedChunkResult],
        retrieval_context: RetrievalContext,
        flow_stage: str,
        route_metadata: Optional[RetrievalRouteMetadata] = None,
        response_mode: str = "",
    ) -> Dict[str, Any]:
        """
        Build service-level answer metadata for the final retrieval result.

        Parameters
        ----------
        question : UserQuestionInput
            Question contract after deterministic semantic normalization.

        generated_answer : Optional[GeneratedAnswer]
            Generated answer payload returned by the answer adapter.

        retrieved_chunks : List[RetrievedChunkResult]
            Raw retrieved chunks returned by storage.

        retrieval_context : RetrievalContext
            Final grounded context passed to answer generation.

        flow_stage : str
            Service stage where the result was finalized.

        route_metadata : Optional[RetrievalRouteMetadata]
            Route, evidence, and grounding metadata attached to the result.

        response_mode : str
            Explicit service response mode for the final answer payload.

        Returns
        -------
        Dict[str, Any]
            Detached metadata dictionary for the final result.
        """

        answer_metadata = dict(generated_answer.metadata) if generated_answer else {}
        answer_metadata.update(
            {
                "flow_stage": flow_stage,
                "normalized_query_text": question.normalized_query_text,
                "formatting_instructions": list(question.formatting_instructions),
                "query_metadata": dict(question.query_metadata),
                "retrieved_chunk_count": len(retrieved_chunks),
                "retrieved_chunk_ids": [
                    chunk.chunk_id for chunk in retrieved_chunks if chunk.chunk_id
                ],
                "context_chunk_count": retrieval_context.chunk_count,
                "context_character_count": retrieval_context.character_count,
                "context_truncated": retrieval_context.truncated,
                "response_mode": response_mode or "deflection",
            }
        )
        if self.settings.metrics_retrieval_quality_enabled:
            answer_metadata["retrieval_quality"] = retrieval_context.retrieval_quality.to_dict()
        if route_metadata is not None:
            answer_metadata["route_metadata"] = route_metadata.to_dict()
        return answer_metadata

    def _build_evidence_deflection_message(
        self,
        retrieval_context: RetrievalContext,
    ) -> str:
        """
        Build a user-facing deflection message for insufficient evidence.

        Parameters
        ----------
        retrieval_context : RetrievalContext
            Context carrying evidence-quality classification.

        Returns
        -------
        str
            Deflection message that does not invent unsupported content.
        """

        evidence_quality = retrieval_context.evidence_quality
        if evidence_quality and evidence_quality.conflict == "conflicting":
            return (
                "A supported answer could not be returned because the retrieved "
                "legal evidence still points to competing governing anchors."
            )

        if (
            self.settings.retrieval_response_policy_clarification_enabled
            and retrieval_context.context_text.strip()
        ):
            return (
                "A supported answer could not be returned because the retrieved "
                "legal evidence is not specific enough yet. Clarify the target "
                "document, article, or legal scenario."
            )

        return (
            "A supported answer could not be returned because the retrieved "
            "legal evidence is insufficient."
        )

    def _build_grounding_deflection_message(
        self,
        grounding_verification: GroundingVerificationResult,
    ) -> str:
        """
        Build a user-facing deflection message for failed grounding validation.

        Parameters
        ----------
        grounding_verification : GroundingVerificationResult
            Deterministic grounding result explaining the failed alignment.

        Returns
        -------
        str
            Deflection message preserving the validation reason.
        """

        reasons = ", ".join(grounding_verification.reasons)
        if reasons:
            return (
                "A supported answer could not be returned because grounding "
                f"validation failed: {reasons}."
            )

        return (
            "A supported answer could not be returned because grounding validation "
            "failed."
        )

    def _build_answer_generation_deflection_message(self) -> str:
        """
        Build a user-facing deflection message for answer-generation failures.

        Returns
        -------
        str
            Deflection message that avoids exposing provider internals.
        """

        return (
            "A supported answer could not be returned because answer generation "
            "failed."
        )

    def _build_citations(
        self,
        retrieval_context: Optional[RetrievalContext],
    ) -> List[str]:
        """
        Build compact deterministic citations from the selected context.

        Parameters
        ----------
        retrieval_context : Optional[RetrievalContext]
            Retrieval context selected for the answer.

        Returns
        -------
        List[str]
            Ordered citation strings derived from the selected chunks.
        """

        if retrieval_context is None:
            return []

        citations: List[str] = []

        for chunk in retrieval_context.chunks:
            citation_parts: List[str] = []
            context_metadata = chunk.context_metadata

            source_label = (
                context_metadata.document_title
                or context_metadata.section_title
                or chunk.document_metadata.get("document_title")
                or chunk.chunk_metadata.get("section_title")
                or chunk.source_file
                or chunk.doc_id
                or chunk.chunk_id
            )
            if source_label:
                citation_parts.append(str(source_label))

            article_number = (
                context_metadata.article_number
                or chunk.chunk_metadata.get("article_number")
            )
            if article_number:
                citation_parts.append(f"article={article_number}")

            page_start = context_metadata.page_start or chunk.chunk_metadata.get(
                "page_start"
            )
            page_end = context_metadata.page_end or chunk.chunk_metadata.get("page_end")
            if page_start and page_end and page_start != page_end:
                citation_parts.append(f"pages={page_start}-{page_end}")
            elif page_start:
                citation_parts.append(f"page={page_start}")

            citation_text = " | ".join(citation_parts).strip()
            if citation_text:
                citations.append(citation_text)

        return citations


def create_retrieval_service(
    settings: Optional[PipelineSettings] = None,
) -> RetrievalService:
    """
    Build the retrieval service using the shared runtime settings.

    Parameters
    ----------
    settings : Optional[PipelineSettings]
        Shared runtime settings. Default settings are loaded when omitted.

    Returns
    -------
    RetrievalService
        Fully wired retrieval orchestration service.
    """

    return RetrievalService(settings=settings)
