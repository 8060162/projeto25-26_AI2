from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from Chunking.config.settings import PipelineSettings
from embedding.provider_factory import EmbeddingProvider, create_embedding_provider
from embedding.storage import ChromaEmbeddingStorage
from retrieval.answer_generator import (
    AnswerGenerator,
    GeneratedAnswer,
    create_answer_generator,
)
from retrieval.context_builder import RetrievalContextBuilder
from retrieval.guardrails import DeterministicGuardrails
from retrieval.metrics import RetrievalMetricsCollector
from retrieval.models import (
    AnswerGenerationInput,
    FinalAnswerResult,
    GuardrailDecision,
    RetrievalContext,
    RetrievedChunkResult,
    UserQuestionInput,
)


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

    def __init__(
        self,
        settings: Optional[PipelineSettings] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        storage: Optional[ChromaEmbeddingStorage] = None,
        context_builder: Optional[RetrievalContextBuilder] = None,
        guardrails: Optional[DeterministicGuardrails] = None,
        answer_generator: Optional[AnswerGenerator] = None,
        metrics_collector: Optional[RetrievalMetricsCollector] = None,
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

        normalized_question = self._normalize_question(question)
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

        with self.metrics_collector.measure_stage("query_embedding"):
            query_vector = self._build_query_embedding(normalized_question)

        with self.metrics_collector.measure_stage("retrieval"):
            retrieved_chunks = self.storage.query_similar_chunks(
                query_vector=query_vector,
                top_k=self.settings.retrieval_top_k,
            )

        with self.metrics_collector.measure_stage("context_builder"):
            retrieval_context = self.context_builder.build_context(retrieved_chunks)

        with self.metrics_collector.measure_stage("answer_generation"):
            generated_answer = self.answer_generator.generate_answer(
                self._build_answer_generation_input(
                    question=normalized_question,
                    retrieval_context=retrieval_context,
                )
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
                    generated_answer=generated_answer,
                    retrieved_chunks=retrieved_chunks,
                    retrieval_context=retrieval_context,
                    flow_stage="post_guardrails",
                ),
            )

        self.metrics_collector.record_request_outcome(successful=True)

        return FinalAnswerResult(
            question=normalized_question,
            status="completed",
            answer_text=generated_answer.answer_text,
            grounded=generated_answer.grounded,
            retrieval_context=retrieval_context,
            pre_guardrail=pre_guardrail_decision,
            post_guardrail=post_guardrail_decision,
            citations=self._build_citations(retrieval_context),
            answer_metadata=self._build_answer_metadata(
                generated_answer=generated_answer,
                retrieved_chunks=retrieved_chunks,
                retrieval_context=retrieval_context,
                flow_stage="completed",
            ),
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

        query_vectors = self.embedding_provider.embed_texts([question.question_text])
        if not query_vectors:
            raise ValueError("Query embedding provider returned no vectors.")

        query_vector = query_vectors[0]
        if not query_vector:
            raise ValueError("Query embedding provider returned an empty vector.")

        return [float(value) for value in query_vector]

    def _build_answer_generation_input(
        self,
        *,
        question: UserQuestionInput,
        retrieval_context: RetrievalContext,
    ) -> AnswerGenerationInput:
        """
        Build the answer-generation payload from question and grounded context.

        Parameters
        ----------
        question : UserQuestionInput
            Normalized user question.

        retrieval_context : RetrievalContext
            Context selected from retrieved chunks.

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
            },
        )

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

        Returns
        -------
        FinalAnswerResult
            Final blocked or deflected result.
        """

        return FinalAnswerResult(
            question=question,
            status=status,
            answer_text=answer_text,
            grounded=grounded,
            retrieval_context=retrieval_context,
            pre_guardrail=pre_guardrail,
            post_guardrail=post_guardrail,
            citations=self._build_citations(retrieval_context),
            answer_metadata=answer_metadata,
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
        generated_answer: GeneratedAnswer,
        retrieved_chunks: List[RetrievedChunkResult],
        retrieval_context: RetrievalContext,
        flow_stage: str,
    ) -> Dict[str, Any]:
        """
        Build service-level answer metadata for the final retrieval result.

        Parameters
        ----------
        generated_answer : GeneratedAnswer
            Generated answer payload returned by the answer adapter.

        retrieved_chunks : List[RetrievedChunkResult]
            Raw retrieved chunks returned by storage.

        retrieval_context : RetrievalContext
            Final grounded context passed to answer generation.

        flow_stage : str
            Service stage where the result was finalized.

        Returns
        -------
        Dict[str, Any]
            Detached metadata dictionary for the final result.
        """

        answer_metadata = dict(generated_answer.metadata)
        answer_metadata.update(
            {
                "flow_stage": flow_stage,
                "retrieved_chunk_count": len(retrieved_chunks),
                "retrieved_chunk_ids": [
                    chunk.chunk_id for chunk in retrieved_chunks if chunk.chunk_id
                ],
                "context_chunk_count": retrieval_context.chunk_count,
                "context_character_count": retrieval_context.character_count,
                "context_truncated": retrieval_context.truncated,
            }
        )
        return answer_metadata

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

            source_label = (
                chunk.document_metadata.get("document_title")
                or chunk.chunk_metadata.get("section_title")
                or chunk.source_file
                or chunk.doc_id
                or chunk.chunk_id
            )
            if source_label:
                citation_parts.append(str(source_label))

            article_number = chunk.chunk_metadata.get("article_number")
            if article_number:
                citation_parts.append(f"article={article_number}")

            page_start = chunk.chunk_metadata.get("page_start")
            page_end = chunk.chunk_metadata.get("page_end")
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
