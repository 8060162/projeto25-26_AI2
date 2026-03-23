from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence


class EmbeddingProviderError(RuntimeError):
    """
    Raised when the embedding provider cannot generate vectors successfully.
    """


@dataclass(slots=True)
class OpenAIEmbeddingProvider:
    """
    Generate embedding vectors through the OpenAI embeddings API.

    Parameters
    ----------
    model : str
        Embedding model name configured for the current run.

    batch_size : int
        Maximum number of texts sent in each provider request.
    """

    model: str
    batch_size: int

    def __post_init__(self) -> None:
        """
        Validate provider configuration after dataclass initialization.
        """

        normalized_model = self.model.strip()
        if not normalized_model:
            raise ValueError("OpenAI embedding model cannot be empty.")
        self.model = normalized_model

        if self.batch_size <= 0:
            raise ValueError("OpenAI embedding batch size must be greater than zero.")

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        """
        Generate embedding vectors for the provided texts.

        Parameters
        ----------
        texts : Sequence[str]
            Ordered texts to be embedded.

        Returns
        -------
        List[List[float]]
            Generated vectors in the same order as the input texts.
        """

        if not texts:
            return []

        normalized_texts = self._normalize_texts(texts)
        client = self._build_client()
        vectors: List[List[float]] = []

        for batch_index, batch_texts in enumerate(
            self._yield_batches(normalized_texts),
            start=1,
        ):
            batch_vectors = self._embed_batch(
                client=client,
                batch_texts=batch_texts,
                batch_index=batch_index,
            )
            vectors.extend(batch_vectors)

        return vectors

    def _build_client(self) -> Any:
        """
        Build the OpenAI client using the API key from the environment.

        Returns
        -------
        Any
            OpenAI SDK client instance.
        """

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise EmbeddingProviderError(
                "Environment variable 'OPENAI_API_KEY' is required for OpenAI embeddings."
            )

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise EmbeddingProviderError(
                "Package 'openai' is required to use the OpenAI embedding provider."
            ) from exc

        return OpenAI(api_key=api_key)

    def _normalize_texts(self, texts: Sequence[str]) -> List[str]:
        """
        Validate and normalize the input texts before batching.

        Parameters
        ----------
        texts : Sequence[str]
            Raw texts provided by the embedding pipeline.

        Returns
        -------
        List[str]
            Validated text list ready for the API call.
        """

        normalized_texts: List[str] = []

        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValueError(
                    f"Embedding text at index {index} must be a string, "
                    f"received '{type(text).__name__}'."
                )

            normalized_text = text.strip()
            if not normalized_text:
                raise ValueError(
                    f"Embedding text at index {index} cannot be empty."
                )

            normalized_texts.append(normalized_text)

        return normalized_texts

    def _yield_batches(self, texts: Sequence[str]) -> Iterable[List[str]]:
        """
        Yield ordered text batches according to the configured batch size.

        Parameters
        ----------
        texts : Sequence[str]
            Validated texts to be embedded.
        """

        for start_index in range(0, len(texts), self.batch_size):
            yield list(texts[start_index : start_index + self.batch_size])

    def _embed_batch(
        self,
        client: Any,
        batch_texts: Sequence[str],
        batch_index: int,
    ) -> List[List[float]]:
        """
        Request one embedding batch from the OpenAI API.

        Parameters
        ----------
        client : Any
            OpenAI SDK client instance.

        batch_texts : Sequence[str]
            Ordered texts for the current batch.

        batch_index : int
            One-based batch index used for deterministic error reporting.

        Returns
        -------
        List[List[float]]
            Batch vectors in the same order as the input texts.
        """

        try:
            response = client.embeddings.create(
                model=self.model,
                input=list(batch_texts),
            )
        except Exception as exc:
            raise EmbeddingProviderError(
                "OpenAI embedding request failed for "
                f"batch {batch_index} with model '{self.model}': {exc}"
            ) from exc

        response_data = getattr(response, "data", None)
        if not isinstance(response_data, list):
            raise EmbeddingProviderError(
                "OpenAI embedding response did not include a valid 'data' list."
            )

        ordered_vectors = self._extract_ordered_vectors(
            response_data=response_data,
            expected_count=len(batch_texts),
            batch_index=batch_index,
        )

        return ordered_vectors

    def _extract_ordered_vectors(
        self,
        response_data: Sequence[Any],
        expected_count: int,
        batch_index: int,
    ) -> List[List[float]]:
        """
        Extract and validate vectors from one OpenAI response payload.

        Parameters
        ----------
        response_data : Sequence[Any]
            Raw embedding items returned by the SDK.

        expected_count : int
            Number of vectors expected for the current batch.

        batch_index : int
            One-based batch index used for deterministic error reporting.

        Returns
        -------
        List[List[float]]
            Ordered vectors aligned with the original batch input.
        """

        ordered_vectors: List[List[float] | None] = [None] * expected_count

        for item in response_data:
            item_index = getattr(item, "index", None)
            embedding = getattr(item, "embedding", None)

            if not isinstance(item_index, int):
                raise EmbeddingProviderError(
                    "OpenAI embedding response item is missing a valid integer index."
                )

            if item_index < 0 or item_index >= expected_count:
                raise EmbeddingProviderError(
                    "OpenAI embedding response item index is out of range for "
                    f"batch {batch_index}: {item_index}."
                )

            ordered_vectors[item_index] = self._validate_vector(
                embedding=embedding,
                batch_index=batch_index,
                item_index=item_index,
            )

        if any(vector is None for vector in ordered_vectors):
            raise EmbeddingProviderError(
                f"OpenAI embedding response for batch {batch_index} is incomplete."
            )

        return [vector for vector in ordered_vectors if vector is not None]

    def _validate_vector(
        self,
        embedding: Any,
        batch_index: int,
        item_index: int,
    ) -> List[float]:
        """
        Validate one returned embedding vector.

        Parameters
        ----------
        embedding : Any
            Raw vector returned by the SDK.

        batch_index : int
            One-based batch index used for deterministic error reporting.

        item_index : int
            Zero-based item index inside the current batch.

        Returns
        -------
        List[float]
            Validated embedding vector.
        """

        if not isinstance(embedding, list) or not embedding:
            raise EmbeddingProviderError(
                "OpenAI embedding response returned an invalid vector for "
                f"batch {batch_index}, item {item_index}."
            )

        validated_vector: List[float] = []

        for value in embedding:
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise EmbeddingProviderError(
                    "OpenAI embedding response returned a non-numeric vector value "
                    f"for batch {batch_index}, item {item_index}."
                )
            validated_vector.append(float(value))

        return validated_vector
