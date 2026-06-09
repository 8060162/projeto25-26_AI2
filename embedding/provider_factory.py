from __future__ import annotations

from typing import Callable, Dict, List, Optional, Protocol, Sequence

from Chunking.config.settings import PipelineSettings
from embedding.providers.openai_provider import OpenAIEmbeddingProvider
from embedding.providers.sentence_transformers_provider import (
    SentenceTransformersEmbeddingProvider,
)


class EmbeddingProvider(Protocol):
    """
    Structural contract implemented by embedding providers.

    The embedding pipeline depends only on the ability to convert ordered
    text inputs into ordered numeric vectors.
    """

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        """
        Generate vectors for the provided texts.

        Parameters
        ----------
        texts : Sequence[str]
            Ordered texts to be embedded.

        Returns
        -------
        List[List[float]]
            Generated vectors aligned with the input order.
        """


ProviderBuilder = Callable[[PipelineSettings], EmbeddingProvider]


def create_embedding_provider(
    settings: Optional[PipelineSettings] = None,
) -> EmbeddingProvider:
    """
    Build the embedding provider configured in runtime settings.

    Parameters
    ----------
    settings : Optional[PipelineSettings]
        Shared runtime settings. When omitted, default settings are loaded.

    Returns
    -------
    EmbeddingProvider
        Configured provider instance ready to generate embeddings.
    """

    resolved_settings = settings or PipelineSettings()
    provider_name = _resolve_provider_name(resolved_settings.embedding_provider)
    provider_builder = _get_provider_builders().get(provider_name)

    if provider_builder is None:
        supported_providers = ", ".join(sorted(_get_provider_builders()))
        raise ValueError(
            "Unsupported embedding provider configured in settings: "
            f"'{resolved_settings.embedding_provider}'. "
            f"Supported providers: {supported_providers}."
        )

    return provider_builder(resolved_settings)


def _get_provider_builders() -> Dict[str, ProviderBuilder]:
    """
    Return the internal registry of supported embedding provider builders.

    Returns
    -------
    Dict[str, ProviderBuilder]
        Mapping between normalized provider names and builder callables.
    """

    return {
        "openai": _build_openai_embedding_provider,
        "sentence_transformers": _build_sentence_transformers_embedding_provider,
    }


def _resolve_provider_name(provider_name: str) -> str:
    """
    Normalize the configured provider name before registry lookup.

    Parameters
    ----------
    provider_name : str
        Raw provider name from settings.

    Returns
    -------
    str
        Lowercase provider name without leading or trailing whitespace.
    """

    normalized_provider_name = provider_name.strip().lower()
    if not normalized_provider_name:
        raise ValueError("Embedding provider name cannot be empty.")

    return normalized_provider_name


def _build_openai_embedding_provider(
    settings: PipelineSettings,
) -> OpenAIEmbeddingProvider:
    """
    Build the OpenAI embedding provider from shared runtime settings.

    Parameters
    ----------
    settings : PipelineSettings
        Shared runtime settings.

    Returns
    -------
    OpenAIEmbeddingProvider
        OpenAI provider configured with the selected model and batch size.
    """

    return OpenAIEmbeddingProvider(
        model=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
    )


def _build_sentence_transformers_embedding_provider(
    settings: PipelineSettings,
) -> SentenceTransformersEmbeddingProvider:
    """
    Build the Sentence Transformers embedding provider from shared runtime settings.

    Parameters
    ----------
    settings : PipelineSettings
        Shared runtime settings.

    Returns
    -------
    SentenceTransformersEmbeddingProvider
        Sentence Transformers provider configured with the selected model and
        batch size.
    """

    return SentenceTransformersEmbeddingProvider(
        model=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
    )
