from __future__ import annotations

import os
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, cast


class EmbeddingProviderError(RuntimeError):
    """
    Raised when the embedding provider cannot generate vectors successfully.
    """


@dataclass(slots=True)
class SentenceTransformersEmbeddingProvider:
    """
    Generate embedding vectors through the Sentence Transformers library.

    Parameters
    ----------
    model : str
        Embedding model name configured for the current run.

    batch_size : int
        Maximum number of texts sent in each provider request.
    """

    model: str
    batch_size: int
    _loaded_model: object | None = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        """
        Validate provider configuration after dataclass initialization.
        """

        normalized_model = self.model.strip()
        if not normalized_model:
            raise ValueError("Sentence Transformers embedding model cannot be empty.")
        self.model = normalized_model

        if self.batch_size <= 0:
            raise ValueError(
                "Sentence Transformers embedding batch size must be greater than zero."
            )

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
        model = self._get_model()
        vectors: List[List[float]] = []

        for batch_index, batch_texts in enumerate(
            self._yield_batches(normalized_texts),
            start=1,
        ):
            batch_vectors = self._embed_batch(
                model=model,
                batch_texts=batch_texts,
                batch_index=batch_index,
            )
            vectors.extend(batch_vectors)

        return vectors

    def _get_model(self) -> "SentenceTransformer":
        """
        Return the cached Sentence Transformer model for this provider instance.

        Returns
        -------
        SentenceTransformer
            Loaded Sentence Transformer model instance reused across batches.
        """

        if self._loaded_model is None:
            self._loaded_model = self._build_model()

        return cast("SentenceTransformer", self._loaded_model)

    def _build_model(self) -> "SentenceTransformer":
        """
        Build the Sentence Transformer model for the configured embedding run.

        Returns
        -------
        SentenceTransformer
            Loaded Sentence Transformer model instance.
        """

        self._configure_runtime_quiet_mode()
        sentence_transformer_class = self._load_sentence_transformer_class()

        try:
            return sentence_transformer_class(self.model, local_files_only=True)
        except Exception as exc:
            raise EmbeddingProviderError(
                "Sentence Transformers model initialization failed for "
                f"model '{self.model}'. {self._build_model_resolution_hint()} "
                f"Original error: {self._summarize_exception(exc)}"
            ) from exc

    def _configure_runtime_quiet_mode(self) -> None:
        """
        Configure third-party model loaders to suppress benign startup noise.
        """

        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("HF_HUB_VERBOSITY", "error")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

        self._set_huggingface_hub_logging_to_error()
        self._set_transformers_logging_to_error()

    def _set_huggingface_hub_logging_to_error(self) -> None:
        """
        Set Hugging Face Hub logging to errors only when the package supports it.
        """

        try:
            from huggingface_hub.utils import logging as huggingface_hub_logging
        except Exception:
            return

        set_verbosity_error = getattr(
            huggingface_hub_logging,
            "set_verbosity_error",
            None,
        )
        if callable(set_verbosity_error):
            set_verbosity_error()

    def _set_transformers_logging_to_error(self) -> None:
        """
        Set Transformers logging and progress bars to quiet embedding startup output.
        """

        try:
            from transformers.utils import logging as transformers_logging
        except Exception:
            return

        set_verbosity_error = getattr(transformers_logging, "set_verbosity_error", None)
        if callable(set_verbosity_error):
            set_verbosity_error()

        disable_progress_bar = getattr(transformers_logging, "disable_progress_bar", None)
        if callable(disable_progress_bar):
            disable_progress_bar()

    def _load_sentence_transformer_class(self) -> type["SentenceTransformer"]:
        """
        Import the Sentence Transformer class with concise runtime diagnostics.

        Returns
        -------
        type[SentenceTransformer]
            Loaded class used to initialize the configured embedding model.
        """

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingProviderError(
                "Package 'sentence-transformers' is required to use the "
                "Sentence Transformers embedding provider."
            ) from exc
        except Exception as exc:
            raise EmbeddingProviderError(
                self._build_runtime_dependency_error_message(exc)
            ) from exc

        return SentenceTransformer

    def _build_runtime_dependency_error_message(self, exc: Exception) -> str:
        """
        Build a concise error for runtime dependency failures during import.

        Parameters
        ----------
        exc : Exception
            Exception raised while importing Sentence Transformers.

        Returns
        -------
        str
            Human-readable error that explains the likely environment issue.
        """

        normalized_message = self._summarize_exception(exc).lower()

        if "libtorchcodec" in normalized_message or "libnppicc" in normalized_message:
            return (
                "Sentence Transformers could not start because the local "
                "PyTorch runtime is incompatible with the installed "
                "TorchCodec/CUDA libraries. "
                f"{self._build_runtime_fix_hint()} "
                f"Detected runtime packages: {self._build_runtime_versions_summary()}. "
                "Original error: "
                f"{self._summarize_exception(exc)}"
            )

        return (
            "Sentence Transformers could not be imported because one of its "
            "runtime dependencies failed to initialize. "
            f"Detected runtime packages: {self._build_runtime_versions_summary()}. "
            "Original error: "
            f"{self._summarize_exception(exc)}"
        )

    def _build_runtime_fix_hint(self) -> str:
        """
        Describe the most likely operator actions for runtime dependency failures.

        Returns
        -------
        str
            Short environment remediation hint.
        """

        installed_versions = self._detect_installed_runtime_versions()
        torch_version = installed_versions.get("torch", "<unknown>")

        if "+" in torch_version:
            return (
                "Install a compatible CPU-only PyTorch stack in the active "
                "virtual environment, or remove 'torchcodec' if this project "
                "does not need media decoding."
            )

        return (
            "Provide the missing CUDA/FFmpeg runtime libraries required by the "
            "installed packages, or switch the environment to a CPU-only "
            "PyTorch stack."
        )

    def _build_runtime_versions_summary(self) -> str:
        """
        Build a concise summary of installed runtime package versions.

        Returns
        -------
        str
            One-line package version summary for diagnostics.
        """

        installed_versions = self._detect_installed_runtime_versions()
        ordered_packages = (
            "torch",
            "torchcodec",
            "sentence-transformers",
            "transformers",
        )
        summary_parts = [
            f"{package}={installed_versions.get(package, '<not-installed>')}"
            for package in ordered_packages
        ]
        return ", ".join(summary_parts)

    def _detect_installed_runtime_versions(self) -> Dict[str, str]:
        """
        Read the installed versions of runtime packages involved in model loading.

        Returns
        -------
        Dict[str, str]
            Mapping between package names and installed versions.
        """

        package_versions: Dict[str, str] = {}

        for package_name in (
            "torch",
            "torchcodec",
            "sentence-transformers",
            "transformers",
        ):
            try:
                package_versions[package_name] = metadata.version(package_name)
            except metadata.PackageNotFoundError:
                package_versions[package_name] = "<not-installed>"
            except Exception:
                package_versions[package_name] = "<unavailable>"

        return package_versions

    def _build_model_resolution_hint(self) -> str:
        """
        Describe how the configured model should be made available locally.

        Returns
        -------
        str
            Short operational hint for model resolution failures.
        """

        if Path(self.model).exists():
            return "Ensure the configured local model directory is complete and readable."

        return (
            "Ensure the model is already available in the local Hugging Face "
            "cache or configure a readable local model directory."
        )

    def _summarize_exception(self, exc: Exception) -> str:
        """
        Compress one exception into a single deterministic message line.

        Parameters
        ----------
        exc : Exception
            Exception raised by a dependency or model loader.

        Returns
        -------
        str
            First meaningful line extracted from the exception text.
        """

        exception_text = str(exc).strip()
        if not exception_text:
            return type(exc).__name__

        for line in exception_text.splitlines():
            normalized_line = line.strip()
            if normalized_line:
                return normalized_line

        return type(exc).__name__

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
            Validated text list ready for model inference.
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
                raise ValueError(f"Embedding text at index {index} cannot be empty.")

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
        model: "SentenceTransformer",
        batch_texts: Sequence[str],
        batch_index: int,
    ) -> List[List[float]]:
        """
        Request one embedding batch from the Sentence Transformers model.

        Parameters
        ----------
        model : SentenceTransformer
            Loaded Sentence Transformer model instance.

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
            batch_vectors = model.encode(
                list(batch_texts),
                batch_size=self.batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise EmbeddingProviderError(
                "Sentence Transformers embedding request failed for "
                f"batch {batch_index} with model '{self.model}': {exc}"
            ) from exc

        return self._validate_batch_vectors(
            batch_vectors=batch_vectors,
            expected_count=len(batch_texts),
            batch_index=batch_index,
        )

    def _validate_batch_vectors(
        self,
        batch_vectors: object,
        expected_count: int,
        batch_index: int,
    ) -> List[List[float]]:
        """
        Validate one batch of generated vectors before returning them.

        Parameters
        ----------
        batch_vectors : object
            Raw vector payload returned by Sentence Transformers.

        expected_count : int
            Number of vectors expected for the current batch.

        batch_index : int
            One-based batch index used for deterministic error reporting.

        Returns
        -------
        List[List[float]]
            Ordered numeric vectors aligned with the original batch input.
        """

        if not hasattr(batch_vectors, "tolist"):
            raise EmbeddingProviderError(
                "Sentence Transformers embedding response did not include a "
                "tolist-compatible array."
            )

        raw_vectors = batch_vectors.tolist()
        if not isinstance(raw_vectors, list):
            raise EmbeddingProviderError(
                "Sentence Transformers embedding response did not produce a valid list."
            )

        if len(raw_vectors) != expected_count:
            raise EmbeddingProviderError(
                "Sentence Transformers embedding response size does not match "
                f"the requested batch size for batch {batch_index}."
            )

        validated_vectors: List[List[float]] = []

        for item_index, vector in enumerate(raw_vectors):
            validated_vectors.append(
                self._validate_vector(
                    vector=vector,
                    batch_index=batch_index,
                    item_index=item_index,
                )
            )

        return validated_vectors

    def _validate_vector(
        self,
        vector: object,
        batch_index: int,
        item_index: int,
    ) -> List[float]:
        """
        Validate one generated vector from a batch response.

        Parameters
        ----------
        vector : object
            Raw vector candidate returned by Sentence Transformers.

        batch_index : int
            One-based batch index used for deterministic error reporting.

        item_index : int
            Zero-based item index inside the current batch.

        Returns
        -------
        List[float]
            Validated dense numeric vector.
        """

        if not isinstance(vector, list) or not vector:
            raise EmbeddingProviderError(
                "Sentence Transformers embedding response item is not a valid "
                f"non-empty vector for batch {batch_index}, item {item_index}."
            )

        normalized_vector: List[float] = []

        for value in vector:
            if not isinstance(value, (int, float)):
                raise EmbeddingProviderError(
                    "Sentence Transformers embedding response item contains a "
                    "non-numeric value for "
                    f"batch {batch_index}, item {item_index}."
                )

            normalized_vector.append(float(value))

        return normalized_vector
