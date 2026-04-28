from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, TypeVar

from Chunking.config.settings import PipelineSettings
from retrieval.evaluation.models import (
    BenchmarkGuardrailCase,
    BenchmarkQuestionCase,
)


BenchmarkCase = TypeVar("BenchmarkCase", BenchmarkQuestionCase, BenchmarkGuardrailCase)


class BenchmarkLoaderError(ValueError):
    """
    Signal invalid benchmark input with deterministic file and line context.
    """


@dataclass(slots=True)
class BenchmarkDatasets:
    """
    Group the currently versioned benchmark datasets loaded from JSONL files.
    """

    question_cases: List[BenchmarkQuestionCase]
    guardrail_cases: List[BenchmarkGuardrailCase]


class BenchmarkLoader:
    """
    Load and validate versioned benchmark datasets for offline evaluation.
    """

    _QUESTION_REQUIRED_FIELDS = (
        "case_id",
        "question",
        "case_type",
        "expected_route",
        "expected_doc_id",
        "expected_article_numbers",
        "expected_chunk_ids",
        "required_facts",
        "forbidden_facts",
        "expected_answer_behavior",
        "grounding_labels",
    )
    _QUESTION_LIST_FIELDS = (
        "expected_article_numbers",
        "expected_chunk_ids",
        "required_facts",
        "forbidden_facts",
    )
    _QUESTION_MAPPING_FIELDS = ("expected_route", "grounding_labels")
    _GUARDRAIL_REQUIRED_FIELDS = (
        "case_id",
        "question",
        "category",
        "expected_action",
        "expected_safe",
        "expected_route",
    )

    def __init__(
        self,
        settings: Optional[PipelineSettings] = None,
        questions_path: Optional[Path] = None,
        guardrails_path: Optional[Path] = None,
    ) -> None:
        """
        Create a benchmark loader using explicit paths or configured defaults.

        Parameters
        ----------
        settings : Optional[PipelineSettings]
            Shared project settings. Default settings are loaded when omitted.

        questions_path : Optional[Path]
            Optional override for the legal QA benchmark JSONL file.

        guardrails_path : Optional[Path]
            Optional override for the guardrail benchmark JSONL file.
        """

        resolved_settings = settings or PipelineSettings()
        self.questions_path = Path(
            questions_path or resolved_settings.benchmark_questions_path
        )
        self.guardrails_path = Path(
            guardrails_path or resolved_settings.benchmark_guardrails_path
        )

    def load_question_cases(self) -> List[BenchmarkQuestionCase]:
        """
        Load typed factual legal QA benchmark cases.

        Returns
        -------
        List[BenchmarkQuestionCase]
            Ordered question cases from the configured JSONL dataset.
        """

        return _load_jsonl_cases(
            dataset_path=self.questions_path,
            dataset_name="question benchmark",
            validator=self._validate_question_record,
            factory=BenchmarkQuestionCase.from_mapping,
        )

    def load_guardrail_cases(self) -> List[BenchmarkGuardrailCase]:
        """
        Load typed deterministic guardrail benchmark cases.

        Returns
        -------
        List[BenchmarkGuardrailCase]
            Ordered guardrail cases from the configured JSONL dataset.
        """

        return _load_jsonl_cases(
            dataset_path=self.guardrails_path,
            dataset_name="guardrail benchmark",
            validator=self._validate_guardrail_record,
            factory=BenchmarkGuardrailCase.from_mapping,
        )

    def load_datasets(self) -> BenchmarkDatasets:
        """
        Load all benchmark datasets currently required by evaluation modules.

        Returns
        -------
        BenchmarkDatasets
            Typed question and guardrail benchmark cases.
        """

        return BenchmarkDatasets(
            question_cases=self.load_question_cases(),
            guardrail_cases=self.load_guardrail_cases(),
        )

    def _validate_question_record(
        self,
        record: Dict[str, Any],
        dataset_path: Path,
        line_number: int,
    ) -> None:
        """
        Validate one raw factual legal QA benchmark record.

        Parameters
        ----------
        record : Dict[str, Any]
            Raw JSONL object to validate.

        dataset_path : Path
            Source dataset path used in error messages.

        line_number : int
            One-based JSONL line number used in error messages.
        """

        _require_fields(
            record=record,
            required_fields=self._QUESTION_REQUIRED_FIELDS,
            dataset_path=dataset_path,
            line_number=line_number,
        )
        _require_non_empty_string(record, "case_id", dataset_path, line_number)
        _require_non_empty_string(record, "question", dataset_path, line_number)
        _require_non_empty_string(record, "case_type", dataset_path, line_number)
        _require_non_empty_string(
            record,
            "expected_answer_behavior",
            dataset_path,
            line_number,
        )

        for field_name in self._QUESTION_LIST_FIELDS:
            _require_string_list(record, field_name, dataset_path, line_number)

        for field_name in self._QUESTION_MAPPING_FIELDS:
            _require_mapping(record, field_name, dataset_path, line_number)

        if record["expected_doc_id"] is not None:
            _require_non_empty_string(
                record,
                "expected_doc_id",
                dataset_path,
                line_number,
            )

    def _validate_guardrail_record(
        self,
        record: Dict[str, Any],
        dataset_path: Path,
        line_number: int,
    ) -> None:
        """
        Validate one raw deterministic guardrail benchmark record.

        Parameters
        ----------
        record : Dict[str, Any]
            Raw JSONL object to validate.

        dataset_path : Path
            Source dataset path used in error messages.

        line_number : int
            One-based JSONL line number used in error messages.
        """

        _require_fields(
            record=record,
            required_fields=self._GUARDRAIL_REQUIRED_FIELDS,
            dataset_path=dataset_path,
            line_number=line_number,
        )
        _require_non_empty_string(record, "case_id", dataset_path, line_number)
        _require_non_empty_string(record, "question", dataset_path, line_number)
        _require_non_empty_string(record, "category", dataset_path, line_number)
        _require_non_empty_string(record, "expected_action", dataset_path, line_number)
        _require_non_empty_string(record, "expected_route", dataset_path, line_number)

        if not isinstance(record["expected_safe"], bool):
            _raise_record_error(
                dataset_path,
                line_number,
                "field 'expected_safe' must be a boolean",
            )

        if "notes" in record and not isinstance(record["notes"], dict):
            _raise_record_error(
                dataset_path,
                line_number,
                "field 'notes' must be an object when present",
            )


def load_benchmark_question_cases(
    settings: Optional[PipelineSettings] = None,
    questions_path: Optional[Path] = None,
) -> List[BenchmarkQuestionCase]:
    """
    Load factual legal QA benchmark cases through the shared loader.

    Parameters
    ----------
    settings : Optional[PipelineSettings]
        Shared project settings. Default settings are loaded when omitted.

    questions_path : Optional[Path]
        Optional explicit dataset path.

    Returns
    -------
    List[BenchmarkQuestionCase]
        Ordered typed question cases.
    """

    return BenchmarkLoader(
        settings=settings,
        questions_path=questions_path,
    ).load_question_cases()


def load_benchmark_guardrail_cases(
    settings: Optional[PipelineSettings] = None,
    guardrails_path: Optional[Path] = None,
) -> List[BenchmarkGuardrailCase]:
    """
    Load deterministic guardrail benchmark cases through the shared loader.

    Parameters
    ----------
    settings : Optional[PipelineSettings]
        Shared project settings. Default settings are loaded when omitted.

    guardrails_path : Optional[Path]
        Optional explicit dataset path.

    Returns
    -------
    List[BenchmarkGuardrailCase]
        Ordered typed guardrail cases.
    """

    return BenchmarkLoader(
        settings=settings,
        guardrails_path=guardrails_path,
    ).load_guardrail_cases()


def load_benchmark_datasets(
    settings: Optional[PipelineSettings] = None,
    questions_path: Optional[Path] = None,
    guardrails_path: Optional[Path] = None,
) -> BenchmarkDatasets:
    """
    Load all versioned benchmark datasets through the shared loader.

    Parameters
    ----------
    settings : Optional[PipelineSettings]
        Shared project settings. Default settings are loaded when omitted.

    questions_path : Optional[Path]
        Optional explicit question dataset path.

    guardrails_path : Optional[Path]
        Optional explicit guardrail dataset path.

    Returns
    -------
    BenchmarkDatasets
        Typed benchmark datasets.
    """

    return BenchmarkLoader(
        settings=settings,
        questions_path=questions_path,
        guardrails_path=guardrails_path,
    ).load_datasets()


def _load_jsonl_cases(
    dataset_path: Path,
    dataset_name: str,
    validator: Callable[[Dict[str, Any], Path, int], None],
    factory: Callable[[Dict[str, Any]], BenchmarkCase],
) -> List[BenchmarkCase]:
    """
    Load one JSONL dataset and convert each validated record into a typed case.

    Parameters
    ----------
    dataset_path : Path
        Source JSONL dataset path.

    dataset_name : str
        Human-readable dataset name used in error messages.

    validator : Callable[[Dict[str, Any], Path, int], None]
        Function that validates one raw JSON object before model construction.

    factory : Callable[[Dict[str, Any]], BenchmarkCase]
        Model factory that converts a validated mapping into a typed case.

    Returns
    -------
    List[BenchmarkCase]
        Ordered typed benchmark cases.
    """

    if not dataset_path.exists():
        raise BenchmarkLoaderError(
            f"Cannot load {dataset_name}: file not found at '{dataset_path}'."
        )

    if not dataset_path.is_file():
        raise BenchmarkLoaderError(
            f"Cannot load {dataset_name}: path is not a file at '{dataset_path}'."
        )

    cases: List[BenchmarkCase] = []

    try:
        with dataset_path.open("r", encoding="utf-8") as benchmark_file:
            for line_number, raw_line in enumerate(benchmark_file, start=1):
                stripped_line = raw_line.strip()

                if not stripped_line:
                    continue

                record = _parse_jsonl_record(
                    raw_line=stripped_line,
                    dataset_path=dataset_path,
                    line_number=line_number,
                )
                validator(record, dataset_path, line_number)
                cases.append(factory(record))
    except OSError as exc:
        raise BenchmarkLoaderError(
            f"Cannot load {dataset_name} from '{dataset_path}': {exc}"
        ) from exc

    if not cases:
        raise BenchmarkLoaderError(
            f"Cannot load {dataset_name}: file '{dataset_path}' contains no cases."
        )

    return cases


def _parse_jsonl_record(
    raw_line: str,
    dataset_path: Path,
    line_number: int,
) -> Dict[str, Any]:
    """
    Parse and validate the JSON object shape for one JSONL line.

    Parameters
    ----------
    raw_line : str
        Raw non-empty JSONL line.

    dataset_path : Path
        Source dataset path used in error messages.

    line_number : int
        One-based JSONL line number used in error messages.

    Returns
    -------
    Dict[str, Any]
        Parsed JSON object.
    """

    try:
        record = json.loads(raw_line)
    except json.JSONDecodeError as exc:
        _raise_record_error(dataset_path, line_number, f"invalid JSON: {exc}")

    if not isinstance(record, dict):
        _raise_record_error(dataset_path, line_number, "line must be a JSON object")

    return record


def _require_fields(
    record: Dict[str, Any],
    required_fields: Sequence[str],
    dataset_path: Path,
    line_number: int,
) -> None:
    """
    Ensure all required fields are present in one benchmark record.

    Parameters
    ----------
    record : Dict[str, Any]
        Raw JSON object to validate.

    required_fields : Sequence[str]
        Field names required by the dataset contract.

    dataset_path : Path
        Source dataset path used in error messages.

    line_number : int
        One-based JSONL line number used in error messages.
    """

    missing_fields = [
        field_name for field_name in required_fields if field_name not in record
    ]

    if missing_fields:
        _raise_record_error(
            dataset_path,
            line_number,
            f"missing required fields: {', '.join(missing_fields)}",
        )


def _require_non_empty_string(
    record: Dict[str, Any],
    field_name: str,
    dataset_path: Path,
    line_number: int,
) -> None:
    """
    Ensure one required field is a non-empty string.

    Parameters
    ----------
    record : Dict[str, Any]
        Raw JSON object to validate.

    field_name : str
        Required string field name.

    dataset_path : Path
        Source dataset path used in error messages.

    line_number : int
        One-based JSONL line number used in error messages.
    """

    value = record.get(field_name)

    if not isinstance(value, str) or not value.strip():
        _raise_record_error(
            dataset_path,
            line_number,
            f"field '{field_name}' must be a non-empty string",
        )


def _require_string_list(
    record: Dict[str, Any],
    field_name: str,
    dataset_path: Path,
    line_number: int,
) -> None:
    """
    Ensure one required field is a list containing only strings.

    Parameters
    ----------
    record : Dict[str, Any]
        Raw JSON object to validate.

    field_name : str
        Required string-list field name.

    dataset_path : Path
        Source dataset path used in error messages.

    line_number : int
        One-based JSONL line number used in error messages.
    """

    value = record.get(field_name)

    if not isinstance(value, list):
        _raise_record_error(
            dataset_path,
            line_number,
            f"field '{field_name}' must be a list",
        )

    for item_index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            _raise_record_error(
                dataset_path,
                line_number,
                f"field '{field_name}' item {item_index} must be a non-empty string",
            )


def _require_mapping(
    record: Dict[str, Any],
    field_name: str,
    dataset_path: Path,
    line_number: int,
) -> None:
    """
    Ensure one required field is a JSON object.

    Parameters
    ----------
    record : Dict[str, Any]
        Raw JSON object to validate.

    field_name : str
        Required mapping field name.

    dataset_path : Path
        Source dataset path used in error messages.

    line_number : int
        One-based JSONL line number used in error messages.
    """

    if not isinstance(record.get(field_name), dict):
        _raise_record_error(
            dataset_path,
            line_number,
            f"field '{field_name}' must be an object",
        )


def _raise_record_error(
    dataset_path: Path,
    line_number: int,
    reason: str,
) -> None:
    """
    Raise a loader error with deterministic source location context.

    Parameters
    ----------
    dataset_path : Path
        Source dataset path used in error messages.

    line_number : int
        One-based JSONL line number used in error messages.

    reason : str
        Validation failure reason.
    """

    raise BenchmarkLoaderError(f"{dataset_path}:{line_number}: {reason}.")
