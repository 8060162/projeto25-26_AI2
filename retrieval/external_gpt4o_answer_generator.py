from __future__ import annotations

import ast
import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from retrieval.answer_generator import (
    AnswerGenerationError,
    GeneratedAnswer,
    OpenAIAnswerGenerator,
)
from retrieval.models import AnswerGenerationInput


RequestSender = Callable[[Request, float], Any]
LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ExternalGPT4oAnswerGenerator:
    """
    Generate grounded answers through the configured external GPT-4o endpoint.

    Parameters
    ----------
    endpoint_url : str
        External HTTP endpoint URL that accepts multipart grounded-generation
        requests.

    auth_env_var : str
        Environment variable containing the endpoint authentication secret.

    question_field_name : str
        Multipart field name used for the user message.

    context_field_name : str
        Multipart field name used for the grounded context.

    instructions_field_name : str
        Multipart field name used for generation instructions.

    metadata_field_name : str
        Multipart field name used for JSON request metadata.

    channel_id : str
        Optional configured channel identifier required by some external
        endpoint contracts.

    thread_id : str
        Optional configured thread identifier required by some external
        endpoint contracts.

    user_info : str
        JSON string sent as the mandatory user-info field when required by
        the external endpoint contract.

    user_id : str
        Optional configured user identifier sent to endpoint contracts that
        support it.

    user_name : str
        Optional configured user display name sent to endpoint contracts that
        support it.

    auth_header_name : str
        HTTP header name used to send the endpoint authentication secret.

    auth_header_prefix : str
        Optional prefix prepended to the authentication secret.

    timeout_seconds : float
        Request timeout in seconds.

    max_retries : int
        Number of retry attempts for transient transport failures.

    grounded_fallback_enabled : bool
        Whether a deterministic fallback answer should be returned when no
        grounded context is available.

    request_sender : Optional[RequestSender]
        Optional transport function used by tests to avoid real network calls.
    """

    endpoint_url: str
    auth_env_var: str = "EXTERNAL_GPT4O_API_KEY"
    question_field_name: str = "question"
    context_field_name: str = "context"
    instructions_field_name: str = "instructions"
    metadata_field_name: str = "metadata"
    channel_id: str = ""
    thread_id: str = ""
    user_info: str = "{}"
    user_id: str = ""
    user_name: str = ""
    timeout_seconds: float = 30.0
    max_retries: int = 2
    grounded_fallback_enabled: bool = True
    auth_header_name: str = "Authorization"
    auth_header_prefix: str = "Bearer"
    request_sender: Optional[RequestSender] = None
    _prompt_builder: OpenAIAnswerGenerator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """
        Validate adapter configuration after dataclass initialization.
        """

        self.endpoint_url = self.endpoint_url.strip()
        self.auth_env_var = self.auth_env_var.strip()
        self.question_field_name = self.question_field_name.strip()
        self.context_field_name = self.context_field_name.strip()
        self.instructions_field_name = self.instructions_field_name.strip()
        self.metadata_field_name = self.metadata_field_name.strip()
        self.channel_id = self.channel_id.strip()
        self.thread_id = self.thread_id.strip()
        self.user_info = self.user_info.strip() or "{}"
        self.user_id = self.user_id.strip()
        self.user_name = self.user_name.strip()
        self.auth_header_name = self.auth_header_name.strip()
        self.auth_header_prefix = self.auth_header_prefix.strip()
        self.timeout_seconds = max(0.1, float(self.timeout_seconds))
        self.max_retries = max(0, int(self.max_retries))
        self._prompt_builder = OpenAIAnswerGenerator(
            model="external-gpt-4o",
            grounded_fallback_enabled=self.grounded_fallback_enabled,
        )

        required_values = {
            "External GPT-4o endpoint URL": self.endpoint_url,
            "External GPT-4o auth env var": self.auth_env_var,
            "External GPT-4o question field name": self.question_field_name,
            "External GPT-4o context field name": self.context_field_name,
            "External GPT-4o instructions field name": self.instructions_field_name,
            "External GPT-4o metadata field name": self.metadata_field_name,
            "External GPT-4o auth header name": self.auth_header_name,
        }

        missing_names = [
            field_label for field_label, field_value in required_values.items()
            if not field_value
        ]
        if missing_names:
            raise ValueError(", ".join(missing_names) + " cannot be empty.")

    def generate_answer(
        self,
        generation_input: AnswerGenerationInput,
    ) -> GeneratedAnswer:
        """
        Generate one grounded answer using the external GPT-4o endpoint.

        Parameters
        ----------
        generation_input : AnswerGenerationInput
            Normalized grounded generation payload.

        Returns
        -------
        GeneratedAnswer
            Answer text plus adapter metadata for retrieval orchestration.
        """

        self._validate_generation_input(generation_input)

        if (
            self.grounded_fallback_enabled
            and not generation_input.context.context_text
        ):
            return self._build_grounded_fallback_answer(generation_input)

        request = self._build_request(generation_input)
        response_body, response_status = self._send_request(request)
        response_payload = self._parse_response_payload(response_body)
        self._raise_for_application_error(response_payload)
        answer_text = self._extract_answer_text(response_payload)

        if not answer_text:
            raise AnswerGenerationError(
                "External GPT-4o answer-generation response did not include answer text."
            )

        return GeneratedAnswer(
            answer_text=answer_text,
            grounded=bool(generation_input.context.context_text),
            metadata={
                "provider": "external_gpt4o",
                "endpoint_url": self.endpoint_url,
                "status_code": response_status,
                "used_grounded_fallback": False,
                "context_chunk_count": generation_input.context.chunk_count,
                "context_chunk_ids": [
                    chunk.chunk_id for chunk in generation_input.context.chunks
                ],
                "usage": self._extract_usage(response_payload),
            },
        )

    def _validate_generation_input(
        self,
        generation_input: AnswerGenerationInput,
    ) -> None:
        """
        Validate the required answer-generation input fields.

        Parameters
        ----------
        generation_input : AnswerGenerationInput
            Candidate generation payload supplied by the caller.
        """

        if not isinstance(generation_input, AnswerGenerationInput):
            raise ValueError(
                "Generation input must be an AnswerGenerationInput instance."
            )

        if not generation_input.question.question_text:
            raise ValueError("Question text cannot be empty.")

    def _build_grounded_fallback_answer(
        self,
        generation_input: AnswerGenerationInput,
    ) -> GeneratedAnswer:
        """
        Build a deterministic fallback answer without calling the endpoint.

        Parameters
        ----------
        generation_input : AnswerGenerationInput
            Normalized generation payload with missing context.

        Returns
        -------
        GeneratedAnswer
            Deterministic fallback answer preserving grounding discipline.
        """

        return GeneratedAnswer(
            answer_text=(
                "No reliable grounded context was retrieved for this question. "
                "A supported answer cannot be generated safely."
            ),
            grounded=False,
            metadata={
                "provider": "external_gpt4o",
                "endpoint_url": self.endpoint_url,
                "used_grounded_fallback": True,
                "context_chunk_count": generation_input.context.chunk_count,
                "context_chunk_ids": [],
                "usage": {},
            },
        )

    def _build_request(
        self,
        generation_input: AnswerGenerationInput,
    ) -> Request:
        """
        Build one authenticated multipart request for the external endpoint.

        Parameters
        ----------
        generation_input : AnswerGenerationInput
            Normalized grounded generation payload.

        Returns
        -------
        Request
            Prepared urllib request carrying the multipart payload.
        """

        boundary = f"----projeto-ai2-{uuid.uuid4().hex}"
        multipart_fields = self._build_multipart_fields(generation_input)
        body = self._encode_multipart_form_data(
            fields=multipart_fields,
            boundary=boundary,
        )
        headers = {
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Accept": "application/json",
            self.auth_header_name: self._build_auth_header_value(),
        }

        return Request(
            url=self.endpoint_url,
            data=body,
            headers=headers,
            method="POST",
        )

    def _build_multipart_fields(
        self,
        generation_input: AnswerGenerationInput,
    ) -> Dict[str, str]:
        """
        Build the text fields sent to the external endpoint contract.

        Parameters
        ----------
        generation_input : AnswerGenerationInput
            Normalized grounded generation payload.

        Returns
        -------
        Dict[str, str]
            Multipart text fields using configured contract names.
        """

        metadata = self._build_request_metadata(generation_input)
        fields = {
            self.question_field_name: generation_input.question.question_text,
            self.context_field_name: generation_input.context.context_text,
            self.instructions_field_name: self._build_instruction_text(
                generation_input
            ),
            self.metadata_field_name: json.dumps(
                metadata,
                ensure_ascii=False,
                sort_keys=True,
            ),
        }

        for operational_field_name in (
            "channel_id",
            "thread_id",
            "user_info",
            "user_id",
            "user_name",
        ):
            operational_value = self._read_metadata_value(
                generation_input=generation_input,
                key=operational_field_name,
            )
            if operational_value:
                fields[operational_field_name] = operational_value

        return fields

    def _build_instruction_text(
        self,
        generation_input: AnswerGenerationInput,
    ) -> str:
        """
        Build the grounded instruction text sent outside the user message.

        Parameters
        ----------
        generation_input : AnswerGenerationInput
            Normalized grounded generation payload.

        Returns
        -------
        str
            Combined system instruction and structural context.
        """

        system_instruction = self._prompt_builder._build_system_instruction(
            generation_input
        )
        structural_context = self._prompt_builder._build_structural_context_text(
            generation_input
        )
        formatting_instructions = (
            self._prompt_builder._build_formatting_instruction_text(
                generation_input
            )
        )

        return (
            f"{system_instruction}\n\n"
            "Structural legal anchors:\n"
            f"{structural_context}\n\n"
            "Formatting expectations:\n"
            f"{formatting_instructions}"
        )

    def _build_request_metadata(
        self,
        generation_input: AnswerGenerationInput,
    ) -> Dict[str, Any]:
        """
        Build JSON metadata for traceability without exposing secrets.

        Parameters
        ----------
        generation_input : AnswerGenerationInput
            Normalized grounded generation payload.

        Returns
        -------
        Dict[str, Any]
            Request metadata aligned with retrieval-domain contracts.
        """

        metadata: Dict[str, Any] = {
            "request_id": generation_input.question.request_id,
            "conversation_id": generation_input.question.conversation_id,
            "normalized_query_text": (
                generation_input.question.normalized_query_text
            ),
            "formatting_instructions": list(
                generation_input.question.formatting_instructions
            ),
            "query_metadata": dict(generation_input.question.query_metadata),
            "generation_metadata": dict(generation_input.metadata),
            "context_chunk_count": generation_input.context.chunk_count,
            "context_chunk_ids": [
                chunk.chunk_id for chunk in generation_input.context.chunks
            ],
        }

        if generation_input.route_metadata is not None:
            metadata["route_metadata"] = generation_input.route_metadata.to_dict()

        return metadata

    def _read_metadata_value(
        self,
        *,
        generation_input: AnswerGenerationInput,
        key: str,
    ) -> str:
        """
        Read one operational endpoint field from request metadata.

        Parameters
        ----------
        generation_input : AnswerGenerationInput
            Normalized grounded generation payload.

        key : str
            Metadata key to resolve.

        Returns
        -------
        str
            String value when present, otherwise an empty string.
        """

        for metadata in (
            generation_input.metadata,
            generation_input.question.metadata,
        ):
            raw_value = metadata.get(key)
            if isinstance(raw_value, str) and raw_value.strip():
                return raw_value.strip()

        configured_value = self._read_configured_operational_value(key)
        if configured_value:
            return configured_value

        for environment_name in self._build_operational_environment_names(key):
            environment_value = os.getenv(environment_name, "").strip()
            if environment_value:
                return environment_value

        if key == "user_info":
            return "{}"

        return ""

    def _read_configured_operational_value(self, key: str) -> str:
        """
        Read one optional operational field from adapter configuration.

        Parameters
        ----------
        key : str
            Operational field name used by the endpoint contract.

        Returns
        -------
        str
            Configured string value when present, otherwise an empty string.
        """

        configured_values = {
            "channel_id": self.channel_id,
            "thread_id": self.thread_id,
            "user_info": self.user_info,
            "user_id": self.user_id,
            "user_name": self.user_name,
        }
        raw_value = configured_values.get(key, "")
        if isinstance(raw_value, str) and raw_value.strip():
            return raw_value.strip()

        return ""

    def _build_operational_environment_names(self, key: str) -> Tuple[str, ...]:
        """
        Build supported environment-variable names for operational fields.

        Parameters
        ----------
        key : str
            Operational field name used by the endpoint contract.

        Returns
        -------
        Tuple[str, ...]
            Environment-variable names checked in priority order.
        """

        normalized_key = key.upper()
        return (
            f"EXTERNAL_GPT4O_{normalized_key}",
            f"EXTERNAL_GPT4O_API_{normalized_key}",
            f"{self.auth_env_var}_{normalized_key}".upper(),
        )

    def _build_auth_header_value(self) -> str:
        """
        Resolve the configured authentication header value from the environment.

        Returns
        -------
        str
            Header value including the configured prefix when present.
        """

        auth_secret = os.getenv(self.auth_env_var, "").strip()
        if not auth_secret:
            raise AnswerGenerationError(
                "Environment variable "
                f"'{self.auth_env_var}' is required for external GPT-4o "
                "answer generation."
            )

        if not self.auth_header_prefix:
            return auth_secret

        return f"{self.auth_header_prefix} {auth_secret}"

    def _send_request(self, request: Request) -> Tuple[bytes, int]:
        """
        Send one request with bounded retries for transient failures.

        Parameters
        ----------
        request : Request
            Prepared HTTP request.

        Returns
        -------
        Tuple[bytes, int]
            Raw response body and HTTP status code.
        """

        request_sender = self.request_sender
        attempts = self.max_retries + 1
        last_error: Optional[Exception] = None

        for attempt_number in range(1, attempts + 1):
            try:
                if request_sender is None:
                    response = urlopen(request, timeout=self.timeout_seconds)
                else:
                    response = request_sender(request, self.timeout_seconds)
                status_code = int(getattr(response, "status", 200))
                response_body = response.read()

                if status_code == 429:
                    self._log_rate_limit_status(status_code, attempt_number, attempts)

                if self._is_retryable_status(status_code) and attempt_number < attempts:
                    last_error = AnswerGenerationError(
                        "External GPT-4o endpoint returned transient status "
                        f"{status_code}."
                    )
                    continue

                if status_code >= 400:
                    raise AnswerGenerationError(
                        "External GPT-4o endpoint returned HTTP status "
                        f"{status_code}."
                    )

                return response_body, status_code
            except HTTPError as exc:
                if exc.code == 429:
                    self._log_rate_limit_status(exc.code, attempt_number, attempts)

                if self._is_retryable_status(exc.code) and attempt_number < attempts:
                    last_error = exc
                    continue
                raise AnswerGenerationError(
                    "External GPT-4o endpoint returned HTTP status "
                    f"{exc.code}."
                ) from exc
            except (TimeoutError, URLError) as exc:
                last_error = exc
                if attempt_number < attempts:
                    continue
                break

        raise AnswerGenerationError(
            "External GPT-4o answer-generation request failed after "
            f"{attempts} attempt(s): {last_error}"
        ) from last_error

    def _parse_response_payload(self, response_body: bytes) -> Any:
        """
        Parse one endpoint response body into a robust payload.

        Parameters
        ----------
        response_body : bytes
            Raw response body returned by the endpoint.

        Returns
        -------
        Any
            Parsed JSON payload when possible, otherwise decoded text.
        """

        response_text = response_body.decode("utf-8", errors="replace").strip()
        if not response_text:
            return {}

        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            stream_events = self._parse_event_stream(response_text)
            if stream_events:
                return stream_events
            return response_text

    def _parse_event_stream(self, response_text: str) -> List[Mapping[str, Any]]:
        """
        Parse a newline-delimited endpoint event stream when present.

        Parameters
        ----------
        response_text : str
            Decoded response body that was not a single JSON document.

        Returns
        -------
        List[Mapping[str, Any]]
            Ordered endpoint events, or an empty list when the body is plain
            response text.
        """

        events: List[Mapping[str, Any]] = []
        non_empty_line_count = 0

        for raw_line in response_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            non_empty_line_count += 1
            parsed_event = self._parse_event_line(line)
            if parsed_event is None:
                continue

            events.append(parsed_event)

        if events and len(events) == non_empty_line_count:
            return events

        return []

    def _parse_event_line(self, line: str) -> Optional[Mapping[str, Any]]:
        """
        Parse one endpoint event line.

        Parameters
        ----------
        line : str
            One non-empty response line.

        Returns
        -------
        Optional[Mapping[str, Any]]
            Parsed event mapping when the line uses a supported event shape.
        """

        if line.startswith("{") and line.endswith("}"):
            parsed_mapping = self._parse_mapping_literal(line)
            if parsed_mapping is not None and self._is_event_mapping(parsed_mapping):
                return parsed_mapping

        key_value_mapping = self._parse_key_value_event_line(line)
        if key_value_mapping and self._is_event_mapping(key_value_mapping):
            return key_value_mapping

        return None

    def _parse_mapping_literal(self, line: str) -> Optional[Mapping[str, Any]]:
        """
        Parse one JSON or Python-literal mapping from an event line.

        Parameters
        ----------
        line : str
            Candidate mapping literal.

        Returns
        -------
        Optional[Mapping[str, Any]]
            Parsed mapping when decoding succeeds, otherwise `None`.
        """

        for parser in (json.loads, ast.literal_eval):
            try:
                parsed_value = parser(line)
            except (SyntaxError, ValueError, json.JSONDecodeError):
                continue

            if isinstance(parsed_value, Mapping):
                return parsed_value

        return None

    def _parse_key_value_event_line(self, line: str) -> Dict[str, str]:
        """
        Parse one comma-delimited key-value event line.

        Parameters
        ----------
        line : str
            Candidate line such as `type=error, content=...`.

        Returns
        -------
        Dict[str, str]
            Parsed event fields when available.
        """

        parsed_fields: Dict[str, str] = {}
        current_key = ""

        for segment in line.split(","):
            if "=" not in segment:
                if current_key:
                    parsed_fields[current_key] = (
                        f"{parsed_fields[current_key]},{segment}"
                    ).strip()
                continue

            raw_key, raw_value = segment.split("=", 1)
            key = raw_key.strip()
            value = raw_value.strip()

            if key:
                parsed_fields[key] = value
                current_key = key

        return parsed_fields

    def _is_event_mapping(self, value: Mapping[str, Any]) -> bool:
        """
        Decide whether one mapping represents an endpoint stream event.

        Parameters
        ----------
        value : Mapping[str, Any]
            Candidate response mapping.

        Returns
        -------
        bool
            `True` when the mapping contains an event type.
        """

        event_type = value.get("type")
        return isinstance(event_type, str) and bool(event_type.strip())

    def _raise_for_application_error(self, response_payload: Any) -> None:
        """
        Raise a controlled error for endpoint-level application failures.

        Parameters
        ----------
        response_payload : Any
            Parsed response payload.
        """

        error_message = self._extract_application_error_message(response_payload)
        if error_message:
            if self._looks_like_rate_limit_message(error_message):
                LOGGER.warning(
                    "External GPT-4o answer generation reported rate limiting. "
                    "Endpoint='%s', auth_env_var='%s', error='%s'.",
                    self.endpoint_url,
                    self.auth_env_var,
                    error_message,
                )
            raise AnswerGenerationError(
                "External GPT-4o endpoint returned application error: "
                f"{error_message}."
            )

    def _is_retryable_status(self, status_code: int) -> bool:
        """
        Decide whether one endpoint status should be retried.

        Parameters
        ----------
        status_code : int
            HTTP status code returned by the external endpoint.

        Returns
        -------
        bool
            `True` when the status represents throttling or a transient failure.
        """

        return status_code == 429 or status_code >= 500

    def _log_rate_limit_status(
        self,
        status_code: int,
        attempt_number: int,
        attempts: int,
    ) -> None:
        """
        Log an external endpoint throttling signal without exposing secrets.

        Parameters
        ----------
        status_code : int
            HTTP status code returned by the external endpoint.

        attempt_number : int
            One-based attempt number.

        attempts : int
            Total number of configured attempts.
        """

        LOGGER.warning(
            "External GPT-4o answer generation hit rate limiting. "
            "Endpoint='%s', auth_env_var='%s', status_code=%s, attempt=%s/%s.",
            self.endpoint_url,
            self.auth_env_var,
            status_code,
            attempt_number,
            attempts,
        )

    def _looks_like_rate_limit_message(self, message: str) -> bool:
        """
        Detect endpoint throttling from application-level error text.

        Parameters
        ----------
        message : str
            Error message emitted by the endpoint payload.

        Returns
        -------
        bool
            `True` when the message appears to describe rate limiting.
        """

        normalized_message = message.lower()
        return "rate limit" in normalized_message or "429" in normalized_message

    def _extract_application_error_message(self, response_payload: Any) -> str:
        """
        Extract an endpoint application error message from supported payloads.

        Parameters
        ----------
        response_payload : Any
            Parsed response payload.

        Returns
        -------
        str
            Error message when the endpoint reports an application failure.
        """

        if isinstance(response_payload, list):
            for event in response_payload:
                error_message = self._extract_application_error_message(event)
                if error_message:
                    return error_message
            return ""

        if not isinstance(response_payload, Mapping):
            return ""

        event_type = response_payload.get("type")
        if isinstance(event_type, str) and event_type.strip().lower() == "error":
            return self._extract_error_text(response_payload)

        error_value = response_payload.get("error")
        if error_value:
            return self._extract_text_value(error_value) or str(error_value).strip()

        data_value = response_payload.get("data")
        if isinstance(data_value, Mapping):
            return self._extract_application_error_message(data_value)

        return ""

    def _extract_error_text(self, response_payload: Mapping[str, Any]) -> str:
        """
        Extract human-readable text from one endpoint error event.

        Parameters
        ----------
        response_payload : Mapping[str, Any]
            Error event mapping.

        Returns
        -------
        str
            Human-readable error text.
        """

        for error_key in ("content", "message", "error", "detail"):
            error_text = self._extract_text_value(response_payload.get(error_key))
            if error_text:
                return error_text

        return "Unknown endpoint application error"

    def _extract_answer_text(self, response_payload: Any) -> str:
        """
        Extract answer text from supported endpoint response shapes.

        Parameters
        ----------
        response_payload : Any
            Parsed response payload.

        Returns
        -------
        str
            Stripped answer text or an empty string when unavailable.
        """

        if isinstance(response_payload, str):
            if self._looks_like_event_stream_text(response_payload):
                return ""
            return response_payload.strip()

        if isinstance(response_payload, list):
            return self._extract_stream_answer_text(response_payload)

        if not isinstance(response_payload, Mapping):
            return ""

        for answer_key in ("answer_text", "answer", "response", "message", "content"):
            raw_value = response_payload.get(answer_key)
            extracted_text = self._extract_text_value(raw_value)
            if extracted_text:
                return extracted_text

        data_value = response_payload.get("data")
        if isinstance(data_value, Mapping):
            return self._extract_answer_text(data_value)

        return ""

    def _extract_stream_answer_text(
        self,
        response_events: List[Any],
    ) -> str:
        """
        Extract answer text from parsed endpoint stream events.

        Parameters
        ----------
        response_events : List[Any]
            Ordered stream events parsed from the endpoint response body.

        Returns
        -------
        str
            Combined answer text from message events, or an empty string when
            no answer event is present.
        """

        answer_parts: List[str] = []

        for response_event in response_events:
            if not isinstance(response_event, Mapping):
                continue

            event_type = self._extract_text_value(response_event.get("type")).lower()
            if event_type in {"start", "done", "error"}:
                continue

            for answer_key in (
                "answer_text",
                "answer",
                "response",
                "message",
                "content",
                "delta",
                "text",
            ):
                answer_text = self._extract_text_value(response_event.get(answer_key))
                if answer_text:
                    answer_parts.append(answer_text)
                    break

        return "\n".join(answer_parts).strip()

    def _looks_like_event_stream_text(self, response_text: str) -> bool:
        """
        Detect unparsed stream-event text that must not be returned as an answer.

        Parameters
        ----------
        response_text : str
            Candidate answer text.

        Returns
        -------
        bool
            `True` when the text resembles endpoint events instead of an answer.
        """

        normalized_text = response_text.strip().lower()
        if not normalized_text:
            return False

        return (
            normalized_text.startswith("type=")
            or "\ntype=" in normalized_text
            or '"type": "error"' in normalized_text
            or "'type': 'error'" in normalized_text
        )

    def _extract_text_value(self, raw_value: Any) -> str:
        """
        Normalize one candidate answer-text value.

        Parameters
        ----------
        raw_value : Any
            Candidate response value.

        Returns
        -------
        str
            Extracted text when available.
        """

        if isinstance(raw_value, str):
            return raw_value.strip()

        if isinstance(raw_value, Mapping):
            for nested_key in ("text", "content", "answer", "answer_text"):
                nested_text = self._extract_text_value(raw_value.get(nested_key))
                if nested_text:
                    return nested_text

        return ""

    def _extract_usage(self, response_payload: Any) -> Dict[str, int]:
        """
        Extract optional usage counters from endpoint response metadata.

        Parameters
        ----------
        response_payload : Any
            Parsed response payload.

        Returns
        -------
        Dict[str, int]
            Token usage counters when present.
        """

        if not isinstance(response_payload, Mapping):
            return {}

        raw_usage = response_payload.get("usage")
        if not isinstance(raw_usage, Mapping):
            return {}

        usage: Dict[str, int] = {}
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            raw_value = raw_usage.get(key)
            if isinstance(raw_value, int):
                usage[key] = raw_value

        return usage

    def _encode_multipart_form_data(
        self,
        *,
        fields: Mapping[str, str],
        boundary: str,
    ) -> bytes:
        """
        Encode text fields as multipart/form-data.

        Parameters
        ----------
        fields : Mapping[str, str]
            Text field mapping to encode.

        boundary : str
            Multipart boundary token.

        Returns
        -------
        bytes
            UTF-8 encoded multipart body.
        """

        body_lines = []

        for field_name, field_value in fields.items():
            body_lines.extend(
                [
                    f"--{boundary}",
                    (
                        "Content-Disposition: form-data; "
                        f'name="{self._escape_multipart_name(field_name)}"'
                    ),
                    "Content-Type: text/plain; charset=utf-8",
                    "",
                    field_value,
                ]
            )

        body_lines.extend([f"--{boundary}--", ""])

        return "\r\n".join(body_lines).encode("utf-8")

    def _escape_multipart_name(self, field_name: str) -> str:
        """
        Escape one multipart field name for a Content-Disposition header.

        Parameters
        ----------
        field_name : str
            Raw field name.

        Returns
        -------
        str
            Header-safe field name.
        """

        return field_name.replace("\\", "\\\\").replace('"', '\\"')
