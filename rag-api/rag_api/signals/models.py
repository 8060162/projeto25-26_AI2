from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class QuerySignal:
    trace_id:           str
    client_id:          str
    docs_retrieved:     int
    retrieval_score:    float
    latency_ms:         int
    timestamp:          datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    query_category:     str = "uncategorized"
    tokens_input:       int = 0
    tokens_output:      int = 0
    success:            bool = True
    error_type:         str | None = None
    question_chars:     int = 0
    grounded:           bool = False
    evidence_strength:  str = "unknown"   # strong / weak / none
    grounding_status:   str = "unknown"   # strong_alignment / weak_alignment / mismatch

    def to_document(self) -> dict:
        doc = {
            "timestamp": self.timestamp,
            "metadata": {
                "client_id":      self.client_id,
                "query_category": self.query_category,
            },
            "trace_id":          self.trace_id,
            "docs_retrieved":    self.docs_retrieved,
            "retrieval_score":   self.retrieval_score,
            "latency_ms":        self.latency_ms,
            "tokens_input":      self.tokens_input,
            "tokens_output":     self.tokens_output,
            "success":           self.success,
            "question_chars":    self.question_chars,
            "grounded":          self.grounded,
            "evidence_strength": self.evidence_strength,
            "grounding_status":  self.grounding_status,
        }
        if self.error_type is not None:
            doc["error_type"] = self.error_type
        return doc