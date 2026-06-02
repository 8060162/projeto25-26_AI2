from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question:   str       = Field(..., min_length=1, max_length=1000)
    session_id: str | None = Field(default=None, description="Para memória curta — opcional")


class FeedbackRequest(BaseModel):
    trace_id: str  = Field(..., description="trace_id da resposta a avaliar")
    rating:   str  = Field(..., pattern="^(positive|negative)$")
    reason:   str | None = Field(default=None, max_length=500)