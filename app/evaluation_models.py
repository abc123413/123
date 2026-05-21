from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class RagasEvaluationSampleRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Evaluation query")
    reference: Optional[str] = Field(default=None, description="Optional reference answer")
    document_ids: Optional[List[str]] = Field(default=None, description="Scope retrieval to these documents")
    title_keyword: Optional[str] = Field(default=None, description="Scope retrieval by document title keyword")
    top_k: int = Field(default=10, ge=1, le=100, description="Retrieval top K")
    use_rerank: Optional[bool] = Field(default=None, description="Enable rerank")


class RagasQuickEvaluationRequest(BaseModel):
    sample_count: int = Field(default=5, ge=1, le=50, description="Number of documents to sample")
    document_ids: Optional[List[str]] = Field(default=None, description="Limit to these documents")


class RagasSampleScore(BaseModel):
    query: str
    reference: Optional[str] = None
    answer: str
    metrics: Dict[str, float] = Field(default_factory=dict)
    retrieved_chunks: int = 0
    document_ids: List[str] = Field(default_factory=list)


class RagasEvaluationResponseData(BaseModel):
    overall_score: float
    metric_averages: Dict[str, float] = Field(default_factory=dict)
    evaluated_samples: int = 0
    samples: List[RagasSampleScore] = Field(default_factory=list)
    score_scale: str = "0-1"
    note: str = ""


class RagasEvaluationResponse(BaseModel):
    success: bool = True
    message: str = "ok"
    data: RagasEvaluationResponseData
