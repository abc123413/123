from statistics import mean
from typing import Dict, List, Optional

from openai import AsyncOpenAI
from ragas import SingleTurnSample
from ragas.llms import llm_factory
from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextPrecisionWithoutReference, LLMContextRecall

from app.config import Settings
from app.evaluation_models import RagasEvaluationResponseData, RagasEvaluationSampleRequest, RagasSampleScore
from app.models.document import DocumentRecord, SearchResult
from app.services.hybrid_search import HybridSearchService
from app.services.llm_service import LLMService
from app.services.reranker import RerankerService
from app.services.vector_store import VectorStoreService


class RagasEvaluationService:
    """Run Ragas-style evaluation over the existing RAG pipeline."""

    def __init__(
        self,
        settings: Settings,
        hybrid_search: HybridSearchService,
        reranker: RerankerService,
        llm_service: LLMService,
        vector_store: VectorStoreService,
    ) -> None:
        self.settings = settings
        self.hybrid_search = hybrid_search
        self.reranker = reranker
        self.llm_service = llm_service
        self.vector_store = vector_store

        client = AsyncOpenAI(api_key=self.settings.llm_api_key, base_url=self.settings.llm_api_base)
        self.evaluator_llm = llm_factory(
            self.settings.llm_model,
            provider="openai",
            client=client,
            temperature=0,
            max_tokens=max(self.settings.llm_max_tokens, 4096),
        )
        self.faithfulness = Faithfulness(llm=self.evaluator_llm)
        self.context_precision_no_ref = LLMContextPrecisionWithoutReference(llm=self.evaluator_llm)
        self.context_precision_ref = LLMContextPrecisionWithReference(llm=self.evaluator_llm)
        self.context_recall = LLMContextRecall(llm=self.evaluator_llm)

    async def evaluate_samples(
        self,
        samples: List[RagasEvaluationSampleRequest],
        note: str = "",
    ) -> RagasEvaluationResponseData:
        scored_samples = [await self._score_sample(sample) for sample in samples]
        return self._build_response(scored_samples, note=note)

    async def quick_evaluate(
        self,
        sample_count: int = 5,
        document_ids: Optional[List[str]] = None,
    ) -> RagasEvaluationResponseData:
        documents = self._pick_documents(sample_count, document_ids)
        samples = [
            RagasEvaluationSampleRequest(
                query=f"请概括《{document.title}》的核心内容，并给出关键点。",
                reference=self._build_reference(document),
                document_ids=[document.document_id],
                top_k=10,
            )
            for document in documents
        ]
        note = "Quick mode uses heuristic questions generated from document titles and retrieval text."
        return await self.evaluate_samples(samples, note=note)

    async def _score_sample(self, sample: RagasEvaluationSampleRequest) -> RagasSampleScore:
        results = self.hybrid_search.search(
            sample.query,
            sample.top_k,
            sample.document_ids,
            sample.title_keyword,
        )
        reranked_results = self.reranker.rerank(sample.query, results, sample.use_rerank)
        structured_answer = await self.llm_service.generate_structured_answer(sample.query, reranked_results)
        answer = structured_answer["answer"]

        if not reranked_results:
            metric_values: Dict[str, float] = {
                "faithfulness": 0.0,
                "context_precision": 0.0,
            }
            if sample.reference:
                metric_values["context_recall"] = 0.0
            return RagasSampleScore(
                query=sample.query,
                reference=sample.reference,
                answer=answer,
                metrics=metric_values,
                retrieved_chunks=0,
                document_ids=[],
            )

        ragas_sample = self._build_ragas_sample(sample.query, answer, reranked_results, sample.reference)
        metric_values: Dict[str, float] = {
            "faithfulness": float(await self.faithfulness.single_turn_ascore(ragas_sample)),
            "context_precision": float(await self._score_context_precision(ragas_sample, sample.reference)),
        }
        if sample.reference:
            metric_values["context_recall"] = float(await self.context_recall.single_turn_ascore(ragas_sample))

        return RagasSampleScore(
            query=sample.query,
            reference=sample.reference,
            answer=answer,
            metrics=metric_values,
            retrieved_chunks=len(reranked_results),
            document_ids=self._extract_document_ids(reranked_results),
        )

    async def _score_context_precision(self, sample: SingleTurnSample, reference: Optional[str]) -> float:
        if reference:
            return float(await self.context_precision_ref.single_turn_ascore(sample))
        return float(await self.context_precision_no_ref.single_turn_ascore(sample))

    def _build_ragas_sample(
        self,
        query: str,
        answer: str,
        results: List[SearchResult],
        reference: Optional[str],
    ) -> SingleTurnSample:
        payload = {
            "user_input": query,
            "response": answer,
            "retrieved_contexts": [item.content for item in results[: self.settings.context_max_chunks]],
        }
        if reference:
            payload["reference"] = reference
        return SingleTurnSample(**payload)

    def _build_response(self, samples: List[RagasSampleScore], note: str = "") -> RagasEvaluationResponseData:
        metric_totals: Dict[str, List[float]] = {}
        for sample in samples:
            for name, value in sample.metrics.items():
                metric_totals.setdefault(name, []).append(value)

        metric_averages = {name: float(mean(values)) for name, values in metric_totals.items() if values}
        overall_score = float(mean(metric_averages.values())) if metric_averages else 0.0
        return RagasEvaluationResponseData(
            overall_score=overall_score,
            metric_averages=metric_averages,
            evaluated_samples=len(samples),
            samples=samples,
            note=note,
        )

    def _pick_documents(self, sample_count: int, document_ids: Optional[List[str]]) -> List[DocumentRecord]:
        all_documents = self.vector_store.list_documents()
        if document_ids:
            allowed = set(document_ids)
            all_documents = [item for item in all_documents if item.document_id in allowed]
        if not all_documents:
            raise ValueError("No indexed documents available for quick RAGAS evaluation")
        return all_documents[:sample_count]

    def _build_reference(self, document: DocumentRecord) -> str:
        retrieval_text = str(document.metadata.get("retrieval_text", "")).strip()
        if retrieval_text:
            return retrieval_text[:2000]
        return f"{document.title} {document.filename}".strip()

    def _extract_document_ids(self, results: List[SearchResult]) -> List[str]:
        seen: List[str] = []
        for item in results:
            if item.document_id and item.document_id not in seen:
                seen.append(item.document_id)
        return seen
