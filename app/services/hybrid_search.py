import math
import re
import hashlib
from collections import defaultdict
from typing import Dict, List

from rank_bm25 import BM25Okapi

from app.config import Settings
from app.models.document import SearchResult
from app.services.embedding import EmbeddingService
from app.services.vector_store import VectorStoreService


class HybridSearchService:
    """实现向量检索与 BM25 的混合召回。"""

    def __init__(
        self,
        settings: Settings,
        embedding_service: EmbeddingService,
        vector_store: VectorStoreService,
    ) -> None:
        self.settings = settings
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def search(self, query: str, top_k: int | None = None) -> List[SearchResult]:
        """根据配置执行向量检索、BM25 或混合检索。"""

        final_top_k = top_k or self.settings.retrieval_top_k
        query_embedding = self.embedding_service.embed_query(query)
        vector_results = self.vector_store.search(query_embedding, self.settings.vector_top_k)

        if not self.settings.hybrid_search_enabled:
            return vector_results[:final_top_k]

        bm25_results = self._bm25_search(query, self.settings.bm25_top_k)
        if self.settings.fusion_mode.lower() == "weighted":
            merged = self._weighted_fusion(vector_results, bm25_results)
        else:
            merged = self._rrf_fusion(vector_results, bm25_results)
        return self._deduplicate_results(merged, final_top_k)

    def _bm25_search(self, query: str, top_k: int) -> List[SearchResult]:
        chunks = self.vector_store.get_all_chunks()
        if not chunks:
            return []

        corpus = [self._tokenize(item["content"]) for item in chunks]
        bm25 = BM25Okapi(corpus)
        query_tokens = self._tokenize(query)
        scores = bm25.get_scores(query_tokens)

        indexed_scores = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]
        results: List[SearchResult] = []
        for index, score in indexed_scores:
            chunk = chunks[index]
            results.append(
                SearchResult(
                    chunk_id=chunk["chunk_id"],
                    document_id=chunk["document_id"],
                    content=chunk["content"],
                    score=float(score),
                    source="bm25",
                    metadata=chunk["metadata"],
                )
            )
        return results

    def _rrf_fusion(self, vector_results: List[SearchResult], bm25_results: List[SearchResult]) -> List[SearchResult]:
        merged_scores: Dict[str, float] = defaultdict(float)
        merged_items: Dict[str, SearchResult] = {}

        for rank, item in enumerate(vector_results, start=1):
            merged_scores[item.chunk_id] += 1 / (self.settings.rrf_k + rank)
            merged_items[item.chunk_id] = item

        for rank, item in enumerate(bm25_results, start=1):
            merged_scores[item.chunk_id] += 1 / (self.settings.rrf_k + rank)
            if item.chunk_id not in merged_items:
                merged_items[item.chunk_id] = item

        ordered = sorted(merged_scores.items(), key=lambda pair: pair[1], reverse=True)
        results = []
        for chunk_id, score in ordered:
            result = merged_items[chunk_id]
            result.score = score
            result.source = "hybrid"
            results.append(result)
        return results

    def _deduplicate_results(self, results: List[SearchResult], limit: int) -> List[SearchResult]:
        deduplicated: List[SearchResult] = []
        seen_hashes = set()

        for item in results:
            normalized = " ".join(item.content.split())
            key = hashlib.sha256(f"{item.document_id}:{normalized}".encode("utf-8")).hexdigest()
            if key in seen_hashes:
                continue
            seen_hashes.add(key)
            deduplicated.append(item)
            if len(deduplicated) >= limit:
                break

        return deduplicated

    def _weighted_fusion(self, vector_results: List[SearchResult], bm25_results: List[SearchResult]) -> List[SearchResult]:
        merged_scores: Dict[str, float] = defaultdict(float)
        merged_items: Dict[str, SearchResult] = {}

        vector_norm = self._normalize_scores(vector_results)
        bm25_norm = self._normalize_scores(bm25_results)

        for item in vector_results:
            merged_scores[item.chunk_id] += vector_norm.get(item.chunk_id, 0.0) * self.settings.vector_weight
            merged_items[item.chunk_id] = item

        for item in bm25_results:
            merged_scores[item.chunk_id] += bm25_norm.get(item.chunk_id, 0.0) * self.settings.bm25_weight
            if item.chunk_id not in merged_items:
                merged_items[item.chunk_id] = item

        ordered = sorted(merged_scores.items(), key=lambda pair: pair[1], reverse=True)
        results = []
        for chunk_id, score in ordered:
            result = merged_items[chunk_id]
            result.score = score
            result.source = "hybrid"
            results.append(result)
        return results

    def _normalize_scores(self, results: List[SearchResult]) -> Dict[str, float]:
        if not results:
            return {}
        scores = [item.score for item in results]
        max_score = max(scores)
        min_score = min(scores)
        if math.isclose(max_score, min_score):
            return {item.chunk_id: 1.0 for item in results}
        return {
            item.chunk_id: (item.score - min_score) / (max_score - min_score)
            for item in results
        }

    def _tokenize(self, text: str) -> List[str]:
        return [token for token in re.split(r"[\s,.;:!?，。；：！？()（）\[\]{}]+", text.lower()) if token]
