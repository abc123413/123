import hashlib
import math
import re
from collections import defaultdict
from typing import Dict, List

from rank_bm25 import BM25Okapi

from app.config import Settings
from app.models.document import SearchResult
from app.services.embedding import EmbeddingService
from app.services.query_rewriter import QueryRewriterService
from app.services.vector_store import VectorStoreService


class HybridSearchService:
    """Hybrid retrieval with query rewrite, reroute, and doc-level filtering."""

    def __init__(
        self,
        settings: Settings,
        embedding_service: EmbeddingService,
        vector_store: VectorStoreService,
        query_rewriter: QueryRewriterService,
    ) -> None:
        self.settings = settings
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.query_rewriter = query_rewriter

    def search(
        self,
        query: str,
        top_k: int | None = None,
        document_ids: List[str] | None = None,
        title_keyword: str | None = None,
    ) -> List[SearchResult]:
        final_top_k = top_k or self.settings.retrieval_top_k
        scoped_document_ids = self._resolve_search_scope(query, document_ids, title_keyword)
        if (document_ids is not None or title_keyword) and not scoped_document_ids:
            return []

        queries = [query]
        if self.settings.query_rewrite_enabled:
            rewrites = self.query_rewriter.rewrite(query)
            if rewrites:
                queries = rewrites

        merged = self._search_multi_query(query, queries, final_top_k, scoped_document_ids)

        if self._should_run_second_pass(merged):
            second_pass_queries = self._build_second_pass_queries(query, queries)
            retry_results = self._search_multi_query(
                query,
                second_pass_queries,
                self.settings.second_pass_top_k,
                scoped_document_ids,
            )
            merged = self._deduplicate_results(merged + retry_results, self.settings.second_pass_top_k)

        if self.settings.neighbor_expand_enabled:
            merged = self._expand_neighbors(merged)

        return self._deduplicate_results(merged, final_top_k)

    def _resolve_search_scope(
        self,
        query: str,
        document_ids: List[str] | None,
        title_keyword: str | None,
    ) -> List[str] | None:
        if document_ids is not None or title_keyword:
            return self.vector_store.resolve_document_ids(document_ids, title_keyword)

        all_documents = self.vector_store.list_documents()
        if len(all_documents) <= self.settings.document_scope_trigger_count:
            return None

        return self.vector_store.shortlist_documents(query, self.settings.document_candidate_top_k)

    def _search_multi_query(
        self,
        original_query: str,
        queries: List[str],
        final_top_k: int,
        document_ids: List[str] | None,
    ) -> List[SearchResult]:
        combined_results: List[SearchResult] = []
        image_query = self._is_image_query(original_query)

        for current_query in queries:
            query_embedding = self.embedding_service.embed_query(current_query)
            vector_results = self.vector_store.search(query_embedding, self.settings.vector_top_k, document_ids)
            if not self.settings.hybrid_search_enabled:
                combined_results.extend(self._apply_modality_boost(original_query, vector_results)[:final_top_k])
                continue

            bm25_results = self._bm25_search(current_query, self.settings.bm25_top_k, document_ids)
            if self.settings.fusion_mode.lower() == "weighted":
                merged = self._weighted_fusion(vector_results, bm25_results)
            else:
                merged = self._rrf_fusion(vector_results, bm25_results)

            boosted = self._apply_modality_boost(original_query, merged)
            if image_query:
                boosted = self._route_image_results(boosted)
            combined_results.extend(boosted[:final_top_k])

        return self._deduplicate_results(combined_results, max(final_top_k, self.settings.second_pass_top_k))

    def _bm25_search(self, query: str, top_k: int, document_ids: List[str] | None) -> List[SearchResult]:
        chunks = self.vector_store.get_all_chunks(document_ids)
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
            modality = chunk["metadata"].get("modality", "text")
            results.append(
                SearchResult(
                    chunk_id=chunk["chunk_id"],
                    document_id=chunk["document_id"],
                    content=chunk["content"],
                    score=float(score),
                    retrieval_score=float(score),
                    source="bm25",
                    metadata=chunk["metadata"],
                    modality_priority=self._modality_priority(modality),
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
            result.retrieval_score = score
            result.source = "hybrid"
            result.modality_priority = self._modality_priority(result.metadata.get("modality", "text"))
            results.append(result)
        return results

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
            result.retrieval_score = score
            result.source = "hybrid"
            result.modality_priority = self._modality_priority(result.metadata.get("modality", "text"))
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
        return {item.chunk_id: (item.score - min_score) / (max_score - min_score) for item in results}

    def _should_run_second_pass(self, results: List[SearchResult]) -> bool:
        if not self.settings.second_pass_enabled:
            return False
        if not results:
            return True
        top_score = results[0].score
        if top_score < self.settings.low_confidence_threshold:
            return True
        if len(results) >= 2 and abs(results[0].score - results[1].score) < self.settings.low_confidence_gap_threshold:
            return True
        return False

    def _build_second_pass_queries(self, original_query: str, existing_queries: List[str]) -> List[str]:
        keywords = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,}", original_query)
        expanded = list(existing_queries)
        if keywords:
            keyword_query = " ".join(keywords[:8])
            if keyword_query not in expanded:
                expanded.append(keyword_query)
        if self._is_image_query(original_query):
            for extra in [f"{original_query} 图片", f"{original_query} 图表", f"{original_query} 截图"]:
                if extra not in expanded:
                    expanded.append(extra)
        else:
            fallback = f"{original_query} 定义 原理 流程"
            if fallback not in expanded:
                expanded.append(fallback)
        return expanded[: self.settings.query_rewrite_max_queries + 3]

    def _expand_neighbors(self, results: List[SearchResult]) -> List[SearchResult]:
        expanded = list(results)
        seen_ids = {item.chunk_id for item in expanded}

        for item in results[: min(5, len(results))]:
            chunk_index = int((item.metadata or {}).get("chunk_index", 0))
            neighbors = self.vector_store.get_chunk_neighbors(
                item.document_id,
                chunk_index,
                self.settings.neighbor_expand_window,
            )
            for neighbor in neighbors:
                chunk_id = neighbor["chunk_id"]
                if chunk_id in seen_ids:
                    continue
                seen_ids.add(chunk_id)
                modality = neighbor["metadata"].get("modality", "text")
                boosted_score = max(item.score * 0.92, 0.0) * self._modality_boost_factor(
                    modality,
                    self._is_image_query(item.content),
                )
                expanded.append(
                    SearchResult(
                        chunk_id=chunk_id,
                        document_id=neighbor["document_id"],
                        content=neighbor["content"],
                        score=boosted_score,
                        retrieval_score=boosted_score,
                        source="neighbor",
                        metadata=neighbor["metadata"],
                        modality_priority=self._modality_priority(modality),
                    )
                )

        return expanded

    def _apply_modality_boost(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        image_query = self._is_image_query(query)
        adjusted: List[SearchResult] = []
        for item in results:
            modality = item.metadata.get("modality", "text")
            factor = self._modality_boost_factor(modality, image_query)
            item.score = item.score * factor
            item.modality_priority = self._modality_priority(modality)
            adjusted.append(item)
        adjusted.sort(key=lambda item: (item.score, item.modality_priority or 0.0), reverse=True)
        return adjusted

    def _route_image_results(self, results: List[SearchResult]) -> List[SearchResult]:
        return sorted(
            results,
            key=lambda item: (
                1 if item.metadata.get("modality") == "vlm" else 0,
                1 if item.metadata.get("modality") == "ocr" else 0,
                item.score,
            ),
            reverse=True,
        )

    def _is_image_query(self, query: str) -> bool:
        keywords = ["图片", "图", "截图", "界面", "图表", "海报", "表格", "照片", "画面", "视觉"]
        return any(keyword in query for keyword in keywords)

    def _modality_priority(self, modality: str) -> float:
        if modality == "vlm":
            return 3.0
        if modality == "ocr":
            return 2.0
        return 1.0

    def _modality_boost_factor(self, modality: str, image_query: bool) -> float:
        if image_query:
            if modality == "vlm":
                return 1.55
            if modality == "ocr":
                return 1.20
            return 0.92
        if modality == "vlm":
            return 0.95
        return 1.0

    def _deduplicate_results(self, results: List[SearchResult], limit: int) -> List[SearchResult]:
        ordered = sorted(results, key=lambda item: (item.score, item.modality_priority or 0.0), reverse=True)
        deduplicated: List[SearchResult] = []
        seen_hashes = set()

        for item in ordered:
            normalized = " ".join(item.content.split())
            key = hashlib.sha256(f"{item.document_id}:{normalized}".encode("utf-8")).hexdigest()
            if key in seen_hashes:
                continue
            seen_hashes.add(key)
            deduplicated.append(item)
            if len(deduplicated) >= limit:
                break

        return deduplicated

    def _tokenize(self, text: str) -> List[str]:
        return [token for token in re.split(r"[\s,.;:!?，。；：！？（）\[\]{}]+", text.lower()) if token]
