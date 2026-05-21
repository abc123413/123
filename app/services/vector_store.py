import hashlib
import json
import os
from typing import Dict, List

import chromadb
from chromadb.api.models.Collection import Collection
from rank_bm25 import BM25Okapi

from app.config import Settings
from app.models.document import DocumentChunk, DocumentRecord, SearchResult


class VectorStoreService:
    """Chroma backed vector store with document registry support."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        os.makedirs(self.settings.chroma_persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.settings.chroma_persist_dir)
        self.collection: Collection = self.client.get_or_create_collection(
            name=self.settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.metadata_path = os.path.join(self.settings.data_dir, "documents.json")
        os.makedirs(self.settings.data_dir, exist_ok=True)
        if not os.path.exists(self.metadata_path):
            with open(self.metadata_path, "w", encoding="utf-8") as file:
                json.dump([], file, ensure_ascii=False)

    def add_document(self, document: DocumentRecord, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> None:
        self.collection.add(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.content for chunk in chunks],
            embeddings=embeddings,
            metadatas=[
                {
                    "document_id": chunk.document_id,
                    "title": document.title,
                    "filename": chunk.metadata.get("filename", ""),
                    "chunk_index": chunk.chunk_index,
                    "file_type": document.file_type,
                    "file_hash": document.file_hash,
                    "created_at": document.created_at.isoformat(),
                    "content_length": chunk.metadata.get("content_length", len(chunk.content)),
                    "modality": chunk.metadata.get("modality", "text"),
                    "has_vlm": chunk.metadata.get("has_vlm", False),
                    "has_ocr": chunk.metadata.get("has_ocr", False),
                }
                for chunk in chunks
            ],
        )
        documents = [item for item in self.list_documents() if item.document_id != document.document_id]
        documents.append(document)
        self._save_documents(documents)

    def list_documents(self) -> List[DocumentRecord]:
        with open(self.metadata_path, "r", encoding="utf-8") as file:
            raw = json.load(file)
        documents = [DocumentRecord.model_validate(item) for item in raw]
        return sorted(documents, key=lambda item: item.created_at, reverse=True)

    def get_document(self, document_id: str) -> DocumentRecord | None:
        return next((item for item in self.list_documents() if item.document_id == document_id), None)

    def find_document_by_hash(self, file_hash: str) -> DocumentRecord | None:
        if not file_hash:
            return None
        return next((item for item in self.list_documents() if item.file_hash == file_hash), None)

    def resolve_document_ids(self, document_ids: List[str] | None = None, title_keyword: str | None = None) -> List[str]:
        documents = self.list_documents()
        filtered = documents
        if document_ids:
            allowed = set(document_ids)
            filtered = [item for item in filtered if item.document_id in allowed]
        if title_keyword:
            keyword = title_keyword.strip().lower()
            filtered = [item for item in filtered if keyword in item.title.lower()]
        return [item.document_id for item in filtered]

    def shortlist_documents(self, query: str, top_k: int) -> List[str]:
        documents = self.list_documents()
        if not documents:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return [item.document_id for item in documents[:top_k]]

        corpus = [self._tokenize(self._document_retrieval_text(item)) for item in documents]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(query_tokens)

        ranked = []
        normalized_query = query.lower()
        for index, score in enumerate(scores):
            document = documents[index]
            title_lower = document.title.lower()
            filename_lower = document.filename.lower()
            boost = 0.0
            if normalized_query in title_lower:
                boost += 3.0
            if normalized_query in filename_lower:
                boost += 2.0
            for token in query_tokens[:6]:
                if token in title_lower:
                    boost += 0.8
                if token in filename_lower:
                    boost += 0.4
            ranked.append((document.document_id, float(score) + boost))

        ranked.sort(key=lambda item: item[1], reverse=True)
        shortlisted = [document_id for document_id, score in ranked if score > 0][:top_k]
        if shortlisted:
            return shortlisted
        return [item.document_id for item in documents[:top_k]]

    def delete_document(self, document_id: str) -> DocumentRecord | None:
        documents = self.list_documents()
        target = next((item for item in documents if item.document_id == document_id), None)
        if target is None:
            return None

        self.collection.delete(where={"document_id": document_id})
        remaining = [item for item in documents if item.document_id != document_id]
        self._save_documents(remaining)
        return target

    def search(self, query_embedding: List[float], top_k: int, document_ids: List[str] | None = None) -> List[SearchResult]:
        where = {"document_id": {"$in": document_ids}} if document_ids else None
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
            where=where,
        )

        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        ids = result.get("ids", [[]])[0]

        search_results: List[SearchResult] = []
        for chunk_id, content, metadata, distance in zip(ids, documents, metadatas, distances):
            metadata = metadata or {}
            score = 1 / (1 + distance)
            search_results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    document_id=metadata.get("document_id", ""),
                    content=content,
                    score=score,
                    retrieval_score=score,
                    source="vector",
                    metadata=metadata,
                )
            )
        return self._deduplicate_results(search_results, top_k)

    def get_all_chunks(self, document_ids: List[str] | None = None) -> List[Dict]:
        where = {"document_id": {"$in": document_ids}} if document_ids else None
        result = self.collection.get(include=["documents", "metadatas"], where=where)
        chunks = []
        seen_keys = set()
        for chunk_id, content, metadata in zip(result["ids"], result["documents"], result["metadatas"]):
            metadata = metadata or {}
            key = self._content_fingerprint(metadata.get("document_id", ""), content)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": metadata.get("document_id", ""),
                    "content": content,
                    "metadata": metadata,
                }
            )
        return chunks

    def get_chunks_by_document(self, document_id: str) -> List[Dict]:
        result = self.collection.get(where={"document_id": document_id}, include=["documents", "metadatas"])
        chunks = []
        for chunk_id, content, metadata in zip(result["ids"], result["documents"], result["metadatas"]):
            metadata = metadata or {}
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": metadata.get("document_id", ""),
                    "content": content,
                    "metadata": metadata,
                }
            )
        return sorted(chunks, key=lambda item: int(item["metadata"].get("chunk_index", 0)))

    def get_chunk_neighbors(self, document_id: str, chunk_index: int, window: int) -> List[Dict]:
        chunks = self.get_chunks_by_document(document_id)
        return [
            item
            for item in chunks
            if abs(int(item["metadata"].get("chunk_index", 0)) - chunk_index) <= window
        ]

    def _deduplicate_results(self, results: List[SearchResult], limit: int) -> List[SearchResult]:
        deduplicated: List[SearchResult] = []
        seen_keys = set()

        for item in results:
            key = self._content_fingerprint(item.document_id, item.content)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduplicated.append(item)
            if len(deduplicated) >= limit:
                break

        return deduplicated

    def _content_fingerprint(self, document_id: str, content: str) -> str:
        normalized = " ".join(content.split())
        return hashlib.sha256(f"{document_id}:{normalized}".encode("utf-8")).hexdigest()

    def _document_retrieval_text(self, document: DocumentRecord) -> str:
        retrieval_text = str(document.metadata.get("retrieval_text", "")).strip()
        if retrieval_text:
            return retrieval_text
        return f"{document.title} {document.filename}"

    def _tokenize(self, text: str) -> List[str]:
        return [token for token in __import__("re").split(r"[\s,.;:!?，。；：！？（）\[\]{}]+", text.lower()) if token]

    def _save_documents(self, documents: List[DocumentRecord]) -> None:
        with open(self.metadata_path, "w", encoding="utf-8") as file:
            json.dump([item.model_dump(mode="json") for item in documents], file, ensure_ascii=False, indent=2)
