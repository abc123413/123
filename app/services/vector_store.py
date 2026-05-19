import json
import os
import hashlib
from typing import Dict, List

import chromadb
from chromadb.api.models.Collection import Collection

from app.config import Settings
from app.models.document import DocumentChunk, DocumentRecord, SearchResult


class VectorStoreService:
    """封装 Chroma 持久化存储及文档元数据管理。"""

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
        """写入文档及切块向量。"""

        self.collection.add(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.content for chunk in chunks],
            embeddings=embeddings,
            metadatas=[
                {
                    "document_id": chunk.document_id,
                    "filename": chunk.metadata.get("filename", ""),
                    "chunk_index": chunk.chunk_index,
                    "file_type": document.file_type,
                    "file_hash": document.file_hash,
                }
                for chunk in chunks
            ],
        )
        documents = self.list_documents()
        documents = [item for item in documents if item.document_id != document.document_id]
        documents.append(document)
        self._save_documents(documents)

    def list_documents(self) -> List[DocumentRecord]:
        """获取所有文档元数据。"""

        with open(self.metadata_path, "r", encoding="utf-8") as file:
            raw = json.load(file)
        return [DocumentRecord.model_validate(item) for item in raw]

    def find_document_by_hash(self, file_hash: str) -> DocumentRecord | None:
        """按文件哈希查找已存在文档。"""

        if not file_hash:
            return None
        return next((item for item in self.list_documents() if item.file_hash == file_hash), None)

    def delete_document(self, document_id: str) -> DocumentRecord | None:
        """删除文档及其关联切块。"""

        documents = self.list_documents()
        target = next((item for item in documents if item.document_id == document_id), None)
        if target is None:
            return None

        self.collection.delete(where={"document_id": document_id})
        remaining = [item for item in documents if item.document_id != document_id]
        self._save_documents(remaining)
        return target

    def search(self, query_embedding: List[float], top_k: int) -> List[SearchResult]:
        """执行向量检索。"""

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
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
                    source="vector",
                    metadata=metadata,
                )
            )
        return self._deduplicate_results(search_results, top_k)

    def get_all_chunks(self) -> List[Dict]:
        """为 BM25 提供全量文本切块。"""

        result = self.collection.get(include=["documents", "metadatas"])
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

    def _save_documents(self, documents: List[DocumentRecord]) -> None:
        with open(self.metadata_path, "w", encoding="utf-8") as file:
            json.dump([item.model_dump(mode="json") for item in documents], file, ensure_ascii=False, indent=2)
