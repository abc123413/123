import os
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from loguru import logger

from app.config import Settings, get_settings
from app.models.document import DocumentRecord, QueryResponseData
from app.models.request import (
    DeleteResponse,
    DocumentsResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    UploadResponse,
)
from app.services.document_processor import DocumentProcessor
from app.services.embedding import EmbeddingService
from app.services.hybrid_search import HybridSearchService
from app.services.llm_service import LLMService
from app.services.query_rewriter import QueryRewriterService
from app.services.reranker import RerankerService
from app.services.vector_store import VectorStoreService
from app.utils.file_handler import FileHandler

router = APIRouter()


def get_file_handler(settings: Settings = Depends(get_settings)) -> FileHandler:
    return FileHandler(settings)


def get_document_processor(settings: Settings = Depends(get_settings)) -> DocumentProcessor:
    return DocumentProcessor(settings)


def get_embedding_service(settings: Settings = Depends(get_settings)) -> EmbeddingService:
    return EmbeddingService(settings)


def get_vector_store(settings: Settings = Depends(get_settings)) -> VectorStoreService:
    return VectorStoreService(settings)


def get_query_rewriter(settings: Settings = Depends(get_settings)) -> QueryRewriterService:
    return QueryRewriterService(settings)


def get_hybrid_search(
    settings: Settings = Depends(get_settings),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStoreService = Depends(get_vector_store),
    query_rewriter: QueryRewriterService = Depends(get_query_rewriter),
) -> HybridSearchService:
    return HybridSearchService(settings, embedding_service, vector_store, query_rewriter)


def get_reranker(settings: Settings = Depends(get_settings)) -> RerankerService:
    return RerankerService(settings)


def get_llm_service(settings: Settings = Depends(get_settings)) -> LLMService:
    return LLMService(settings)


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    title: str | None = Form(default=None),
    replace_document_id: str | None = Form(default=None),
    file_handler: FileHandler = Depends(get_file_handler),
    document_processor: DocumentProcessor = Depends(get_document_processor),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStoreService = Depends(get_vector_store),
) -> UploadResponse:
    """Upload, parse, embed, and index a document."""

    try:
        replaced_document: DocumentRecord | None = None
        if replace_document_id:
            replaced_document = vector_store.delete_document(replace_document_id)
            if replaced_document is None:
                raise HTTPException(status_code=404, detail="Document to replace was not found")
            file_handler.delete_file(replaced_document.file_path)

        document_id, file_path = await file_handler.save_upload_file(file)
        file_hash = file_handler.calculate_file_hash(file_path)
        existing_document = vector_store.find_document_by_hash(file_hash)
        if existing_document is not None:
            file_handler.delete_file(file_path)
            logger.info("Duplicate document skipped: {}", existing_document.filename)
            return UploadResponse(message="Document already exists, duplicate indexing skipped", document=existing_document)

        chunks = document_processor.process_document(document_id, file_path, file.filename or "unknown")
        if not chunks:
            raise HTTPException(status_code=400, detail="Document has no valid text content to index")

        embeddings = embedding_service.embed_texts([item.content for item in chunks])
        resolved_title = file_handler.build_title(file.filename or "unknown", title)
        document = DocumentRecord(
            document_id=document_id,
            title=resolved_title,
            filename=file.filename or "unknown",
            file_type=Path(file.filename or "").suffix.lower(),
            file_path=file_path,
            file_size=os.path.getsize(file_path),
            file_hash=file_hash,
            chunk_count=len(chunks),
            metadata={
                "original_filename": file.filename or "unknown",
                "replaced_document_id": replaced_document.document_id if replaced_document else None,
                "retrieval_text": " ".join(
                    [
                        resolved_title,
                        file.filename or "unknown",
                        " ".join(chunk.content for chunk in chunks[:2])[:2000],
                    ]
                ),
            },
        )
        vector_store.add_document(document, chunks, embeddings)
        logger.info("Document indexed successfully: {}", document.filename)
        return UploadResponse(message="Document uploaded and indexed successfully", document=document)
    except HTTPException:
        raise
    except ValueError as exc:
        logger.exception("Upload processing failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Upload processing failed")
        raise HTTPException(status_code=500, detail=f"Document processing failed: {exc}") from exc


@router.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    hybrid_search: HybridSearchService = Depends(get_hybrid_search),
    reranker: RerankerService = Depends(get_reranker),
    llm_service: LLMService = Depends(get_llm_service),
) -> QueryResponse | StreamingResponse:
    """Run hybrid retrieval, rerank, and answer generation."""

    try:
        results = hybrid_search.search(
            request.query,
            request.top_k,
            request.document_ids,
            request.title_keyword,
        )
        reranked_results = reranker.rerank(request.query, results, request.use_rerank)
        prompt = llm_service.build_prompt(request.query, reranked_results)

        if request.stream:
            stream_generator = await llm_service.generate_answer(request.query, reranked_results, stream=True)

            async def event_stream():
                async for chunk in stream_generator:
                    yield chunk

            return StreamingResponse(event_stream(), media_type="text/plain; charset=utf-8")

        structured_answer = await llm_service.generate_structured_answer(request.query, reranked_results)
        return QueryResponse(
            data=QueryResponseData(
                answer=structured_answer["answer"],
                conclusion=structured_answer["conclusion"],
                key_points=structured_answer["key_points"],
                citations=structured_answer["citations"],
                query=request.query,
                results=reranked_results,
                prompt=prompt,
            )
        )
    except ValueError as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc


@router.get("/documents", response_model=DocumentsResponse)
async def list_documents(
    vector_store: VectorStoreService = Depends(get_vector_store),
) -> DocumentsResponse:
    """List all indexed documents."""

    return DocumentsResponse(documents=vector_store.list_documents())


@router.delete("/documents/{document_id}", response_model=DeleteResponse)
async def delete_document(
    document_id: str,
    vector_store: VectorStoreService = Depends(get_vector_store),
    file_handler: FileHandler = Depends(get_file_handler),
) -> DeleteResponse:
    """Delete one document and its vectors."""

    document = vector_store.delete_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    file_handler.delete_file(document.file_path)
    return DeleteResponse(message="Document deleted successfully", document_id=document_id)


@router.delete("/documents", response_model=DeleteResponse)
async def clear_documents(
    vector_store: VectorStoreService = Depends(get_vector_store),
    file_handler: FileHandler = Depends(get_file_handler),
) -> DeleteResponse:
    """Delete all indexed documents."""

    documents = vector_store.list_documents()
    for document in documents:
        vector_store.delete_document(document.document_id)
        file_handler.delete_file(document.file_path)
    return DeleteResponse(message="All documents deleted successfully", document_id="all")


@router.get("/health", response_model=HealthResponse)
async def health_check(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Health check."""

    return HealthResponse(app_name=settings.app_name)
