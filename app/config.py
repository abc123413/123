from functools import lru_cache
from typing import Annotated, List

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    """统一管理应用配置，支持从 .env 文件加载。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = Field(default="enterprise-rag", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    app_debug: bool = Field(default=False, alias="APP_DEBUG")
    api_prefix: str = Field(default="/api/v1", alias="API_PREFIX")
    cors_origins: Annotated[List[str], NoDecode] = Field(
        default_factory=lambda: ["*"],
        alias="CORS_ORIGINS",
    )

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_dir: str = Field(default="logs", alias="LOG_DIR")

    data_dir: str = Field(default="data", alias="DATA_DIR")
    upload_dir: str = Field(default="data/uploads", alias="UPLOAD_DIR")
    temp_dir: str = Field(default="data/tmp", alias="TEMP_DIR")

    max_file_size_mb: int = Field(default=50, alias="MAX_FILE_SIZE_MB")
    allowed_extensions: Annotated[List[str], NoDecode] = Field(
        default_factory=lambda: [".pdf", ".docx", ".txt", ".md", ".html"],
        alias="ALLOWED_EXTENSIONS",
    )
    enable_ocr: bool = Field(default=False, alias="ENABLE_OCR")
    ocr_language: str = Field(default="chi_sim+eng", alias="OCR_LANGUAGE")

    chunk_strategy: str = Field(default="semantic", alias="CHUNK_STRATEGY")
    chunk_size: int = Field(default=800, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=150, alias="CHUNK_OVERLAP")
    chunk_min_size: int = Field(default=200, alias="CHUNK_MIN_SIZE")
    preserve_sentence_boundary: bool = Field(default=True, alias="PRESERVE_SENTENCE_BOUNDARY")

    chroma_persist_dir: str = Field(default="data/chroma", alias="CHROMA_PERSIST_DIR")
    collection_name: str = Field(default="enterprise_docs", alias="COLLECTION_NAME")

    embedding_provider: str = Field(default="local", alias="EMBEDDING_PROVIDER")
    embedding_api_base: str = Field(default="https://api.openai.com/v1", alias="EMBEDDING_API_BASE")
    embedding_api_key: str = Field(default="", alias="EMBEDDING_API_KEY")
    embedding_model: str = Field(default="BAAI/bge-small-zh-v1.5", alias="EMBEDDING_MODEL")
    embedding_dimension: int = Field(default=512, alias="EMBEDDING_DIMENSION")
    embedding_batch_size: int = Field(default=32, alias="EMBEDDING_BATCH_SIZE")
    embedding_timeout: int = Field(default=60, alias="EMBEDDING_TIMEOUT")
    embedding_device: str = Field(default="cpu", alias="EMBEDDING_DEVICE")
    embedding_normalize: bool = Field(default=True, alias="EMBEDDING_NORMALIZE")
    embedding_cache_enabled: bool = Field(default=True, alias="EMBEDDING_CACHE_ENABLED")
    embedding_cache_dir: str = Field(default="data/cache/embeddings", alias="EMBEDDING_CACHE_DIR")

    retrieval_top_k: int = Field(default=10, alias="RETRIEVAL_TOP_K")
    hybrid_search_enabled: bool = Field(default=True, alias="HYBRID_SEARCH_ENABLED")
    vector_top_k: int = Field(default=20, alias="VECTOR_TOP_K")
    bm25_top_k: int = Field(default=20, alias="BM25_TOP_K")
    vector_weight: float = Field(default=0.7, alias="VECTOR_WEIGHT")
    bm25_weight: float = Field(default=0.3, alias="BM25_WEIGHT")
    fusion_mode: str = Field(default="rrf", alias="FUSION_MODE")
    rrf_k: int = Field(default=60, alias="RRF_K")

    rerank_enabled: bool = Field(default=True, alias="RERANK_ENABLED")
    rerank_initial_k: int = Field(default=50, alias="RERANK_INITIAL_K")
    rerank_final_k: int = Field(default=10, alias="RERANK_FINAL_K")
    rerank_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-12-v2",
        alias="RERANK_MODEL",
    )

    llm_api_base: str = Field(default="https://api.deepseek.com/v1", alias="LLM_API_BASE")
    llm_api_key: str = Field(default="", alias="LLM_API_KEY")
    llm_model: str = Field(default="deepseek-chat", alias="LLM_MODEL")
    llm_timeout: int = Field(default=120, alias="LLM_TIMEOUT")
    llm_max_retries: int = Field(default=3, alias="LLM_MAX_RETRIES")
    llm_temperature: float = Field(default=0.2, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=2048, alias="LLM_MAX_TOKENS")
    llm_stream: bool = Field(default=True, alias="LLM_STREAM")
    prompt_system_message: str = Field(
        default="你是企业级知识库问答助手，请严格基于检索到的上下文回答问题；如果上下文不足，明确说明不知道，不要编造。",
        alias="PROMPT_SYSTEM_MESSAGE",
    )
    context_max_chunks: int = Field(default=6, alias="CONTEXT_MAX_CHUNKS")
    context_max_tokens: int = Field(default=6000, alias="CONTEXT_MAX_TOKENS")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def split_cors_origins(cls, value: str | List[str]) -> List[str]:
        if isinstance(value, list):
            return value
        if not value:
            return ["*"]
        return [item.strip() for item in value.split(",") if item.strip()]

    @field_validator("allowed_extensions", mode="before")
    @classmethod
    def split_extensions(cls, value: str | List[str]) -> List[str]:
        if isinstance(value, list):
            return value
        return [item.strip() for item in value.split(",") if item.strip()]

    @field_validator("embedding_provider")
    @classmethod
    def normalize_embedding_provider(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"local", "openai"}:
            raise ValueError("EMBEDDING_PROVIDER 仅支持 local 或 openai")
        return normalized

    @model_validator(mode="after")
    def validate_embedding_config(self) -> "Settings":
        if self.embedding_provider == "openai" and not self.embedding_api_key:
            raise ValueError("当 EMBEDDING_PROVIDER=openai 时，必须配置 EMBEDDING_API_KEY")
        return self


@lru_cache
def get_settings() -> Settings:
    """返回单例配置对象，避免重复解析环境变量。"""

    return Settings()
