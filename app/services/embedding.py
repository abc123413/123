import hashlib
import json
import os
from typing import List

import httpx
import torch
from tenacity import retry, stop_after_attempt, wait_exponential
from transformers import AutoModel, AutoTokenizer

from app.config import Settings


class EmbeddingService:
    """统一封装本地和 OpenAI 兼容两种向量化实现。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._local_model = None
        self._local_tokenizer = None
        self._torch_device = self._resolve_torch_device(settings.embedding_device)
        if self.settings.embedding_cache_enabled:
            os.makedirs(self.settings.embedding_cache_dir, exist_ok=True)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """批量向量化文本，并优先命中缓存。"""

        if not texts:
            return []

        vectors: List[List[float]] = [[] for _ in texts]
        uncached_texts: List[str] = []
        uncached_indexes: List[int] = []

        for index, text in enumerate(texts):
            cached = self._load_cache(text)
            if cached is not None:
                self._validate_vector_dimension(cached)
                vectors[index] = cached
            else:
                uncached_texts.append(text)
                uncached_indexes.append(index)

        if uncached_texts:
            for start in range(0, len(uncached_texts), self.settings.embedding_batch_size):
                batch = uncached_texts[start : start + self.settings.embedding_batch_size]
                batch_vectors = self._embed_batch(batch)
                for offset, vector in enumerate(batch_vectors):
                    self._validate_vector_dimension(vector)
                    text = batch[offset]
                    global_index = uncached_indexes[start + offset]
                    vectors[global_index] = vector
                    self._save_cache(text, vector)

        return vectors

    def embed_query(self, query: str) -> List[float]:
        """向量化单条查询文本。"""

        result = self.embed_texts([query])
        return result[0] if result else []

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        if self.settings.embedding_provider == "local":
            return self._embed_batch_local(texts)
        return self._embed_batch_openai(texts)

    def _embed_batch_local(self, texts: List[str]) -> List[List[float]]:
        tokenizer, model = self._get_local_components()
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {key: value.to(self._torch_device) for key, value in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)
            token_embeddings = outputs.last_hidden_state
            attention_mask = encoded["attention_mask"]
            embeddings = self._mean_pooling(token_embeddings, attention_mask)
            if self.settings.embedding_normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().tolist()

    @retry(wait=wait_exponential(min=1, max=8), stop=stop_after_attempt(3), reraise=True)
    def _embed_batch_openai(self, texts: List[str]) -> List[List[float]]:
        url = f"{self.settings.embedding_api_base.rstrip('/')}/embeddings"
        payload = {
            "model": self.settings.embedding_model,
            "input": texts,
        }
        headers = {
            "Authorization": f"Bearer {self.settings.embedding_api_key}",
            "Content-Type": "application/json",
        }

        with httpx.Client(timeout=self.settings.embedding_timeout) as client:
            response = client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        return [item["embedding"] for item in data["data"]]

    def _get_local_components(self):
        if self._local_tokenizer is None or self._local_model is None:
            model_path = self.settings.embedding_model
            if not os.path.exists(model_path):
                raise ValueError(f"本地 embedding 模型目录不存在: {model_path}")

            self._local_tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
            )
            self._local_model = AutoModel.from_pretrained(
                model_path,
                local_files_only=True,
            )
            self._local_model.to(self._torch_device)
            self._local_model.eval()

            hidden_size = getattr(self._local_model.config, "hidden_size", None)
            if hidden_size is not None and hidden_size != self.settings.embedding_dimension:
                raise ValueError(
                    "本地 embedding 维度与配置不一致: "
                    f"模型实际维度={hidden_size}, 配置维度={self.settings.embedding_dimension}"
                )

        return self._local_tokenizer, self._local_model

    def _mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        masked_embeddings = token_embeddings * mask
        summed = torch.sum(masked_embeddings, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def _resolve_torch_device(self, configured_device: str) -> str:
        normalized = configured_device.strip().lower()
        if normalized == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return normalized

    def _validate_vector_dimension(self, vector: List[float]) -> None:
        if len(vector) != self.settings.embedding_dimension:
            raise ValueError(
                "向量维度不匹配: "
                f"期望 {self.settings.embedding_dimension}, 实际 {len(vector)}"
            )

    def _cache_path(self, text: str) -> str:
        provider = self.settings.embedding_provider
        model = self.settings.embedding_model.replace("/", "_")
        key_source = f"{provider}:{model}:{self.settings.embedding_dimension}:{text}"
        key = hashlib.sha256(key_source.encode("utf-8")).hexdigest()
        return os.path.join(self.settings.embedding_cache_dir, f"{key}.json")

    def _load_cache(self, text: str) -> List[float] | None:
        if not self.settings.embedding_cache_enabled:
            return None
        path = self._cache_path(text)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    def _save_cache(self, text: str, vector: List[float]) -> None:
        if not self.settings.embedding_cache_enabled:
            return
        path = self._cache_path(text)
        with open(path, "w", encoding="utf-8") as file:
            json.dump(vector, file, ensure_ascii=False)
