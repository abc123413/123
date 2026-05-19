from typing import List

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from app.config import Settings
from app.models.document import SearchResult


class RerankerService:
    """使用本地或远程 Cross-Encoder 模型对召回结果做精排。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None
        self._tokenizer = None
        self._torch_device = self._resolve_torch_device("cpu")

    def rerank(self, query: str, results: List[SearchResult], use_rerank: bool | None = None) -> List[SearchResult]:
        """根据开关决定是否执行重排序。"""

        enabled = self.settings.rerank_enabled if use_rerank is None else use_rerank
        if not enabled or not results:
            return results[: self.settings.rerank_final_k]

        tokenizer, model = self._get_local_components()
        candidates = results[: self.settings.rerank_initial_k]
        pairs = [(query, item.content) for item in candidates]

        with torch.no_grad():
            encoded = tokenizer(
                [pair[0] for pair in pairs],
                [pair[1] for pair in pairs],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {key: value.to(self._torch_device) for key, value in encoded.items()}
            outputs = model(**encoded)
            logits = outputs.logits.squeeze(-1)
            scores = logits.detach().cpu().tolist()

        if isinstance(scores, float):
            scores = [scores]

        for item, score in zip(candidates, scores):
            item.rerank_score = float(score)

        candidates.sort(key=lambda item: item.rerank_score or 0.0, reverse=True)
        return candidates[: self.settings.rerank_final_k]

    def _get_local_components(self):
        if self._tokenizer is None or self._model is None:
            model_path = self.settings.rerank_model
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=True,
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                local_files_only=True,
            )
            self._model.to(self._torch_device)
            self._model.eval()

        return self._tokenizer, self._model

    def _resolve_torch_device(self, configured_device: str) -> str:
        normalized = configured_device.strip().lower()
        if normalized == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return normalized
