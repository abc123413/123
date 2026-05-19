import json
import re
from typing import AsyncGenerator, List

import httpx

from app.config import Settings
from app.models.document import CitationItem, SearchResult


class LLMService:
    """Answer generation service with real and demo modes."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def generate_answer(self, query: str, results: List[SearchResult], stream: bool = False) -> str | AsyncGenerator[str, None]:
        prompt = self.build_prompt(query, results)
        if self.settings.demo_mode:
            return self._stream_demo_answer(query, results) if stream else self._demo_raw_answer(query, results)
        if stream:
            return self.stream_answer(prompt)
        return await self._request_chat_completion(prompt)

    async def generate_structured_answer(self, query: str, results: List[SearchResult]) -> dict:
        raw_answer = await self.generate_answer(query, results, stream=False)
        if not isinstance(raw_answer, str):
            raise ValueError("Failed to generate structured answer")
        return self.parse_structured_answer(raw_answer, results)

    def build_prompt(self, query: str, results: List[SearchResult]) -> str:
        contexts = []
        selected_results = self._select_context_results(results)
        for index, result in enumerate(selected_results, start=1):
            metadata = result.metadata or {}
            compressed_content = self._compress_context(result.content)
            contexts.append(f"[{index}] 来源文档: {metadata.get('filename', 'unknown')}\n{compressed_content}")

        context_text = "\n\n".join(contexts) if contexts else "无可用上下文"
        prompt = (
            f"{self.settings.prompt_system_message}\n\n"
            f"检索上下文如下：\n{context_text}\n\n"
            f"用户问题：{query}\n\n"
            "请直接输出 JSON，不要输出 Markdown，不要输出代码块，不要输出 JSON 之外的解释。\n"
            'JSON 结构必须为：{"conclusion":"", "key_points":[""], "citations":["[1]","[2]"]}\n'
            "字段要求：\n"
            "- conclusion：用 2-4 句直接回答用户问题。\n"
            "- key_points：2-6 条关键要点，每条尽量简短。\n"
            "- citations：列出本回答主要依据了哪些片段编号，元素必须类似 [1]、[2]。\n\n"
            "约束要求：\n"
            "- 只能依据上面的检索上下文回答，不要补充上下文之外的知识。\n"
            "- 如果信息不足，明确写“依据不足”。\n"
            "- 引用编号只能使用上面出现过的片段编号。\n"
            "- 不要输出“我认为”“可能”“推测”等措辞。"
        )
        return prompt

    def _select_context_results(self, results: List[SearchResult]) -> List[SearchResult]:
        selected: List[SearchResult] = []
        for result in results:
            if len(selected) >= self.settings.context_max_chunks:
                break
            if self._is_low_signal_chunk(result.content):
                continue
            selected.append(result)

        if selected:
            return selected
        return results[: self.settings.context_max_chunks]

    def _compress_context(self, content: str) -> str:
        text = content.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)

        segments = [segment.strip() for segment in re.split(r"\n\n+", text) if segment.strip()]
        refined_segments: List[str] = []

        for segment in segments:
            normalized = " ".join(segment.split())
            if not normalized:
                continue
            if self._is_directory_like(normalized):
                continue
            refined_segments.append(normalized)

        if not refined_segments:
            refined_segments = [" ".join(text.split())]

        return "\n".join(refined_segments[:4]).strip()

    def _is_low_signal_chunk(self, content: str) -> bool:
        normalized = " ".join(content.split())
        if len(normalized) < 40:
            return True

        keyword_hits = sum(
            1
            for keyword in ["定义", "原理", "优势", "流程", "组件", "步骤", "优化", "场景", "问题"]
            if keyword in normalized
        )
        if keyword_hits >= 2:
            return False

        return self._is_directory_like(normalized)

    def _is_directory_like(self, text: str) -> bool:
        labels = ["目录", "版本", "适用", "日期", "核心定义", "核心优势", "整体架构流程", "主流", "简易搭建流程"]
        hit_count = sum(1 for label in labels if label in text)
        numbered_items = len(re.findall(r"\b\d+\.", text))
        return hit_count >= 4 and numbered_items <= 3

    def parse_structured_answer(self, raw_answer: str, results: List[SearchResult]) -> dict:
        parsed = self._try_parse_json(raw_answer)
        if parsed is None:
            parsed = self._fallback_parse_text(raw_answer)

        conclusion = str(parsed.get("conclusion", "")).strip()
        key_points = [
            str(item).strip()
            for item in parsed.get("key_points", [])
            if str(item).strip()
        ]
        citation_refs = [
            str(item).strip()
            for item in parsed.get("citations", [])
            if str(item).strip()
        ]

        citations = self._build_citations(citation_refs, results)
        answer = self._compose_answer(conclusion, key_points, citations)

        return {
            "answer": answer,
            "conclusion": conclusion,
            "key_points": key_points,
            "citations": citations,
        }

    def _try_parse_json(self, raw_answer: str) -> dict | None:
        text = raw_answer.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        return None

    def _fallback_parse_text(self, raw_answer: str) -> dict:
        conclusion = raw_answer.strip()
        key_points = []
        citation_refs = re.findall(r"\[\d+\]", raw_answer)
        return {
            "conclusion": conclusion,
            "key_points": key_points,
            "citations": citation_refs,
        }

    def _build_citations(self, citation_refs: List[str], results: List[SearchResult]) -> List[CitationItem]:
        citation_map = {}
        for index, result in enumerate(self._select_context_results(results), start=1):
            ref = f"[{index}]"
            metadata = result.metadata or {}
            citation_map[ref] = CitationItem(ref=ref, filename=metadata.get("filename", ""))

        citations: List[CitationItem] = []
        for ref in citation_refs:
            if ref in citation_map:
                citations.append(citation_map[ref])

        if citations:
            return citations

        return list(citation_map.values())[:2]

    def _compose_answer(self, conclusion: str, key_points: List[str], citations: List[CitationItem]) -> str:
        parts: List[str] = []
        if conclusion:
            parts.append(f"结论：{conclusion}")
        if key_points:
            bullet_text = "\n".join(f"- {item}" for item in key_points)
            parts.append(f"关键要点：\n{bullet_text}")
        if citations:
            refs = "".join(item.ref for item in citations)
            parts.append(f"依据：{refs}")
        return "\n\n".join(parts).strip()

    def _demo_raw_answer(self, query: str, results: List[SearchResult]) -> str:
        selected_results = self._select_context_results(results)
        conclusion = self._build_demo_conclusion(query, selected_results)
        key_points = self._build_demo_key_points(selected_results)
        citations = [f"[{index}]" for index, _ in enumerate(selected_results[:2], start=1)]
        payload = {
            "conclusion": conclusion,
            "key_points": key_points,
            "citations": citations,
        }
        return json.dumps(payload, ensure_ascii=False)

    def _build_demo_conclusion(self, query: str, results: List[SearchResult]) -> str:
        if not results:
            return f"{self.settings.demo_answer_prefix} 当前没有检索到可用片段，无法根据上下文回答“{query}”。"

        filenames = []
        for item in results:
            filename = (item.metadata or {}).get("filename", "unknown")
            if filename not in filenames:
                filenames.append(filename)

        return (
            f"{self.settings.demo_answer_prefix} 系统已成功完成检索与回答链路演示。"
            f" 当前问题是“{query}”，答案基于已检索到的文档片段生成，主要来源于：{', '.join(filenames[:3])}。"
        )

    def _build_demo_key_points(self, results: List[SearchResult]) -> List[str]:
        if not results:
            return ["未命中有效上下文，因此只返回演示性结果。"]

        points: List[str] = []
        for index, result in enumerate(results[:3], start=1):
            metadata = result.metadata or {}
            filename = metadata.get("filename", "unknown")
            content = " ".join(result.content.split())
            snippet = content[:80]
            points.append(f"[{index}] 来源 {filename}，命中内容片段：{snippet}")
        points.append("当前为 Demo 模式，未调用真实大模型 API。")
        return points

    async def _stream_demo_answer(self, query: str, results: List[SearchResult]) -> AsyncGenerator[str, None]:
        structured = self.parse_structured_answer(self._demo_raw_answer(query, results), results)
        text = structured["answer"]
        for chunk in re.findall(r".{1,24}", text, flags=re.S):
            yield chunk

    async def _request_chat_completion(self, prompt: str) -> str:
        if not self.settings.llm_api_key:
            raise ValueError("LLM_API_KEY is required when demo mode is disabled")

        url = f"{self.settings.llm_api_base.rstrip('/')}/chat/completions"
        payload = {
            "model": self.settings.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.settings.llm_temperature,
            "max_tokens": self.settings.llm_max_tokens,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self.settings.llm_api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=self.settings.llm_timeout) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        return data["choices"][0]["message"]["content"]

    async def stream_answer(self, prompt: str) -> AsyncGenerator[str, None]:
        if not self.settings.llm_api_key:
            raise ValueError("LLM_API_KEY is required when demo mode is disabled")

        url = f"{self.settings.llm_api_base.rstrip('/')}/chat/completions"
        payload = {
            "model": self.settings.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.settings.llm_temperature,
            "max_tokens": self.settings.llm_max_tokens,
            "stream": True,
        }
        headers = {
            "Authorization": f"Bearer {self.settings.llm_api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=self.settings.llm_timeout) as client:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    payload_line = line.removeprefix("data:").strip()
                    if payload_line == "[DONE]":
                        break
                    try:
                        chunk = httpx.Response(200, content=payload_line).json()
                    except Exception:
                        continue
                    delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                    if delta:
                        yield delta
