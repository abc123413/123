import html
import os
import re
from pathlib import Path
from typing import List

from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from pypdf import PdfReader

from app.config import Settings
from app.models.document import DocumentChunk


class DocumentProcessor:
    """负责文档解析、清洗和智能切块。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def extract_text(self, file_path: str) -> str:
        """根据文件类型提取纯文本。"""

        suffix = Path(file_path).suffix.lower()
        if suffix == ".pdf":
            return self._extract_pdf_text(file_path)
        if suffix == ".docx":
            return self._extract_docx_text(file_path)
        if suffix in {".txt", ".md"}:
            return Path(file_path).read_text(encoding="utf-8", errors="ignore")
        if suffix == ".html":
            return self._extract_html_text(file_path)
        raise ValueError(f"暂不支持解析该文件类型: {suffix}")

    def clean_text(self, raw_text: str) -> str:
        """进行基础清洗，去除无效字符与多余空白。"""

        text = html.unescape(raw_text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", " ", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"([A-Za-z])\.\n+(\d)", r"\1.\2", text)
        text = re.sub(r"(\d)\.\n+(\d)", r"\1.\2", text)
        text = re.sub(r"([：:])\n+", r"\1 ", text)
        text = re.sub(r"([A-Za-z0-9\u4e00-\u9fff])\n(?=[A-Za-z0-9\u4e00-\u9fff])", r"\1 ", text)
        text = re.sub(r"\n(?=\d+\.)", "\n\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def split_into_chunks(self, document_id: str, text: str, filename: str) -> List[DocumentChunk]:
        """按语义边界优先、字符长度兜底的方式切块。"""

        if not text.strip():
            return []

        paragraphs = [item.strip() for item in text.split("\n\n") if item.strip()]
        if self.settings.chunk_strategy == "fixed":
            return self._fixed_chunks(document_id, text, filename)

        chunks: List[DocumentChunk] = []
        current_parts: List[str] = []
        current_length = 0
        chunk_index = 0

        for paragraph in paragraphs:
            units = self._paragraph_to_units(paragraph)

            for unit in units:
                separator_length = 2 if current_parts else 0
                candidate_length = current_length + separator_length + len(unit)

                if candidate_length <= self.settings.chunk_size:
                    current_parts.append(unit)
                    current_length = candidate_length
                    continue

                if current_parts:
                    chunk_text = "\n\n".join(current_parts).strip()
                    chunks.append(self._build_chunk(document_id, filename, chunk_index, chunk_text))
                    chunk_index += 1

                    overlap_parts = self._select_overlap_parts(current_parts)
                    current_parts = overlap_parts[:]
                    current_length = self._joined_length(current_parts)

                if len(unit) > self.settings.chunk_size:
                    forced_parts = self._force_split(unit)
                    for part in forced_parts[:-1]:
                        chunks.append(self._build_chunk(document_id, filename, chunk_index, part))
                        chunk_index += 1
                    current_parts = [forced_parts[-1]] if forced_parts else []
                    current_length = self._joined_length(current_parts)
                    continue

                separator_length = 2 if current_parts else 0
                if current_length + separator_length + len(unit) > self.settings.chunk_size and current_parts:
                    chunk_text = "\n\n".join(current_parts).strip()
                    chunks.append(self._build_chunk(document_id, filename, chunk_index, chunk_text))
                    chunk_index += 1
                    current_parts = self._select_overlap_parts(current_parts)
                    current_length = self._joined_length(current_parts)

                current_parts.append(unit)
                current_length = self._joined_length(current_parts)

        if current_parts:
            chunks.append(self._build_chunk(document_id, filename, chunk_index, "\n\n".join(current_parts)))

        return chunks

    def process_document(self, document_id: str, file_path: str, filename: str) -> List[DocumentChunk]:
        """完成从文档到切块的完整流程。"""

        raw_text = self.extract_text(file_path)
        cleaned_text = self.clean_text(raw_text)
        return self.split_into_chunks(document_id, cleaned_text, filename)

    def _extract_pdf_text(self, file_path: str) -> str:
        reader = PdfReader(file_path)
        texts = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            texts.append(page_text)
        return "\n".join(texts)

    def _extract_docx_text(self, file_path: str) -> str:
        doc = DocxDocument(file_path)
        return "\n".join(paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip())

    def _extract_html_text(self, file_path: str) -> str:
        content = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(content, "html.parser")
        return soup.get_text(separator="\n")

    def _split_sentences(self, paragraph: str) -> List[str]:
        pattern = r"(?<=[。！？!?；;.\n])"
        parts = [item.strip() for item in re.split(pattern, paragraph) if item.strip()]
        return parts or [paragraph]

    def _force_split(self, text: str) -> List[str]:
        units = self._split_sentences(text)
        if len(units) <= 1:
            return self._force_split_by_chars(text)

        parts: List[str] = []
        current_parts: List[str] = []
        current_length = 0

        for unit in units:
            separator_length = 1 if current_parts else 0
            candidate_length = current_length + separator_length + len(unit)
            if candidate_length <= self.settings.chunk_size:
                current_parts.append(unit)
                current_length = candidate_length
                continue

            if current_parts:
                parts.append(" ".join(current_parts).strip())
            current_parts = [unit]
            current_length = len(unit)

            if len(unit) > self.settings.chunk_size:
                parts.extend(self._force_split_by_chars(unit))
                current_parts = []
                current_length = 0

        if current_parts:
            parts.append(" ".join(current_parts).strip())

        return [item for item in parts if item]

    def _force_split_by_chars(self, text: str) -> List[str]:
        size = self.settings.chunk_size
        result = []
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            result.append(text[start:end].strip())
            start = end
        return [item for item in result if item]

    def _fixed_chunks(self, document_id: str, text: str, filename: str) -> List[DocumentChunk]:
        parts = self._force_split(text)
        return [
            self._build_chunk(document_id, filename, index, content)
            for index, content in enumerate(parts)
        ]

    def _paragraph_to_units(self, paragraph: str) -> List[str]:
        if len(paragraph) >= self.settings.chunk_size:
            return self._split_sentences(paragraph)
        return [paragraph]

    def _select_overlap_parts(self, parts: List[str]) -> List[str]:
        if not parts or self.settings.chunk_overlap <= 0:
            return []

        selected: List[str] = []
        current_length = 0
        for part in reversed(parts):
            separator_length = 2 if selected else 0
            if current_length + separator_length + len(part) > self.settings.chunk_overlap:
                break
            selected.insert(0, part)
            current_length += separator_length + len(part)
        return selected

    def _joined_length(self, parts: List[str]) -> int:
        if not parts:
            return 0
        return sum(len(part) for part in parts) + (2 * (len(parts) - 1))

    def _build_chunk(self, document_id: str, filename: str, chunk_index: int, content: str) -> DocumentChunk:
        chunk_id = f"{document_id}_{chunk_index}"
        normalized_content = re.sub(r"\n{3,}", "\n\n", content).strip()
        return DocumentChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            content=normalized_content,
            chunk_index=chunk_index,
            metadata={
                "filename": filename,
                "chunk_index": chunk_index,
                "content_length": len(normalized_content),
            },
        )
