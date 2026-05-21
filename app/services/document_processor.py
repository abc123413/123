import base64
import html
import io
import re
import time
from pathlib import Path
from typing import List

import fitz
import httpx
import pytesseract
from loguru import logger
from bs4 import BeautifulSoup
from docx import Document as DocxDocument
from PIL import Image, ImageOps, ImageFilter
from pypdf import PdfReader

from app.config import Settings
from app.models.document import DocumentChunk


class DocumentProcessor:
    """Document parsing, OCR/VLM enrichment, cleanup and chunking."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        if self.settings.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = self.settings.tesseract_cmd

    def extract_text(self, file_path: str) -> str:
        suffix = Path(file_path).suffix.lower()
        if suffix == ".pdf":
            return self._extract_pdf_text(file_path)
        if suffix == ".docx":
            return self._extract_docx_text(file_path)
        if suffix in {".txt", ".md"}:
            return Path(file_path).read_text(encoding="utf-8", errors="ignore")
        if suffix == ".html":
            return self._extract_html_text(file_path)
        if suffix in {".png", ".jpg", ".jpeg"}:
            return self._extract_image_text(file_path)
        raise ValueError(f"Unsupported file type: {suffix}")

    def clean_text(self, raw_text: str) -> str:
        text = html.unescape(raw_text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", " ", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+\n", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"([A-Za-z])\.\n+(\d)", r"\1.\2", text)
        text = re.sub(r"(\d)\.\n+(\d)", r"\1.\2", text)
        text = re.sub(r"([。！？；：])\n+", r"\1 ", text)
        text = re.sub(r"([A-Za-z0-9\u4e00-\u9fff])\n(?=[A-Za-z0-9\u4e00-\u9fff])", r"\1 ", text)
        text = re.sub(r"\n(?=\d+\.)", "\n\n", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def split_into_chunks(self, document_id: str, text: str, filename: str) -> List[DocumentChunk]:
        if not text.strip():
            return []

        paragraphs = [item.strip() for item in text.split("\n\n") if item.strip()]
        if self.settings.chunk_strategy == "fixed":
            return self._fixed_chunks(document_id, text, filename)

        chunks: List[DocumentChunk] = []
        current_parts: List[str] = []
        current_modalities: List[str] = []
        current_length = 0
        chunk_index = 0
        current_title = ""

        for paragraph in paragraphs:
            detected_title = self._detect_title(paragraph)
            if detected_title:
                current_title = detected_title

            units = self._paragraph_to_units(paragraph)
            for unit in units:
                if self._should_drop_unit(unit):
                    continue

                unit_with_title = self._attach_title(unit, current_title)
                modality = self._detect_modality(unit_with_title)
                separator_length = 2 if current_parts else 0
                candidate_length = current_length + separator_length + len(unit_with_title)

                if candidate_length <= self.settings.chunk_size:
                    current_parts.append(unit_with_title)
                    current_modalities.append(modality)
                    current_length = candidate_length
                    continue

                if current_parts:
                    chunk_text = "\n\n".join(current_parts).strip()
                    chunk = self._build_chunk(document_id, filename, chunk_index, chunk_text, current_modalities)
                    if chunk is not None:
                        chunks.append(chunk)
                        chunk_index += 1

                    overlap_parts = self._select_overlap_parts(current_parts)
                    current_parts = overlap_parts[:]
                    current_modalities = [self._detect_modality(part) for part in current_parts]
                    current_length = self._joined_length(current_parts)

                if len(unit_with_title) > self.settings.chunk_size:
                    forced_parts = self._force_split(unit_with_title)
                    for part in forced_parts:
                        if self._should_drop_unit(part):
                            continue
                        chunk = self._build_chunk(
                            document_id,
                            filename,
                            chunk_index,
                            part,
                            [self._detect_modality(part)],
                        )
                        if chunk is not None:
                            chunks.append(chunk)
                            chunk_index += 1
                    current_parts = []
                    current_modalities = []
                    current_length = 0
                    continue

                if current_length + (2 if current_parts else 0) + len(unit_with_title) > self.settings.chunk_size and current_parts:
                    chunk_text = "\n\n".join(current_parts).strip()
                    chunk = self._build_chunk(document_id, filename, chunk_index, chunk_text, current_modalities)
                    if chunk is not None:
                        chunks.append(chunk)
                        chunk_index += 1
                    current_parts = self._select_overlap_parts(current_parts)
                    current_modalities = [self._detect_modality(part) for part in current_parts]
                    current_length = self._joined_length(current_parts)

                current_parts.append(unit_with_title)
                current_modalities.append(modality)
                current_length = self._joined_length(current_parts)

        if current_parts:
            chunk = self._build_chunk(document_id, filename, chunk_index, "\n\n".join(current_parts), current_modalities)
            if chunk is not None:
                chunks.append(chunk)

        return chunks

    def process_document(self, document_id: str, file_path: str, filename: str) -> List[DocumentChunk]:
        raw_text = self.extract_text(file_path)
        cleaned_text = self.clean_text(raw_text)
        return self.split_into_chunks(document_id, cleaned_text, filename)

    def _extract_pdf_text(self, file_path: str) -> str:
        reader = PdfReader(file_path)
        texts = []
        pdf_document = fitz.open(file_path)

        for page_index, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            ocr_text = ""
            vlm_text = ""

            if self.settings.enable_ocr or self.settings.vlm_enabled:
                pix = pdf_document.load_page(page_index).get_pixmap(matrix=fitz.Matrix(2.5, 2.5))
                image = Image.open(io.BytesIO(pix.tobytes("png")))
                if self.settings.enable_ocr:
                    ocr_text = self._run_ocr(image)
                if self.settings.vlm_enabled and page_index < self.settings.vlm_max_images_per_document:
                    vlm_text = self._run_vlm_on_image(image)

            combined = self._merge_modal_text(page_text, ocr_text, vlm_text)
            texts.append(combined)

        pdf_document.close()
        return "\n\n".join(item for item in texts if item.strip())

    def _extract_docx_text(self, file_path: str) -> str:
        doc = DocxDocument(file_path)
        paragraphs = [paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()]
        extras: List[str] = []

        if self.settings.enable_ocr or self.settings.vlm_enabled:
            image_count = 0
            for rel in doc.part.rels.values():
                if "image" not in rel.target_ref:
                    continue
                image_count += 1
                image_bytes = rel.target_part.blob
                image = Image.open(io.BytesIO(image_bytes))
                if self.settings.vlm_enabled and image_count <= self.settings.vlm_max_images_per_document:
                    vlm_text = self._run_vlm_on_image(image)
                    if vlm_text:
                        extras.append(f"[VLM_IMAGE_{image_count}]\n{vlm_text}")
                if self.settings.enable_ocr:
                    ocr_text = self._run_ocr(image)
                    if ocr_text:
                        extras.append(f"[OCR_IMAGE_{image_count}]\n{ocr_text}")

        return "\n".join(item for item in paragraphs + extras if item and item.strip())

    def _extract_html_text(self, file_path: str) -> str:
        content = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(content, "html.parser")
        return soup.get_text(separator="\n")

    def _extract_image_text(self, file_path: str) -> str:
        image = Image.open(file_path)
        ocr_text = self._run_ocr(image) if self.settings.enable_ocr else ""
        vlm_text = self._run_vlm_on_image(image) if self.settings.vlm_enabled else ""
        combined = self._merge_modal_text("", ocr_text, vlm_text)
        if not combined:
            raise ValueError("No text could be extracted from image document")
        return combined

    def _run_ocr(self, image: Image.Image) -> str:
        try:
            prepared = self._prepare_image_for_ocr(image)
            started_at = time.perf_counter()
            result = pytesseract.image_to_string(prepared, lang=self.settings.ocr_language, config="--psm 6").strip()
            duration_ms = (time.perf_counter() - started_at) * 1000
            logger.info("OCR succeeded | duration_ms={:.2f}", duration_ms)
            return self._normalize_ocr_text(result)
        except Exception:
            logger.exception("OCR failed")
            return ""

    def _prepare_image_for_ocr(self, image: Image.Image) -> Image.Image:
        grayscale = ImageOps.grayscale(image)
        enhanced = ImageOps.autocontrast(grayscale)
        sharpened = enhanced.filter(ImageFilter.SHARPEN)
        return sharpened

    def _normalize_ocr_text(self, text: str) -> str:
        lines = []
        for line in text.splitlines():
            normalized = " ".join(line.split())
            if not normalized:
                continue
            if self._looks_like_ocr_noise(normalized):
                continue
            lines.append(normalized)
        return "\n".join(lines).strip()

    def _run_vlm_on_image(self, image: Image.Image) -> str:
        if self.settings.demo_mode or not self.settings.vlm_enabled or not self.settings.vlm_api_key:
            return ""

        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=90)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        image_url = self._build_vlm_image_url(encoded)
        payload = {
            "model": self.settings.vlm_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": self.settings.vlm_prompt},
                    ],
                }
            ],
            "temperature": 0.1,
            "max_tokens": 512,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {self.settings.vlm_api_key}",
            "Content-Type": "application/json",
        }
        try:
            started_at = time.perf_counter()
            with httpx.Client(timeout=self.settings.vlm_timeout) as client:
                response = client.post(
                    f"{self.settings.vlm_api_base.rstrip('/')}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                if response.is_error:
                    logger.error(
                        "VLM request failed | model={} status={} body={}",
                        self.settings.vlm_model,
                        response.status_code,
                        response.text[:1000],
                    )
                response.raise_for_status()
                data = response.json()
            duration_ms = (time.perf_counter() - started_at) * 1000
            logger.info("VLM request succeeded | model={} duration_ms={:.2f}", self.settings.vlm_model, duration_ms)
            content = data["choices"][0]["message"]["content"].strip()
            return self._normalize_vlm_text(content)
        except Exception:
            logger.exception("VLM request failed | model={}", self.settings.vlm_model)
            return ""

    def _normalize_vlm_text(self, text: str) -> str:
        text = re.sub(r"```(?:json)?", "", text)
        text = text.replace("```", "")
        text = re.sub(r"\n{3,}", "\n\n", text)
        lines = [" ".join(line.split()) for line in text.splitlines() if line.strip()]
        unique_lines = []
        seen = set()
        for line in lines:
            if line in seen:
                continue
            seen.add(line)
            unique_lines.append(line)
        return "\n".join(unique_lines[:8]).strip()

    def _build_vlm_image_url(self, encoded: str) -> str:
        api_base = self.settings.vlm_api_base.lower()
        if "open.bigmodel.cn" in api_base:
            return encoded
        return f"data:image/jpeg;base64,{encoded}"

    def _merge_modal_text(self, page_text: str, ocr_text: str, vlm_text: str) -> str:
        parts = []
        normalized_page = self._normalize_page_text(page_text)
        if normalized_page:
            parts.append(normalized_page)
        if vlm_text.strip():
            parts.append(f"[VLM]\n{vlm_text.strip()}")
        if ocr_text.strip():
            parts.append(f"[OCR]\n{ocr_text.strip()}")
        return "\n\n".join(parts).strip()

    def _normalize_page_text(self, text: str) -> str:
        lines = []
        for line in text.splitlines():
            normalized = " ".join(line.split())
            if not normalized:
                continue
            if self._looks_like_ocr_noise(normalized):
                continue
            lines.append(normalized)
        return "\n".join(lines).strip()

    def _split_sentences(self, paragraph: str) -> List[str]:
        pattern = r"(?<=[。！？；!?\.])"
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
        chunks: List[DocumentChunk] = []
        for index, content in enumerate(parts):
            if self._should_drop_unit(content):
                continue
            chunk = self._build_chunk(document_id, filename, index, content, [self._detect_modality(content)])
            if chunk is not None:
                chunks.append(chunk)
        return chunks

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

    def _detect_title(self, paragraph: str) -> str:
        if not self.settings.chunk_title_boost:
            return ""
        normalized = " ".join(paragraph.split())
        if not normalized:
            return ""
        if len(normalized) <= 40 and re.match(r"^(\d+(\.\d+)*\s+)?[\u4e00-\u9fffA-Za-z0-9].*$", normalized):
            return normalized
        return ""

    def _attach_title(self, content: str, title: str) -> str:
        if not title or title in content:
            return content
        return f"标题：{title}\n{content}"

    def _detect_modality(self, content: str) -> str:
        stripped = content.lstrip()
        if stripped.startswith("[VLM]") or stripped.startswith("[VLM_IMAGE_"):
            return "vlm"
        if stripped.startswith("[OCR]") or stripped.startswith("[OCR_IMAGE_"):
            return "ocr"
        return "text"

    def _should_drop_unit(self, content: str) -> bool:
        normalized = " ".join(content.split())
        if not normalized:
            return True
        modality = self._detect_modality(content)
        min_len = 20 if modality in {"vlm", "ocr"} else min(30, self.settings.chunk_min_size)
        if len(normalized) < min_len:
            return True
        if modality == "text" and self._looks_like_ocr_noise(normalized):
            return True
        if modality == "ocr" and self._looks_like_ocr_noise(normalized):
            return True
        return False

    def _looks_like_ocr_noise(self, text: str) -> bool:
        if not text:
            return True
        letters = sum(1 for ch in text if ch.isalpha())
        digits = sum(1 for ch in text if ch.isdigit())
        chinese = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
        spaces = text.count(" ")
        symbols = len(text) - letters - digits - chinese - spaces
        useful = letters + digits + chinese
        if useful == 0:
            return True
        if chinese == 0 and letters > 0 and symbols > useful * 0.4:
            return True
        if chinese == 0 and letters > 0 and re.search(r"(?:[A-Za-z]\s*){18,}", text):
            return True
        if len(text) >= 24 and useful / max(len(text), 1) < 0.45:
            return True
        return False

    def _build_chunk(
        self,
        document_id: str,
        filename: str,
        chunk_index: int,
        content: str,
        modalities: List[str],
    ) -> DocumentChunk | None:
        normalized_content = re.sub(r"\n{3,}", "\n\n", content).strip()
        if self._should_drop_unit(normalized_content):
            return None

        chunk_id = f"{document_id}_{chunk_index}"
        modality = self._resolve_chunk_modality(modalities)
        return DocumentChunk(
            chunk_id=chunk_id,
            document_id=document_id,
            content=normalized_content,
            chunk_index=chunk_index,
            metadata={
                "filename": filename,
                "chunk_index": chunk_index,
                "content_length": len(normalized_content),
                "modality": modality,
                "has_vlm": modality == "vlm",
                "has_ocr": modality in {"vlm", "ocr"},
            },
        )

    def _resolve_chunk_modality(self, modalities: List[str]) -> str:
        if "vlm" in modalities:
            return "vlm"
        if "ocr" in modalities:
            return "ocr"
        return "text"
