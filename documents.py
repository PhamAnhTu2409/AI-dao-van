"""Trích xuất văn bản PDF/DOCX/TXT và chia đoạn — dùng chung cho app và indexer."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from docx import Document
from pypdf import PdfReader

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ALLOWED_SUFFIXES = {".pdf", ".docx", ".txt"}
CHUNK_WORDS = 180
CHUNK_OVERLAP = 35


@dataclass
class TextChunk:
    text: str
    page: int | None = None


def _decode_txt(payload: bytes) -> str:
    if payload.startswith((b"\xff\xfe", b"\xfe\xff")):
        return payload.decode("utf-16")
    try:
        return payload.decode("utf-8-sig")
    except UnicodeDecodeError:
        return payload.decode("cp1258", errors="replace")


def extract_text(name: str, payload: bytes) -> list[TextChunk]:
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise ValueError("Chỉ nhận PDF, DOCX hoặc TXT.")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise ValueError("Tệp vượt quá giới hạn 25 MB.")

    if suffix == ".pdf":
        reader = PdfReader(io.BytesIO(payload), strict=False)
        pages = [
            TextChunk(text=page.extract_text() or "", page=index)
            for index, page in enumerate(reader.pages, start=1)
        ]
        pages = [page for page in pages if page.text.strip()]
        if not pages:
            raise ValueError("PDF không có lớp văn bản đọc được; có thể đây là PDF scan. Cần OCR trước khi kiểm tra.")
        return pages

    if suffix == ".docx":
        document = Document(io.BytesIO(payload))
        parts = [p.text for p in document.paragraphs if p.text.strip()]
        parts.extend(
            " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
            for table in document.tables
            for row in table.rows
        )
        text = "\n".join(part for part in parts if part.strip())
        if not text.strip():
            raise ValueError("DOCX không chứa văn bản có thể trích xuất.")
        return [TextChunk(text=text)]

    text = _decode_txt(payload)
    if not text.strip():
        raise ValueError("Tệp TXT rỗng.")
    return [TextChunk(text=text)]


def split_chunks(pages: Iterable[TextChunk]) -> list[TextChunk]:
    """Cắt theo cửa sổ ``CHUNK_WORDS`` từ, chồng lấn ``CHUNK_OVERLAP``; giữ số trang."""
    result: list[TextChunk] = []
    for page in pages:
        words = page.text.split()
        start = 0
        while start < len(words):
            end = min(len(words), start + CHUNK_WORDS)
            result.append(TextChunk(" ".join(words[start:end]), page.page))
            if end == len(words):
                break
            start = end - CHUNK_OVERLAP
    return result


def load_chunks(name: str, payload: bytes) -> list[tuple[str, int | None]]:
    """Định dạng ``indexer`` cần: danh sách ``(text, page)``."""
    return [(c.text, c.page) for c in split_chunks(extract_text(name, payload))]
