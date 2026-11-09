"""Kho thử nghiệm: báo VietNews (harouzie/vietnews, Apache-2.0) đã tải về ``data/corpus``.

Văn bản gốc đã tách từ (``Cơ_quan``, dấu câu cách rời); ``detokenize`` trả lại dạng viết thường
để giống tài liệu thật. Tài liệu kho có id ``corpus:vietnews:<guid>`` — app không coi chúng là
tệp trong ``data/library`` nên không tự gỡ khỏi index.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORPUS_DIR = ROOT / "data" / "corpus"
CORPUS_INDEX_DIR = ROOT / "data" / "index_vietnews"
CORPUS_PREFIX = "corpus:"
SPLITS = {
    "train": "train-00000-of-00001-faa04287fdc0e299.parquet",
    "validation": "validation-00000-of-00001-10a6e2d4aced7ee4.parquet",
}


def detokenize(text: str) -> str:
    text = text.replace("_", " ")
    text = re.sub(r"\s+([,.;:!?%)\]}…])", r"\1", text)
    text = re.sub(r"([(\[{])\s+", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def doc_id(split: str, guid: int) -> str:
    return f"{CORPUS_PREFIX}vietnews:{split}:{guid}"


def load_split(split: str, limit: int | None = None) -> list[dict]:
    """Danh sách bài: {guid, title, abstract, article} đã detokenize."""
    import pyarrow.parquet as pq

    path = CORPUS_DIR / SPLITS[split]
    if not path.exists():
        raise FileNotFoundError(f"Chưa có {path}. Tải bộ VietNews theo README trước.")
    table = pq.read_table(path)
    if limit is not None:
        table = table.slice(0, limit)
    cols = {name: table.column(name).to_pylist() for name in ("guid", "title", "abstract", "article")}
    return [
        {"guid": int(g), "title": detokenize(t or ""), "abstract": detokenize(a or ""), "article": detokenize(x or "")}
        for g, t, a, x in zip(cols["guid"], cols["title"], cols["abstract"], cols["article"])
    ]
