"""Embedding nhanh trên CPU cho tầng ngữ nghĩa: intfloat/multilingual-e5-small.

~118M tham số, 384 chiều, tối đa 512 token, hỗ trợ tiếng Việt. Nhanh hơn DeepX
(~9 s/đoạn trên CPU máy này) vài chục lần nên dùng được cho cả encode kho lẫn
truy vấn. So sánh đối xứng (đoạn với đoạn) nên cả hai phía đều dùng tiền tố
``"query: "`` theo khuyến nghị của model card.

Cùng giao diện với ``semantic.SemanticEncoder``: ``encode(list[str]) -> (N, D)``
đã chuẩn hóa L2, nên ``indexer`` dùng được bất kỳ encoder nào.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "multilingual-e5-small"
MODEL_NAME = "multilingual-e5-small"
REQUIRED_FILES = ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")
MAX_TOKENS = 512
PREFIX = "query: "


def missing_files(model_dir: Path = MODEL_DIR) -> list[str]:
    return [n for n in REQUIRED_FILES if not (model_dir / n).is_file() or (model_dir / n).stat().st_size == 0]


class FastEncoder:
    """Nạp e5-small lên CPU (float32) và mã hóa văn bản bằng mean pooling."""

    name = MODEL_NAME

    def __init__(self, model_dir: Path | str = MODEL_DIR, threads: int | None = None):
        model_dir = Path(model_dir)
        missing = missing_files(model_dir)
        if missing:
            raise FileNotFoundError(f"Thiếu file model trong {model_dir}: {', '.join(missing)}")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        import torch
        from transformers import AutoModel, AutoTokenizer

        if threads:
            torch.set_num_threads(threads)
        started = time.perf_counter()
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_dir, local_files_only=True).eval().requires_grad_(False)
        self.dim = int(self.model.config.hidden_size)
        self.load_seconds = time.perf_counter() - started

    def encode(self, texts: list[str] | str, batch_size: int = 16, max_tokens: int = MAX_TOKENS,
               progress=None) -> np.ndarray:
        """``progress(done, total)`` được gọi sau mỗi batch nếu có."""
        torch = self.torch
        single = isinstance(texts, str)
        texts = [texts] if single else list(texts)
        result = np.zeros((len(texts), self.dim), dtype=np.float32)
        # Gom các đoạn dài tương đương vào cùng batch để giảm token đệm.
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        with torch.inference_mode():
            for start in range(0, len(order), batch_size):
                idx = order[start : start + batch_size]
                enc = self.tokenizer(
                    [PREFIX + texts[i] for i in idx], padding=True, truncation=True,
                    max_length=max_tokens, return_tensors="pt",
                )
                hidden = self.model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                result[idx] = torch.nn.functional.normalize(pooled, dim=-1).numpy()
                if progress is not None:
                    progress(min(start + batch_size, len(order)), len(order))
        return result[0] if single else result
