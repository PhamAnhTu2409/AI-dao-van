"""Đo thời gian truy vấn của indexer.py trên kho giả lập (không cần model).

    .venv\\Scripts\\python.exe scripts\\bench_index.py --docs 2000 --chunks 50

Encoder giả trả vector ngẫu nhiên có cấu trúc: đoạn "viết lại" có vector gần đoạn
gốc, nên kiểm được cả đường ngữ nghĩa lẫn đúng/sai của kết quả.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from indexer import LibraryIndex  # noqa: E402

VOCAB = [f"từ{i}" for i in range(20_000)]
DIM = 384


class FakeEncoder:
    """Vector = hash(chủ đề của đoạn) + nhiễu nhỏ; đoạn cùng chủ đề -> cosine cao."""

    name = "fake"

    def encode(self, texts, batch_size=8):
        out = []
        for text in texts:
            topic = text.split()[0]
            seed = int.from_bytes(hashlib.blake2b(topic.encode(), digest_size=4).digest(), "little")
            base = np.random.default_rng(seed).standard_normal(DIM)
            # Nhiễu cố định theo nội dung (không dùng hash() vì Python đổi seed mỗi lần chạy).
            noise_seed = int.from_bytes(hashlib.blake2b(text.encode(), digest_size=4).digest(), "little")
            noise = np.random.default_rng(noise_seed).standard_normal(DIM) * 0.25
            v = base + noise
            out.append(v / np.linalg.norm(v))
        return np.asarray(out, dtype=np.float32)


def make_chunk(rng, topic: str, words: int = 180) -> str:
    return topic + " " + " ".join(rng.choice(VOCAB, words - 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=1000)
    ap.add_argument("--chunks", type=int, default=50)
    ap.add_argument("--query-chunks", type=int, default=60)
    ap.add_argument("--dim", type=int, default=384, help="số chiều vector giả")
    ap.add_argument("--coarse-dim", type=int, default=None, help="bật tìm 2 tầng (Matryoshka)")
    args = ap.parse_args()
    global DIM
    DIM = args.dim
    rng = np.random.default_rng(0)
    enc = FakeEncoder()

    with tempfile.TemporaryDirectory() as tmp:
        index = LibraryIndex(tmp, vector_model="fake", coarse_dim=args.coarse_dim)
        try:
            library: list[list[str]] = []
            t = time.perf_counter()
            for d in range(args.docs):
                chunks = [make_chunk(rng, f"chủđề{d}_{c}") for c in range(args.chunks)]
                library.append(chunks)
                index.add_document(f"doc{d:06d}", f"Tài liệu {d}.pdf", [(c, 1 + i // 3) for i, c in enumerate(chunks)], enc)
            build = time.perf_counter() - t
            t = time.perf_counter()
            index.reload()
            print(f"Index {args.docs} tài liệu × {args.chunks} đoạn: build {build:.1f}s, nạp {time.perf_counter() - t:.2f}s")
            print("  ", index.stats())

            # Tài liệu mới: 1/4 chép nguyên văn từ doc 7, 1/4 chép rồi sửa 1/8 số từ từ doc 99,
            # 1/4 "viết lại" từ doc 42 (cùng chủ đề, khác hẳn từ), 1/4 hoàn toàn mới.
            q = args.query_chunks // 4
            copied = library[7][:q]
            edited = []
            for chunk in library[99][:q]:
                words = chunk.split()
                for pos in rng.choice(len(words), len(words) // 8, replace=False):
                    words[pos] = str(rng.choice(VOCAB))
                edited.append(" ".join(words))
            paraphrased = [make_chunk(rng, f"chủđề42_{c}") for c in range(q)]
            fresh = [make_chunk(rng, f"mới{c}") for c in range(args.query_chunks - 3 * q)]
            query = [(c, None) for c in copied + edited + paraphrased + fresh]

            for label, encoder in (("chỉ từ vựng", None), ("từ vựng + ngữ nghĩa", enc)):
                t = time.perf_counter()
                report = index.query(query, encoder)
                total = time.perf_counter() - t
                top = report.documents[:3]
                print(f"\n[{label}] {len(query)} đoạn truy vấn: {total * 1000:.0f} ms "
                      f"(encode {report.encoded_chunks} đoạn) — " +
                      ", ".join(f"{k} {v * 1000:.0f}ms" for k, v in report.timings.items()))
                for r in top:
                    sem = "—" if r.best_semantic is None else f"{r.best_semantic:.2f}"
                    print(f"   {r.name}: phủ {r.coverage:.0%}, 5-gram {r.exact_5gram:.0%}, "
                          f"tfidf {r.best_lexical:.2f}, ngữ nghĩa {sem}, {len(r.matches)} cặp đoạn")
                names = [r.name for r in top]
                assert {"Tài liệu 7.pdf", "Tài liệu 99.pdf"} <= set(names), names
                if encoder is not None:
                    assert "Tài liệu 42.pdf" in names, names
                assert report.flagged_chunks == (3 * q if encoder is not None else 2 * q), report.flagged_chunks
        finally:
            index.close()


if __name__ == "__main__":
    main()
