"""Nạp bộ báo VietNews vào một index riêng (mặc định ``data/index_vietnews``) để thử ở quy mô 100k.

    .venv\\Scripts\\python.exe scripts\\import_corpus.py                  # 99k bài train, chỉ từ vựng
    .venv\\Scripts\\python.exe scripts\\import_corpus.py --vectors         # thêm vector e5-small (vài giờ, tiếp tục được)
    .venv\\Scripts\\python.exe scripts\\import_corpus.py --limit 20000     # thử nhanh với ít bài hơn

Chạy lại an toàn: bài đã có bị bỏ qua; ``--vectors`` chỉ encode những bài còn thiếu vector.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from corpus import CORPUS_INDEX_DIR, doc_id, load_split  # noqa: E402
from documents import TextChunk, split_chunks  # noqa: E402
from indexer import LibraryIndex  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", default=str(CORPUS_INDEX_DIR))
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-docs", type=int, default=10_000, help="số bài mỗi segment")
    ap.add_argument("--vectors", action="store_true", help="encode vector e5-small cho các bài chưa có")
    args = ap.parse_args()

    t = time.perf_counter()
    articles = load_split(args.split, args.limit)
    print(f"Đọc {len(articles):,} bài {args.split} trong {time.perf_counter() - t:.1f}s", flush=True)

    with LibraryIndex(args.index_dir) as index:
        todo = [a for a in articles if not index.has(doc_id(args.split, a["guid"]))]
        print(f"Cần index {len(todo):,} bài (đã có {len(articles) - len(todo):,})", flush=True)
        started = time.perf_counter()
        for start in range(0, len(todo), args.batch_docs):
            batch = todo[start : start + args.batch_docs]
            items = []
            for a in batch:
                chunks = split_chunks([TextChunk(a["article"])])
                if chunks:
                    items.append((doc_id(args.split, a["guid"]), a["title"] or str(a["guid"]),
                                  [(c.text, None) for c in chunks]))
            t = time.perf_counter()
            added = index.add_documents(items)
            done = start + len(batch)
            rate = done / (time.perf_counter() - started)
            print(f"  {done:,}/{len(todo):,} bài: +{added:,} trong {time.perf_counter() - t:.1f}s "
                  f"(~{(len(todo) - done) / rate / 60:.1f} phút còn lại)", flush=True)
        print("Index từ vựng:", index.stats(), flush=True)

        if args.vectors:
            from embedder import FastEncoder

            encoder = FastEncoder()
            started = time.perf_counter()
            last = [0.0]

            def progress(done: int, total: int) -> None:
                now = time.perf_counter()
                if now - last[0] > 30 or done == total:
                    last[0] = now
                    rate = done / max(now - started, 1e-9)
                    print(f"  vector {done:,}/{total:,} đoạn ({rate:.0f} đoạn/s, "
                          f"~{(total - done) / max(rate, 1e-9) / 60:.0f} phút còn lại)", flush=True)

            n = index.add_vectors(encoder, batch_size=32, progress=progress)
            print(f"Đã encode {n:,} bài trong {(time.perf_counter() - started) / 60:.1f} phút:", index.stats(), flush=True)


if __name__ == "__main__":
    main()
