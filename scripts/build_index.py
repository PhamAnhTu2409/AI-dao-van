"""Index (hoặc cập nhật) toàn bộ data/library. Chạy lại an toàn: chỉ xử lý tài liệu mới.

    .venv\\Scripts\\python.exe scripts\\build_index.py            # từ vựng + vector e5-small
    .venv\\Scripts\\python.exe scripts\\build_index.py --lexical  # chỉ từ vựng, không nạp model

Encode kho là bước tốn thời gian nhất (~20 ms/đoạn với e5-small trên CPU, một lần cho
mỗi tài liệu); có thể dừng giữa chừng (Ctrl+C) và chạy lại, tài liệu đã xong được giữ nguyên.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from documents import ALLOWED_SUFFIXES, load_chunks  # noqa: E402
from indexer import LibraryIndex  # noqa: E402

LIBRARY_DIR = ROOT / "data" / "library"


def display_name(path: Path) -> str:
    name_path = path.with_suffix(".name")
    return name_path.read_text(encoding="utf-8") if name_path.exists() else path.name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexical", action="store_true", help="bỏ qua vector ngữ nghĩa")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--rebuild", action="store_true", help="xóa index cũ và làm lại từ đầu")
    args = ap.parse_args()

    index = LibraryIndex()
    if args.rebuild:
        index.clear()
    files = sorted(p for p in LIBRARY_DIR.glob("*") if p.suffix.lower() in ALLOWED_SUFFIXES)
    on_disk = {p.stem for p in files}
    for doc in index.documents:
        if doc.doc_id not in on_disk:
            index.remove_document(doc.doc_id)
            print(f"Gỡ khỏi index (file đã xóa): {doc.name}")

    todo = [p for p in files if not index.has(p.stem) or (not args.lexical and not index.has_vectors(p.stem))]
    print(f"{len(files)} tài liệu trong kho, cần index {len(todo)}.")
    if not todo:
        return

    encoder = None
    if not args.lexical:
        from embedder import FastEncoder

        encoder = FastEncoder()
        print(f"Đã nạp {encoder.name} trong {encoder.load_seconds:.1f}s")

    started = time.perf_counter()
    done_chunks = 0
    for n, path in enumerate(todo, 1):
        name = display_name(path)
        try:
            chunks = load_chunks(path.name, path.read_bytes())
        except (ValueError, OSError) as exc:
            print(f"[{n}/{len(todo)}] BỎ QUA {name}: {exc}")
            continue
        t = time.perf_counter()
        info = index.add_document(path.stem, name, chunks, encoder, batch_size=args.batch_size)
        done_chunks += info.n_chunks
        rate = done_chunks / max(time.perf_counter() - started, 1e-9)
        print(f"[{n}/{len(todo)}] {name}: {info.n_chunks} đoạn, {time.perf_counter() - t:.1f}s "
              f"(~{rate:.1f} đoạn/s)")
    print("Xong:", index.stats())


if __name__ == "__main__":
    main()
