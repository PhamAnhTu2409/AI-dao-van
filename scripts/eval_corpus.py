"""Đánh giá phát hiện đạo văn và thời gian truy vấn trên kho VietNews (cần chạy import_corpus.py trước).

    .venv\\Scripts\\python.exe scripts\\eval_corpus.py                 # chỉ từ vựng
    .venv\\Scripts\\python.exe scripts\\eval_corpus.py --semantic      # thêm e5-small (cần vector trong index)

Mỗi tài liệu truy vấn được ghép từ: văn bản "của người viết" (bài tập validation, không có trong kho)
và đoạn lấy từ bài trong kho theo từng kiểu đạo văn. Biết trước nguồn nên đo được:
- ``phát hiện``: nguồn nằm trong danh sách bị đánh dấu; ``top-1``: nguồn đứng đầu danh sách;
- ``báo nhầm``: có tài liệu KHÔNG phải nguồn bị đánh dấu mà không hề chung văn bản.
  Tài liệu không phải nguồn nhưng chung >= 5% 5-gram, hoặc chung nguyên văn một câu >= 10 từ (>= 3 cụm
  8 từ, không lấy mẫu, không bỏ văn mẫu), được tính riêng là "trùng thật": VietNews có nhiều tin cùng một
  sự kiện đăng lại giữa các báo, và cả bài trùng giữa tập train/validation, nên đánh dấu chúng là đúng.

    --no-span --no-windows   # tắt hai luật mới, đo lại phiên bản trước trên cùng bộ truy vấn
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from corpus import CORPUS_INDEX_DIR, doc_id, load_split  # noqa: E402
from documents import TextChunk, split_chunks  # noqa: E402
from indexer import (EXACT_FLAG, LEXICAL_FLAG, OVERLAP_FLAG, SEMANTIC_FLAG, SPAN_SHARED_FLAG,  # noqa: E402
                     WINDOW_MARGIN, LibraryIndex, shingle_hashes)

CATEGORIES = {
    "chep_nguyen_van": "Chép nguyên văn ~150 từ, chèn giữa văn bản khác",
    "sua_nhe": "Chép ~150 từ rồi sửa ~12% số từ (thay/xóa/đảo)",
    "sua_nang": "Chép ~150 từ, sửa ~30% số từ và xáo trộn câu",
    "hai_nguon": "Hai đoạn ~100 từ từ hai bài khác nhau",
    "tom_tat": "Sapo/tóm tắt do tòa soạn viết lại của bài nguồn (diễn đạt khác)",
    "am_tinh": "Bài không có trong kho (không được đánh dấu)",
}
SENT = re.compile(r"(?<=[.!?…])\s+")
REUSE = 0.05  # tài liệu không phải nguồn nhưng chung >= 5% 5-gram: văn bản trùng thật
VERBATIM = 3  # ... hoặc chung >= 3 cụm 8 từ liên tiếp (một câu >= 10 từ chép nguyên văn)


def verbatim(index: LibraryIndex, text: str, doc: str) -> int:
    """Số cụm 8 từ (đủ, không lấy mẫu, không bỏ văn mẫu) của ``text`` có trong tài liệu kho ``doc``."""
    doc_idx = index._docs[doc][0]
    body = " ".join(t for (t,) in index._db.execute("SELECT text FROM chunks WHERE doc_idx=? ORDER BY ord", (doc_idx,)))
    a, b = shingle_hashes(text, 8, sample=False), shingle_hashes(body, 8, sample=False)
    return int(np.isin(a, b).sum())


def words(text: str) -> list[str]:
    return text.split()


def edit(rng, tokens: list[str], rate: float, vocab: list[str]) -> list[str]:
    out = []
    for w in tokens:
        r = rng.random()
        if r < rate / 3:
            continue                                   # xóa
        if r < 2 * rate / 3:
            out.append(vocab[rng.integers(len(vocab))])  # thay
            continue
        out.append(w)
        if r < rate:
            out.append(vocab[rng.integers(len(vocab))])  # chèn
    return out


def shuffle_sentences(rng, text: str) -> str:
    parts = SENT.split(text)
    rng.shuffle(parts)
    return " ".join(parts)


def build_cases(rng, library: list[dict], outside: list[dict], n: int) -> list[dict]:
    vocab = [w for a in library[:2000] for w in words(a["article"]) if w.isalpha()]
    long_lib = [a for a in library if len(words(a["article"])) >= 220]
    own_pool = [a for a in outside if len(words(a["article"])) >= 260]

    def own(k: int) -> tuple[str, str]:
        a = own_pool[rng.integers(len(own_pool))]
        w = words(a["article"])
        return " ".join(w[:k]), " ".join(w[k : k + 80])

    def span(a: dict, k: int) -> list[str]:
        w = words(a["article"])
        start = int(rng.integers(0, max(1, len(w) - k)))
        return w[start : start + k]

    cases = []
    for cat in CATEGORIES:
        for _ in range(n):
            pre, post = own(120)
            if cat == "am_tinh":
                a = own_pool[rng.integers(len(own_pool))]
                cases.append({"cat": cat, "text": " ".join(words(a["article"])[:300]), "sources": []})
                continue
            src = long_lib[rng.integers(len(long_lib))]
            if cat == "chep_nguyen_van":
                body = " ".join(span(src, 150))
            elif cat == "sua_nhe":
                body = " ".join(edit(rng, span(src, 150), 0.12, vocab))
            elif cat == "sua_nang":
                body = shuffle_sentences(rng, " ".join(edit(rng, span(src, 150), 0.30, vocab)))
            elif cat == "hai_nguon":
                src2 = long_lib[rng.integers(len(long_lib))]
                body = " ".join(span(src, 100)) + " " + post + " " + " ".join(span(src2, 100))
                cases.append({"cat": cat, "text": f"{pre} {body}",
                              "sources": [doc_id("train", src["guid"]), doc_id("train", src2["guid"])]})
                continue
            else:  # tom_tat
                if not src["abstract"]:
                    continue
                body = src["abstract"]
            cases.append({"cat": cat, "text": f"{pre} {body} {post}", "sources": [doc_id("train", src["guid"])]})
    return cases


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", default=str(CORPUS_INDEX_DIR))
    ap.add_argument("--n", type=int, default=40, help="số truy vấn mỗi loại")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--semantic", action="store_true")
    ap.add_argument("--semantic-flag", type=float, default=SEMANTIC_FLAG)
    ap.add_argument("--lexical-flag", type=float, default=LEXICAL_FLAG)
    ap.add_argument("--overlap-flag", type=float, default=OVERLAP_FLAG)
    ap.add_argument("--exact-flag", type=float, default=EXACT_FLAG)
    ap.add_argument("--span-flag", type=int, default=SPAN_SHARED_FLAG)
    ap.add_argument("--window-margin", type=float, default=WINDOW_MARGIN)
    ap.add_argument("--no-span", action="store_true", help="tắt luật trùng 3-gram rải rác")
    ap.add_argument("--no-windows", action="store_true", help="tắt luật cửa sổ câu")
    ap.add_argument("--out", default=None, help="ghi kết quả chi tiết ra JSON")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    t = time.perf_counter()
    index = LibraryIndex(args.index_dir)
    print(f"Nạp index: {time.perf_counter() - t:.2f}s —", index.stats(), flush=True)
    present = {d.doc_id for d in index.documents}
    library = [a for a in load_split("train") if doc_id("train", a["guid"]) in present]
    outside = load_split("validation")
    cases = build_cases(rng, library, outside, args.n)
    print(f"{len(cases)} truy vấn từ {len(library):,} bài trong kho", flush=True)

    encoder = None
    if args.semantic:
        from embedder import FastEncoder

        encoder = FastEncoder()

    rows = []
    for case in cases:
        chunks = [(c.text, None) for c in split_chunks([TextChunk(case["text"])])]
        t = time.perf_counter()
        report = index.query(chunks, encoder, semantic_flag=args.semantic_flag, lexical_flag=args.lexical_flag,
                             overlap_flag=args.overlap_flag, exact_flag=args.exact_flag,
                             span_flag=0 if args.no_span else args.span_flag,
                             window_margin=None if args.no_windows else args.window_margin)
        ms = (time.perf_counter() - t) * 1000
        flagged = [r for r in report.documents if r.flagged]
        flagged_ids = [r.doc_id for r in flagged]
        src_rows = [r for r in report.documents if r.doc_id in case["sources"]]
        others = [r for r in report.documents if r.doc_id not in case["sources"]]
        real = {r.doc_id for r in flagged if r.doc_id not in case["sources"]
                and (r.exact_5gram >= REUSE or verbatim(index, case["text"], r.doc_id) >= VERBATIM)}
        unrelated = [r for r in others if r.exact_5gram < REUSE]
        best = lambda rs, key: max((getattr(r, key) or 0.0 for r in rs), default=0.0)  # noqa: E731
        rows.append({
            "cat": case["cat"], "ms": ms, "n_chunks": len(chunks), "timings": report.timings,
            "found": sum(s in flagged_ids for s in case["sources"]), "n_sources": len(case["sources"]),
            "top1": bool(flagged_ids) and flagged_ids[0] in case["sources"],
            "top1_real": bool(flagged_ids) and (flagged_ids[0] in case["sources"] or flagged_ids[0] in real),
            "false_flags": [r.doc_id for r in flagged if r.doc_id not in case["sources"] and r.doc_id not in real],
            "reuse_flags": sorted(real),
            "src_reasons": sorted({x for r in src_rows if r.flagged for m in r.matches for x in m.reasons}),
            "src_lexical": best(src_rows, "best_lexical"), "src_overlap": best(src_rows, "best_overlap"),
            "src_semantic": best(src_rows, "best_semantic"), "src_5gram": best(src_rows, "exact_5gram"),
            "other_lexical": best(unrelated, "best_lexical"), "other_overlap": best(unrelated, "best_overlap"),
            "other_semantic": best(unrelated, "best_semantic"), "other_5gram": best(unrelated, "exact_5gram"),
            "n_reported": len(report.documents),
        })

    by = defaultdict(list)
    for r in rows:
        by[r["cat"]].append(r)
    mode = "từ vựng + ngữ nghĩa" if encoder else "chỉ từ vựng"
    print(f"\n=== Kết quả ({mode}; ngưỡng TF-IDF {args.lexical_flag}, 3-gram {args.overlap_flag}, "
          f"5-gram {args.exact_flag}, ngữ nghĩa {args.semantic_flag}; 3-gram rải rác "
          f"{'tắt' if args.no_span else args.span_flag}; cửa sổ câu "
          f"{'tắt' if args.no_windows or not encoder else args.window_margin}) ===")
    print("Cột nguồn = trung vị điểm của bài nguồn; cột 'khác' = p95 điểm cao nhất của bài không liên quan. "
          "top-1*: đứng đầu là nguồn hoặc bài trùng thật.")
    head = ["loại", "n", "phát hiện", "top-1", "top-1*", "báo nhầm", "trùng thật", "5g nguồn", "3g nguồn", "TFIDF nguồn",
            "NN nguồn", "3g khác", "TFIDF khác", "NN khác"]
    widths = [16, 4, 10, 7, 8, 9, 11, 9, 9, 12, 9, 8, 11, 8]
    print("".join(h.rjust(w) if i else h.ljust(w) for i, (h, w) in enumerate(zip(head, widths))))
    for cat in CATEGORIES:
        rs = by.get(cat, [])
        if not rs:
            continue
        pos = cat != "am_tinh"
        med = lambda key: f"{np.median([r[key] for r in rs]):.2f}" if pos else "—"  # noqa: E731
        p95 = lambda key: f"{np.percentile([r[key] for r in rs], 95):.2f}"  # noqa: E731
        cells = [
            cat, len(rs),
            f"{sum(r['found'] for r in rs) / sum(r['n_sources'] for r in rs):.0%}" if pos else "—",
            f"{np.mean([r['top1'] for r in rs]):.0%}" if pos else "—",
            f"{np.mean([r['top1_real'] for r in rs]):.0%}" if pos else "—",
            f"{np.mean([bool(r['false_flags']) for r in rs]):.0%}",
            f"{np.mean([bool(r['reuse_flags']) for r in rs]):.0%}",
            med("src_5gram"), med("src_overlap"), med("src_lexical"),
            med("src_semantic") if encoder else "—",
            p95("other_overlap"), p95("other_lexical"), p95("other_semantic") if encoder else "—",
        ]
        print("".join(str(c).rjust(w) if i else str(c).ljust(w) for i, (c, w) in enumerate(zip(cells, widths))))
    print("\nLuật đã đánh dấu bài nguồn (số truy vấn):")
    for cat in CATEGORIES:
        counts = defaultdict(int)
        for r in by.get(cat, []):
            for reason in r["src_reasons"]:
                counts[reason] += 1
        if counts:
            print(f"  {cat}: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])))
    ms = np.array([r["ms"] for r in rows])
    print(f"\nThời gian truy vấn ({len(rows)} truy vấn, {np.mean([r['n_chunks'] for r in rows]):.1f} đoạn/truy vấn): "
          f"p50 {np.percentile(ms, 50):.0f} ms · p95 {np.percentile(ms, 95):.0f} ms · max {ms.max():.0f} ms")
    parts = defaultdict(list)
    for r in rows:
        for k, v in r["timings"].items():
            parts[k].append(v * 1000)
    print("  trung bình theo bước: " + " · ".join(f"{k} {np.mean(v):.0f} ms" for k, v in parts.items()))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    index.close()


if __name__ == "__main__":
    main()
