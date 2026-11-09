"""Index kho tài liệu để truy vấn đạo văn nhanh, thiết kế cho cỡ 100k tài liệu / vài trăm nghìn đoạn.

Lưu trữ (``data/index``):

- ``catalog.sqlite`` (WAL): tài liệu, văn bản từng đoạn, danh sách segment/shard vector.
  SQLite lo khóa ghi nên script nhập kho chạy song song được với app; app chỉ nạp phần
  thay đổi (theo bộ đếm ``generation``) thay vì dựng lại toàn bộ.
- ``segments/<id>/*.npy``: dấu vân tay từ vựng của một lô tài liệu, bất biến sau khi ghi.
  Nhiều segment nhỏ (thêm từng tệp từ app) được gộp tự động.
- ``vectors/<model>/<id>.npy``: vector đoạn float16 theo lô, tách khỏi segment để bổ sung
  vector sau (encode kho lớn mất hàng giờ, làm dần và tiếp tục được khi bị ngắt).
- ``df.npy``: số đoạn chứa mỗi đặc trưng băm (cho IDF), cập nhật tăng dần.

Truy vấn (mọi thứ nằm trong RAM trừ văn bản đoạn, lấy từ SQLite khi cần):

1. Trùng nguyên văn cấp tài liệu: hash 5-gram *lấy mẫu* (giữ 1/4 không gian hash, cùng quy
   tắc cho kho và truy vấn nên tỷ lệ trùng vẫn ước lượng đúng) -> ``searchsorted``.
2. Từ vựng cấp đoạn, lọc rồi mới chấm:
   - lọc: hash 3-gram lấy mẫu -> đoạn; đoạn có chung >= ``MIN_SHARED`` dấu vân tay mới là
     ứng viên (dấu xuất hiện ở quá nhiều đoạn - văn mẫu - bị bỏ);
   - chấm: TF-IDF cosine trên ứng viên. Văn bản ứng viên lấy từ SQLite và vector hóa lại lúc
     truy vấn, nên không phải giữ ma trận TF-IDF toàn kho (vài GB ở 300k đoạn) trong RAM.
   - trùng rải rác: số dấu 3-gram chung giữa 2 đoạn mới liên tiếp và cả một tài liệu kho (bắt
     đoạn chép bị sửa nhiều, xáo câu, vắt qua ranh giới đoạn).
3. Ngữ nghĩa (encoder bất kỳ có ``encode(list[str]) -> (N, D)`` đã chuẩn hóa L2): quét chính
   xác ma trận vector float16 theo khối. Model Matryoshka lớn (``coarse_dim``, vd. DeepX):
   tầng thô ``coarse_dim`` chiều đầu lấy top-K, tầng tinh chấm lại bằng đủ số chiều.
   Đoạn đã trùng từ vựng rất cao thì không encode (encode là bước tốn nhất).
   Ngoài đoạn 180 từ, đoạn mới còn được cắt thành cửa sổ câu ~60 từ, so với chính vector đoạn
   kho; cặp gần nghĩa hơn nền phải qua thêm bước xác minh từ vựng (không cần vector câu phía kho).

Module không phụ thuộc Streamlit: caller tự trích xuất văn bản, chia đoạn rồi truyền vào
danh sách ``(text, page)`` (xem ``documents.load_chunks``).
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sqlite3
import threading
import time
import unicodedata
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Protocol, Sequence

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer

ROOT = Path(__file__).resolve().parent
INDEX_DIR = ROOT / "data" / "index"
INDEX_VERSION = 3
DEFAULT_VECTOR_MODEL = "multilingual-e5-small"

DOC_SHINGLE = 5           # cấp tài liệu: tỷ lệ trùng nguyên văn
CHUNK_SHINGLE = 3         # cấp đoạn: lọc ứng viên, chịu được sửa lẻ tẻ tốt hơn 5-gram
SAMPLE_SHIFT = 62         # giữ hash có 2 bit cao = 0 -> 1/4 số cụm từ (cả kho lẫn truy vấn)
MIN_SHARED = 2            # số dấu 3-gram (đã lấy mẫu) chung tối thiểu để thành ứng viên
LEX_CANDIDATES = 30       # số ứng viên từ vựng tối đa mỗi đoạn truy vấn
DOC_FP_CAP = 25           # 5-gram có trong > số tài liệu này là văn mẫu ("theo thông tin từ..."), không tính
MIN_EXACT_SHARED = 3      # số 5-gram (đã lấy mẫu) chung tối thiểu để báo tỷ lệ trùng nguyên văn
MIN_OVERLAP_BASE = 24     # mẫu số tối thiểu khi tính tỷ lệ 3-gram chung (đoạn cuối ngắn dễ trùng ngẫu nhiên)
CHUNK_FP_CAP = 50         # 3-gram có trong > số đoạn này là cụm phổ biến, không dùng để lọc/tính tỷ lệ
HASH_FEATURES = 2**20
COARSE_CANDIDATES = 50    # số ứng viên mỗi đoạn truy vấn đưa sang tầng tinh (model Matryoshka)
SCAN_BLOCK = 65_536       # số đoạn kho quét mỗi khối (giới hạn RAM ma trận điểm)
SMALL_SEGMENT = 20_000    # segment/shard nhỏ hơn số đoạn này được gộp khi có quá nhiều
MAX_SMALL_SEGMENTS = 16

# Ngưỡng đánh dấu một đoạn là nghi trùng. Hiệu chỉnh trên kho báo VietNews (scripts/eval_corpus.py).
LEXICAL_FLAG = 0.50
OVERLAP_FLAG = 0.25        # tỷ lệ 3-gram đặc trưng của đoạn mới có trong đoạn kho
EXACT_FLAG = 0.10          # tỷ lệ 5-gram đặc trưng của cả tài liệu mới có trong một tài liệu kho
SEMANTIC_FLAG = 0.90       # ngưỡng sàn; kho dày (vd. 100k bài báo) tự nâng theo nền (xem SEMANTIC_MARGIN)
# Ngưỡng ngữ nghĩa thích ứng: một đoạn kho chỉ bị đánh dấu khi cosine vượt "nền" của chính đoạn mới
# (trung bình láng giềng thứ 6-10) ít nhất SEMANTIC_MARGIN. Kho 100k bài báo: tin khác sự kiện cùng thể loại
# đã đạt 0.92-0.94 với e5-small, nền ~0.91, nên ngưỡng thực tế ~0.95; kho nhỏ nhiều chủ đề: nền thấp,
# ngưỡng giữ 0.90.
SEMANTIC_MARGIN = 0.04
SEMANTIC_BACKGROUND = (5, 10)  # hạng láng giềng (từ 0) dùng làm nền
SEMANTIC_REPORT = 0.85     # nguồn có đoạn gần nghĩa từ mức này được liệt kê (không đánh dấu) để người rà soát xem
LEXICAL_SKIP_SEMANTIC = 0.80  # TF-IDF (hoặc tỷ lệ 3-gram chung) >= ngưỡng này: đã rõ trùng, bỏ qua encode
# Trùng 3-gram rải rác: đoạn chép bị sửa nhiều và xáo câu thường nằm vắt qua ranh giới đoạn (cả phía mới lẫn
# phía kho), nên tỷ lệ 3-gram theo từng cặp đoạn bị chia nhỏ. Đếm thêm số dấu 3-gram đặc trưng (đã lấy mẫu,
# bỏ văn mẫu) mà SPAN_CHUNKS đoạn mới liên tiếp (~325 từ) chung với cả một tài liệu kho.
SPAN_CHUNKS = 2
SPAN_SHARED_FLAG = 8       # ~32 cụm 3 từ đặc trưng thật; tin cùng thể loại không liên quan hiếm khi quá 5
# Cửa sổ câu: sapo/đoạn diễn đạt lại ngắn bị pha loãng trong đoạn 180 từ. Mỗi đoạn mới được cắt thêm thành
# các cửa sổ ~WINDOW_WORDS từ bắt đầu ở đầu câu, encode và so với vector đoạn kho sẵn có (không cần encode lại
# kho). e5-small không đủ phân biệt riêng ở cấp này (tin cùng thể loại sát nền), nên cặp chỉ bị đánh dấu khi
# vừa cao hơn nền của chính cửa sổ >= WINDOW_MARGIN, vừa có >= WINDOW_CONTAIN_FLAG trọng số IDF của từ trong
# cửa sổ xuất hiện trong đoạn kho.
WINDOW_WORDS = 60
WINDOW_STRIDE = 30         # cửa sổ kế tiếp bắt đầu ở đầu câu cách cửa sổ trước >= số từ này (~6 cửa sổ / đoạn)
WINDOW_PIECE = 50          # câu dài hơn số từ này bị cắt nhỏ
WINDOW_MIN_WORDS = 12
WINDOW_TOP = 3             # số đoạn kho gần nhất mỗi cửa sổ được xác minh từ vựng
WINDOW_MARGIN = 0.03
WINDOW_FLOOR = SEMANTIC_REPORT
WINDOW_CONTAIN_FLAG = 0.50
_SENTENCE = re.compile(r"(?<=[.!?…])\s+")

Chunk = tuple[str, "int | None"]
_PAIR = np.int64(1 << 40)  # khóa cặp (đoạn truy vấn, đoạn kho) = q * _PAIR + chunk_id


class Encoder(Protocol):
    name: str

    def encode(self, texts: list[str], batch_size: int = ...) -> np.ndarray: ...


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


_WORD = re.compile(r"\w+", flags=re.UNICODE)
_MIX = np.uint64(0x9E3779B97F4A7C15)
_word_cache: dict[str, int] = {}


def _word_ids(text: str) -> np.ndarray:
    """Hash 64-bit ổn định (blake2b, không phụ thuộc PYTHONHASHSEED) cho từng từ."""
    words = _WORD.findall(normalize(text))
    for w in words:
        if w not in _word_cache:
            if len(_word_cache) > 2_000_000:
                _word_cache.clear()
            _word_cache[w] = int.from_bytes(hashlib.blake2b(w.encode("utf-8"), digest_size=8).digest(), "little")
    return np.fromiter((_word_cache[w] for w in words), dtype=np.uint64, count=len(words))


def shingle_hashes(text: str, size: int = DOC_SHINGLE, sample: bool = True) -> np.ndarray:
    """Tập hash (unique, sắp xếp) của các cụm ``size`` từ liên tiếp; mặc định chỉ giữ mẫu 1/4."""
    ids = _word_ids(text)
    if not len(ids):
        return np.zeros(0, dtype=np.uint64)
    size = min(size, len(ids))
    with np.errstate(over="ignore"):
        h = np.zeros(len(ids) - size + 1, dtype=np.uint64)
        for offset in range(size):
            h = h * _MIX + ids[offset : offset + len(h)]
    if sample:
        h = h[(h >> np.uint64(SAMPLE_SHIFT)) == 0]
    return np.unique(h)


def chunk_shingles(texts: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Hash 3-gram (đã lấy mẫu) của từng đoạn, dạng CSR: (hashes, indptr)."""
    parts = [shingle_hashes(t, CHUNK_SHINGLE) for t in texts]
    indptr = np.zeros(len(parts) + 1, dtype=np.int64)
    indptr[1:] = np.cumsum([len(p) for p in parts])
    return (np.concatenate(parts) if parts else np.zeros(0, np.uint64)), indptr


_hasher = HashingVectorizer(
    n_features=HASH_FEATURES, ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b",
    preprocessor=normalize, alternate_sign=False, norm=None,
)


def term_counts(texts: Sequence[str]) -> sp.csr_matrix:
    return _hasher.transform(texts).astype(np.float32).tocsr()


def _tfidf(counts: sp.csr_matrix, idf: np.ndarray) -> sp.csr_matrix:
    x = counts.copy()
    np.log1p(x.data, out=x.data)  # sublinear tf
    x.data *= idf[x.indices]
    rows = np.repeat(np.arange(x.shape[0]), np.diff(x.indptr))
    norms = np.sqrt(np.bincount(rows, weights=x.data.astype(np.float64) ** 2, minlength=x.shape[0]))
    norms[norms == 0] = 1.0
    x.data /= norms[rows].astype(np.float32)
    return x


def sentence_windows(text: str, size: int = WINDOW_WORDS, stride: int = WINDOW_STRIDE) -> list[str]:
    """Cửa sổ ~``size`` từ, mỗi cửa sổ bắt đầu ở đầu câu, cách nhau >= ``stride`` từ (câu quá dài bị cắt thành
    mẩu ``WINDOW_PIECE`` từ). Dừng khi cửa sổ chạm cuối văn bản."""
    pieces = []
    for sentence in _SENTENCE.split(text):
        words = sentence.split()
        pieces.extend(words[k : k + WINDOW_PIECE] for k in range(0, len(words), WINDOW_PIECE))
    starts = np.cumsum([0] + [len(p) for p in pieces])
    windows, i = [], 0
    while i < len(pieces):
        words, j = [], i
        while j < len(pieces) and len(words) < size:
            words += pieces[j]
            j += 1
        if len(words) >= WINDOW_MIN_WORDS or not windows:
            windows.append(" ".join(words))
        if j == len(pieces):
            break
        # Cửa sổ kế tiếp bắt đầu ở đầu câu đầu tiên cách đầu cửa sổ này >= ``stride`` từ.
        i = max(i + 1, int(np.searchsorted(starts, starts[i] + stride, "left")))
    return windows


def _containment(short: Sequence[str], long: Sequence[str], idf: np.ndarray) -> np.ndarray:
    """Tỷ lệ trọng số IDF của các từ/cặp từ trong ``short[i]`` có mặt trong ``long[i]``."""
    a, b = term_counts(short), term_counts(long)
    a.data[:] = 1.0
    b.data[:] = 1.0
    weighted = a.multiply(idf[None, :]).tocsr()
    total = np.asarray(weighted.sum(axis=1)).ravel()
    shared = np.asarray(weighted.multiply(b).sum(axis=1)).ravel()
    return (shared / np.maximum(total, 1e-9)).astype(np.float32)


def _doc_freq(counts: sp.csr_matrix) -> np.ndarray:
    return np.bincount(counts.indices, minlength=HASH_FEATURES).astype(np.int32)


def _head(full: np.ndarray, dim: int | None) -> np.ndarray:
    """``dim`` chiều đầu (Matryoshka), chuẩn hóa lại L2; ``None`` = đủ chiều."""
    head = np.ascontiguousarray(full if dim is None else full[:, :dim], dtype=np.float32)
    norms = np.linalg.norm(head, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return head / norms


def _expand_ranges(left: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Ghép các khoảng [left, left+len) thành một mảng chỉ số, không lặp Python."""
    offsets = np.repeat(left - np.cumsum(lengths) + lengths, lengths)
    return offsets + np.arange(lengths.sum())


def _postings(sorted_hash: np.ndarray, query: np.ndarray, cap: int | None = None):
    """Vị trí trong ``sorted_hash`` khớp từng phần tử ``query`` -> (vị trí, chỉ số query)."""
    left = np.searchsorted(sorted_hash, query, "left")
    lengths = np.searchsorted(sorted_hash, query, "right") - left
    keep = lengths > 0 if cap is None else (lengths > 0) & (lengths <= cap)
    left, lengths, owner = left[keep], lengths[keep], np.flatnonzero(keep)
    return _expand_ranges(left, lengths), np.repeat(owner, lengths)


def _topk(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Top-k theo hàng, sắp giảm dần, không sort toàn bộ hàng."""
    k = min(k, scores.shape[1])
    if k == 0:
        return np.zeros((scores.shape[0], 0), scores.dtype), np.zeros((scores.shape[0], 0), np.int64)
    if k < scores.shape[1]:
        idx = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    else:
        idx = np.tile(np.arange(k), (len(scores), 1))
    part = np.take_along_axis(scores, idx, axis=1)
    order = np.argsort(-part, axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1), np.take_along_axis(idx, order, axis=1).astype(np.int64)


_torch = None


def _scores_f16(query: np.ndarray, block: np.ndarray) -> np.ndarray:
    """``query`` (float32) · ``block`` (float16)ᵀ.

    numpy không có phép nhân float16 nhanh: đổi khối sang float32 mất ~0.7 s cho 365k vector.
    PyTorch nhân thẳng float16 trên CPU (~20 ms, sai số < 1e-3) mà không tạo bản sao float32;
    không nạp được torch thì quay về numpy.
    """
    global _torch
    if _torch is None:
        try:
            import torch

            _torch = torch
        except Exception:  # thiếu torch hoặc DLL không nạp được
            _torch = False
    if _torch:
        t = _torch.from_numpy(np.ascontiguousarray(query, dtype=np.float32)).to(_torch.float16)
        return (t @ _torch.from_numpy(block).T).float().numpy()
    return query @ block.T.astype(np.float32)


def _sorted_pairs(hashes: np.ndarray, owners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(hashes, kind="stable")
    return hashes[order], owners[order]


@dataclass
class DocInfo:
    doc_id: str
    name: str
    n_chunks: int
    n_shingles: int
    has_vectors: bool


@dataclass
class Match:
    query_chunk: int
    doc_id: str
    doc_chunk: int
    lexical: float
    semantic: float | None
    overlap: float = 0.0        # tỷ lệ 3-gram đặc trưng của đoạn mới có trong đoạn kho
    reasons: tuple[str, ...] = ()   # luật đánh dấu: "3-gram", "tfidf", "ngữ nghĩa", "3-gram rải rác", "câu"
    window: str | None = None   # cửa sổ câu của đoạn mới khớp (luật "câu")


@dataclass
class DocResult:
    doc_id: str
    name: str
    exact_5gram: float          # tỷ lệ 5-gram của tài liệu mới có trong tài liệu này
    best_lexical: float
    best_semantic: float | None
    coverage: float             # tỷ lệ đoạn của tài liệu mới bị đánh dấu với nguồn này
    best_overlap: float = 0.0
    flagged: bool = False       # nghi trùng: có đoạn vượt ngưỡng hoặc trùng 5-gram >= EXACT_FLAG
    span_shared: int = 0        # số dấu 3-gram (mẫu 1/4) đặc trưng chung nhiều nhất trong SPAN_CHUNKS đoạn mới
    matches: list[Match] = field(default_factory=list)


@dataclass
class QueryReport:
    documents: list[DocResult]
    flagged_chunks: int
    total_chunks: int
    semantic_used: bool
    encoded_chunks: int
    timings: dict[str, float]
    encoded_windows: int = 0


@dataclass
class _Segment:
    seg_id: int
    chunk_ids: np.ndarray   # int64: hàng -> id đoạn toàn cục
    chunk_doc: np.ndarray   # int32: hàng -> doc_idx
    chunk_ord: np.ndarray   # int32: hàng -> thứ tự đoạn trong tài liệu
    fp_hash: np.ndarray     # uint64 sắp xếp: dấu 3-gram cấp đoạn
    fp_row: np.ndarray      # int32: hàng sở hữu dấu
    doc_hash: np.ndarray    # uint64 sắp xếp: dấu 5-gram cấp tài liệu
    doc_owner: np.ndarray   # int32: doc_idx sở hữu dấu

    @property
    def nbytes(self) -> int:
        return sum(a.nbytes for a in (self.chunk_ids, self.chunk_doc, self.chunk_ord, self.fp_hash,
                                      self.fp_row, self.doc_hash, self.doc_owner))


@dataclass
class _Shard:
    shard_id: int
    chunk_ids: np.ndarray   # int64
    vectors: np.ndarray     # float16 (n, D), đã chuẩn hóa L2
    coarse: np.ndarray      # float16 (n, coarse_dim) chuẩn hóa lại, hoặc chính ``vectors``
    alive: np.ndarray       # bool (n,)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS docs(
    doc_idx INTEGER PRIMARY KEY, doc_id TEXT UNIQUE NOT NULL, name TEXT NOT NULL,
    n_chunks INTEGER NOT NULL, n_shingles INTEGER NOT NULL, first_chunk INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS chunks(
    chunk_id INTEGER PRIMARY KEY, doc_idx INTEGER NOT NULL, ord INTEGER NOT NULL, page INTEGER, text TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_idx, ord);
CREATE TABLE IF NOT EXISTS segments(seg_id INTEGER PRIMARY KEY, n_chunks INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS vector_shards(shard_id INTEGER PRIMARY KEY, model TEXT NOT NULL, n_chunks INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS doc_vectors(doc_idx INTEGER NOT NULL, model TEXT NOT NULL, PRIMARY KEY(doc_idx, model));
CREATE TABLE IF NOT EXISTS removed(lo INTEGER NOT NULL, hi INTEGER NOT NULL);
"""


class LibraryIndex:
    """Index toàn kho trên đĩa (SQLite + segment .npy), nạp tăng dần vào RAM; một khóa cho ghi/đọc."""

    def __init__(self, index_dir: Path | str = INDEX_DIR, vector_model: str = DEFAULT_VECTOR_MODEL,
                 coarse_dim: int | None = None):
        self.vector_model = vector_model
        self.coarse_dim = coarse_dim
        self.dir = Path(index_dir)
        self.seg_dir = self.dir / "segments"
        self.vec_dir = self.dir / "vectors" / re.sub(r"[^\w.-]", "_", vector_model)
        for d in (self.seg_dir, self.vec_dir):
            d.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = self._connect()
        self._cleanup()
        self._reset_memory()
        self.reload()

    def close(self) -> None:
        """Đóng kết nối SQLite (cần trên Windows trước khi xóa thư mục index)."""
        with self._lock:
            self._db.close()

    def __enter__(self) -> "LibraryIndex":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---------- SQLite ----------
    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.dir / "catalog.sqlite", timeout=60, check_same_thread=False,
                             isolation_level=None)
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=NORMAL")
        db.executescript(_SCHEMA)
        version = db.execute("SELECT value FROM meta WHERE key='version'").fetchone()
        if version is None:
            db.execute("INSERT INTO meta VALUES('version', ?), ('generation', '0'), ('next_chunk', '0'), ('next_doc', '0')",
                       (str(INDEX_VERSION),))
        elif int(version[0]) != INDEX_VERSION:
            raise RuntimeError(f"Index ở {self.dir} là phiên bản {version[0]}, cần {INDEX_VERSION}: hãy xóa thư mục và index lại.")
        return db

    def _generation(self) -> int:
        return int(self._meta("generation"))

    def _meta(self, key: str) -> str:
        return self._db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()[0]

    def _cleanup(self) -> None:
        """Xóa thư mục/tệp mồ côi do lần ghi bị ngắt (chỉ những cái cũ hơn 1 giờ, tránh đụng lần ghi đang chạy)."""
        known_segs = {f"{r[0]:08d}" for r in self._db.execute("SELECT seg_id FROM segments")}
        known_shards = {f"{r[0]:08d}.npz" for r in self._db.execute("SELECT shard_id FROM vector_shards")}
        cutoff = time.time() - 3600
        for path in self.seg_dir.iterdir():
            if path.name not in known_segs and path.stat().st_mtime < cutoff:
                shutil.rmtree(path, ignore_errors=True)
        for path in (self.dir / "vectors").glob("*/*"):
            if path.name not in known_shards and path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)

    def _write(self):
        """Giao dịch ghi độc quyền (khóa giữa các tiến trình), tăng ``generation`` khi xong."""
        index = self

        class _Tx:
            def __enter__(self_):
                index._db.execute("BEGIN IMMEDIATE")
                return index._db

            def __exit__(self_, exc_type, exc, tb):
                if exc_type is None:
                    index._db.execute("UPDATE meta SET value = CAST(value AS INTEGER) + 1 WHERE key='generation'")
                    index._db.execute("COMMIT")
                else:
                    index._db.execute("ROLLBACK")
                return False

        return _Tx()

    def _load_df(self) -> np.ndarray:
        path = self.dir / "df.npy"
        return np.load(path) if path.exists() else np.zeros(HASH_FEATURES, dtype=np.int32)

    def _save_df(self, df: np.ndarray) -> None:
        tmp = self.dir / f"df.{uuid.uuid4().hex}.npy"
        np.save(tmp, df)
        tmp.replace(self.dir / "df.npy")

    # ---------- trạng thái trong RAM ----------
    def _reset_memory(self) -> None:
        self._segments: dict[int, _Segment] = {}
        self._shards: dict[int, _Shard] = {}
        self._gen = -1
        self._checked = 0.0
        self._docs: dict[str, tuple[int, str, int, int]] = {}   # doc_id -> (doc_idx, name, n_chunks, n_shingles)
        self._doc_ids: dict[int, str] = {}                       # doc_idx -> doc_id
        self._with_vectors: set[int] = set()
        self._alive = np.zeros(0, dtype=bool)
        self._chunk_doc = np.zeros(0, dtype=np.int32)
        self._chunk_ord = np.zeros(0, dtype=np.int32)
        self._idf = np.ones(HASH_FEATURES, dtype=np.float32)
        self._n_alive = 0
        self._doc_list: list[DocInfo] | None = None
        self.load_seconds = 0.0

    def reload(self) -> None:
        """Nạp lại toàn bộ (vd. sau khi thư mục index bị sửa tay)."""
        with self._lock:
            self._reset_memory()
            self._refresh(force=True)

    def _refresh(self, force: bool = False) -> None:
        """Đồng bộ RAM với đĩa nếu ``generation`` đổi; chỉ nạp segment/shard mới."""
        now = time.monotonic()
        if not force and now - self._checked < 1.0:
            return
        self._checked = now
        gen = self._generation()
        if gen == self._gen:
            return
        started = time.perf_counter()
        db = self._db
        seg_ids = {r[0] for r in db.execute("SELECT seg_id FROM segments")}
        shard_ids = {r[0] for r in db.execute("SELECT shard_id FROM vector_shards WHERE model=?", (self.vector_model,))}
        try:
            for sid in list(self._segments):
                if sid not in seg_ids:
                    del self._segments[sid]
            for sid in sorted(seg_ids - set(self._segments)):
                self._segments[sid] = self._load_segment(sid)
            for sid in list(self._shards):
                if sid not in shard_ids:
                    del self._shards[sid]
            for sid in sorted(shard_ids - set(self._shards)):
                self._shards[sid] = self._load_shard(sid)
        except FileNotFoundError:
            # Tiến trình khác vừa gộp segment: nạp lại từ đầu theo danh sách mới.
            if force:
                raise
            self._reset_memory()
            return self._refresh(force=True)

        self._docs, self._doc_ids = {}, {}
        for doc_idx, doc_id, name, n_chunks, n_shingles in db.execute(
                "SELECT doc_idx, doc_id, name, n_chunks, n_shingles FROM docs"):
            self._docs[doc_id] = (doc_idx, name, n_chunks, n_shingles)
            self._doc_ids[doc_idx] = doc_id
        self._with_vectors = {r[0] for r in db.execute("SELECT doc_idx FROM doc_vectors WHERE model=?", (self.vector_model,))}

        size = 1 + max((int(s.chunk_ids[-1]) for s in self._segments.values() if len(s.chunk_ids)), default=-1)
        self._alive = np.zeros(size, dtype=bool)
        self._chunk_doc = np.full(size, -1, dtype=np.int32)
        self._chunk_ord = np.zeros(size, dtype=np.int32)
        for s in self._segments.values():
            self._alive[s.chunk_ids] = True
            self._chunk_doc[s.chunk_ids] = s.chunk_doc
            self._chunk_ord[s.chunk_ids] = s.chunk_ord
        for lo, hi in db.execute("SELECT lo, hi FROM removed"):
            self._alive[lo : min(hi, size)] = False
        for shard in self._shards.values():
            ids = shard.chunk_ids
            shard.alive = (ids < size) & self._alive[np.minimum(ids, max(size - 1, 0))] if size else np.zeros(len(ids), bool)

        df = self._load_df()
        n_alive = int(self._alive.sum())
        self._idf = (np.log((1 + n_alive) / (1 + df.astype(np.float32))) + 1).astype(np.float32)
        self._n_alive = n_alive
        self._doc_list = None
        self._gen = gen
        if self._shards and _torch is None:
            _scores_f16(np.zeros((1, 1), np.float32), np.zeros((1, 1), np.float16))  # nạp torch trước truy vấn đầu
        self.load_seconds = time.perf_counter() - started

    def _load_segment(self, seg_id: int) -> _Segment:
        d = self.seg_dir / f"{seg_id:08d}"
        load = lambda name: np.load(d / f"{name}.npy")  # noqa: E731
        return _Segment(seg_id, load("chunk_ids"), load("chunk_doc"), load("chunk_ord"), load("fp_hash"),
                        load("fp_row"), load("doc_hash"), load("doc_owner"))

    def _load_shard(self, shard_id: int) -> _Shard:
        z = np.load(self.vec_dir / f"{shard_id:08d}.npz")
        vectors = z["vectors"]
        coarse = vectors if not self.coarse_dim else _head(vectors, self.coarse_dim).astype(np.float16)
        return _Shard(shard_id, z["chunk_ids"], vectors, coarse, np.ones(len(vectors), bool))

    def _ensure_fresh(self) -> None:
        self._refresh()

    # ---------- ghi ----------
    def _write_segment(self, seg_id: int, chunk_ids, chunk_doc, chunk_ord, fp_hash, fp_row, doc_hash, doc_owner) -> None:
        final = self.seg_dir / f"{seg_id:08d}"
        tmp = self.seg_dir / f"tmp-{uuid.uuid4().hex}"
        tmp.mkdir()
        fp_hash, fp_row = _sorted_pairs(fp_hash, fp_row.astype(np.int32))
        doc_hash, doc_owner = _sorted_pairs(doc_hash, doc_owner.astype(np.int32))
        arrays = {"chunk_ids": chunk_ids.astype(np.int64), "chunk_doc": chunk_doc.astype(np.int32),
                  "chunk_ord": chunk_ord.astype(np.int32), "fp_hash": fp_hash, "fp_row": fp_row,
                  "doc_hash": doc_hash, "doc_owner": doc_owner}
        for name, arr in arrays.items():
            np.save(tmp / f"{name}.npy", arr)
        if final.exists():
            shutil.rmtree(final)
        tmp.rename(final)

    def _write_shard(self, shard_id: int, chunk_ids: np.ndarray, vectors: np.ndarray) -> None:
        tmp = self.vec_dir / f"tmp-{uuid.uuid4().hex}.npz"
        np.savez(tmp, chunk_ids=chunk_ids.astype(np.int64), vectors=vectors.astype(np.float16))
        tmp.replace(self.vec_dir / f"{shard_id:08d}.npz")

    def has(self, doc_id: str) -> bool:
        with self._lock:
            self._ensure_fresh()
            return doc_id in self._docs

    def has_vectors(self, doc_id: str) -> bool:
        with self._lock:
            self._ensure_fresh()
            info = self._docs.get(doc_id)
            return bool(info and info[0] in self._with_vectors)

    def add_document(self, doc_id: str, name: str, chunks: Sequence[Chunk], encoder: Encoder | None = None,
                     batch_size: int = 16) -> DocInfo:
        """Index một tài liệu. Đã có thì chỉ bổ sung vector (nếu có encoder và còn thiếu)."""
        self.add_documents([(doc_id, name, chunks)], encoder, batch_size)
        with self._lock:
            self._refresh(force=True)
            doc_idx, name, n_chunks, n_shingles = self._docs[doc_id]
            return DocInfo(doc_id, name, n_chunks, n_shingles, doc_idx in self._with_vectors)

    def add_documents(self, items: Iterable[tuple[str, str, Sequence[Chunk]]], encoder: Encoder | None = None,
                      batch_size: int = 16, progress: Callable[[int, int], None] | None = None) -> int:
        """Index một lô tài liệu thành MỘT segment (nhanh hơn nhiều so với từng tài liệu).

        Trả về số tài liệu mới được thêm. Tài liệu đã có chỉ được bổ sung vector nếu cần.
        """
        if encoder is not None and encoder.name != self.vector_model:
            raise ValueError(f"Encoder {encoder.name} khác model của index ({self.vector_model}).")
        with self._lock:
            self._refresh(force=True)
            fresh, need_vectors, seen = [], [], set()
            for doc_id, name, chunks in items:
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                if doc_id in self._docs:
                    if encoder is not None and self._docs[doc_id][0] not in self._with_vectors:
                        need_vectors.append(doc_id)
                elif chunks:
                    fresh.append((doc_id, name, [(t, p) for t, p in chunks]))
        if need_vectors:
            self.add_vectors(encoder, doc_ids=need_vectors, batch_size=batch_size)
        if not fresh:
            return 0

        texts = [t for _, _, chunks in fresh for t, _ in chunks]
        counts = term_counts(texts)
        ch_hash, ch_ptr = chunk_shingles(texts)
        doc_sh = [shingle_hashes("\n".join(t for t, _ in chunks), DOC_SHINGLE) for _, _, chunks in fresh]
        vectors = None
        if encoder is not None:
            vectors = np.asarray(encoder.encode(texts, batch_size=batch_size, **({"progress": progress} if progress else {})),
                                 dtype=np.float16)

        with self._lock, self._write() as db:
            # Id tăng đơn điệu, không tái sử dụng id của tài liệu đã xóa (segment cũ vẫn còn tham chiếu tới).
            next_chunk, next_doc = int(self._meta("next_chunk")), int(self._meta("next_doc"))
            db.execute("UPDATE meta SET value=? WHERE key='next_chunk'", (str(next_chunk + len(texts)),))
            db.execute("UPDATE meta SET value=? WHERE key='next_doc'", (str(next_doc + len(fresh)),))
            seg_id = db.execute("INSERT INTO segments(n_chunks) VALUES (?)", (len(texts),)).lastrowid
            chunk_ids = np.arange(next_chunk, next_chunk + len(texts), dtype=np.int64)
            chunk_doc = np.empty(len(texts), dtype=np.int32)
            chunk_ord = np.empty(len(texts), dtype=np.int32)
            doc_rows, chunk_rows, doc_owner = [], [], []
            row = 0
            for k, (doc_id, name, chunks) in enumerate(fresh):
                doc_idx = next_doc + k
                n = len(chunks)
                chunk_doc[row : row + n] = doc_idx
                chunk_ord[row : row + n] = np.arange(n)
                doc_rows.append((doc_idx, doc_id, name, n, len(doc_sh[k]), int(chunk_ids[row])))
                chunk_rows.extend((int(chunk_ids[row + i]), doc_idx, i, p, t) for i, (t, p) in enumerate(chunks))
                doc_owner.append(np.full(len(doc_sh[k]), doc_idx, dtype=np.int32))
                row += n
            fp_row = np.repeat(np.arange(len(texts), dtype=np.int32), np.diff(ch_ptr))
            self._write_segment(seg_id, chunk_ids, chunk_doc, chunk_ord, ch_hash, fp_row,
                                np.concatenate(doc_sh), np.concatenate(doc_owner))
            db.executemany("INSERT INTO docs VALUES (?,?,?,?,?,?)", doc_rows)
            db.executemany("INSERT INTO chunks VALUES (?,?,?,?,?)", chunk_rows)
            df = self._load_df()
            df += _doc_freq(counts)
            self._save_df(df)
            if vectors is not None:
                shard_id = db.execute("INSERT INTO vector_shards(model, n_chunks) VALUES (?, ?)",
                                      (self.vector_model, len(texts))).lastrowid
                self._write_shard(shard_id, chunk_ids, vectors)
                db.executemany("INSERT OR IGNORE INTO doc_vectors VALUES (?, ?)",
                               [(r[0], self.vector_model) for r in doc_rows])
        self._maybe_compact()
        return len(fresh)

    def add_vectors(self, encoder: Encoder, doc_ids: Sequence[str] | None = None, batch_size: int = 16,
                    docs_per_shard: int = 2000, progress: Callable[[int, int], None] | None = None) -> int:
        """Encode các tài liệu chưa có vector của model hiện tại, ghi từng shard (tiếp tục được nếu bị ngắt)."""
        if encoder.name != self.vector_model:
            raise ValueError(f"Encoder {encoder.name} khác model của index ({self.vector_model}).")
        with self._lock:
            self._refresh(force=True)
            wanted = self._docs.keys() if doc_ids is None else [d for d in doc_ids if d in self._docs]
            todo = sorted(self._docs[d][0] for d in wanted if self._docs[d][0] not in self._with_vectors)
        total_chunks = sum(self._docs[self._doc_ids[i]][2] for i in todo)
        done_chunks = 0
        for start in range(0, len(todo), docs_per_shard):
            batch = todo[start : start + docs_per_shard]
            marks = ",".join("?" * len(batch))
            rows = self._db.execute(
                f"SELECT chunk_id, text FROM chunks WHERE doc_idx IN ({marks}) ORDER BY chunk_id", batch).fetchall()
            if not rows:
                continue
            offset = done_chunks
            report = (lambda d, t: progress(offset + d, total_chunks)) if progress else None
            vectors = encoder.encode([t for _, t in rows], batch_size=batch_size, **({"progress": report} if report else {}))
            done_chunks += len(rows)
            with self._lock, self._write() as db:
                shard_id = db.execute("INSERT INTO vector_shards(model, n_chunks) VALUES (?, ?)",
                                      (self.vector_model, len(rows))).lastrowid
                self._write_shard(shard_id, np.array([c for c, _ in rows], dtype=np.int64), np.asarray(vectors))
                db.executemany("INSERT OR IGNORE INTO doc_vectors VALUES (?, ?)", [(i, self.vector_model) for i in batch])
        self._maybe_compact()
        return len(todo)

    def remove_document(self, doc_id: str) -> None:
        with self._lock:
            self._refresh(force=True)
            if doc_id not in self._docs:
                return
            with self._write() as db:
                row = db.execute("SELECT doc_idx, first_chunk, n_chunks FROM docs WHERE doc_id=?", (doc_id,)).fetchone()
                if row is None:
                    return
                doc_idx, first, n = row
                texts = [t for (t,) in db.execute("SELECT text FROM chunks WHERE doc_idx=? ORDER BY ord", (doc_idx,))]
                if texts:
                    df = self._load_df()
                    df -= _doc_freq(term_counts(texts))
                    np.maximum(df, 0, out=df)
                    self._save_df(df)
                db.execute("DELETE FROM chunks WHERE doc_idx=?", (doc_idx,))
                db.execute("DELETE FROM doc_vectors WHERE doc_idx=?", (doc_idx,))
                db.execute("DELETE FROM docs WHERE doc_idx=?", (doc_idx,))
                db.execute("INSERT INTO removed VALUES (?, ?)", (first, first + n))
            self._refresh(force=True)

    def clear(self) -> None:
        with self._lock:
            with self._write() as db:
                for table in ("docs", "chunks", "segments", "vector_shards", "doc_vectors", "removed"):
                    db.execute(f"DELETE FROM {table}")
                (self.dir / "df.npy").unlink(missing_ok=True)
            shutil.rmtree(self.seg_dir, ignore_errors=True)
            shutil.rmtree(self.dir / "vectors", ignore_errors=True)
            for d in (self.seg_dir, self.vec_dir):
                d.mkdir(parents=True, exist_ok=True)
            self.reload()

    # ---------- gộp segment ----------
    def _maybe_compact(self) -> None:
        with self._lock:
            self._refresh(force=True)
            small = [s for s in self._segments.values() if len(s.chunk_ids) < SMALL_SEGMENT]
            small_shards = [s for s in self._shards.values() if len(s.chunk_ids) < SMALL_SEGMENT]
            if len(small) > MAX_SMALL_SEGMENTS or len(small_shards) > MAX_SMALL_SEGMENTS:
                self.compact(only_small=True)

    def compact(self, only_small: bool = False) -> None:
        """Gộp segment/shard (mặc định tất cả; ``only_small``: chỉ các mảnh nhỏ) và bỏ hẳn đoạn đã xóa."""
        with self._lock:
            self._refresh(force=True)
            pick = lambda parts: [p for p in parts if not only_small or len(p.chunk_ids) < SMALL_SEGMENT]  # noqa: E731
            segs = sorted(pick(self._segments.values()), key=lambda s: s.seg_id)
            shards = sorted(pick(self._shards.values()), key=lambda s: s.shard_id)
            alive = self._alive
            with self._write() as db:
                if len(segs) > 1 or (segs and not all(alive[s.chunk_ids].all() for s in segs)):
                    parts = {k: [] for k in ("ids", "doc", "ord", "fh", "fr", "dh", "do")}
                    rows = 0
                    alive_docs = set(self._doc_ids)
                    for s in segs:
                        keep = alive[s.chunk_ids]
                        new_row = np.cumsum(keep) - 1 + rows
                        parts["ids"].append(s.chunk_ids[keep])
                        parts["doc"].append(s.chunk_doc[keep])
                        parts["ord"].append(s.chunk_ord[keep])
                        fp_keep = keep[s.fp_row]
                        parts["fh"].append(s.fp_hash[fp_keep])
                        parts["fr"].append(new_row[s.fp_row[fp_keep]].astype(np.int32))
                        doc_keep = np.isin(s.doc_owner, np.fromiter(alive_docs, np.int32, len(alive_docs)))
                        parts["dh"].append(s.doc_hash[doc_keep])
                        parts["do"].append(s.doc_owner[doc_keep])
                        rows += int(keep.sum())
                    cat = {k: np.concatenate(v) for k, v in parts.items()}
                    seg_id = db.execute("INSERT INTO segments(n_chunks) VALUES (?)", (rows,)).lastrowid
                    self._write_segment(seg_id, cat["ids"], cat["doc"], cat["ord"], cat["fh"], cat["fr"], cat["dh"], cat["do"])
                    db.executemany("DELETE FROM segments WHERE seg_id=?", [(s.seg_id,) for s in segs])
                if len(shards) > 1 or (shards and not all(s.alive.all() for s in shards)):
                    ids = np.concatenate([s.chunk_ids[s.alive] for s in shards])
                    vecs = np.concatenate([s.vectors[s.alive] for s in shards])
                    shard_id = db.execute("INSERT INTO vector_shards(model, n_chunks) VALUES (?, ?)",
                                          (self.vector_model, len(ids))).lastrowid
                    self._write_shard(shard_id, ids, vecs)
                    db.executemany("DELETE FROM vector_shards WHERE shard_id=?", [(s.shard_id,) for s in shards])
                if not only_small:
                    db.execute("DELETE FROM removed")
            for s in segs:
                if s.seg_id not in {r[0] for r in self._db.execute("SELECT seg_id FROM segments")}:
                    shutil.rmtree(self.seg_dir / f"{s.seg_id:08d}", ignore_errors=True)
            live_shards = {r[0] for r in self._db.execute("SELECT shard_id FROM vector_shards")}
            for s in shards:
                if s.shard_id not in live_shards:
                    (self.vec_dir / f"{s.shard_id:08d}.npz").unlink(missing_ok=True)
            self.reload()

    # ---------- thống kê ----------
    @property
    def documents(self) -> list[DocInfo]:
        with self._lock:
            self._ensure_fresh()
            if self._doc_list is None:
                self._doc_list = [
                    DocInfo(doc_id, name, n_chunks, n_shingles, doc_idx in self._with_vectors)
                    for doc_id, (doc_idx, name, n_chunks, n_shingles) in self._docs.items()
                ]
            return self._doc_list

    def stats(self) -> dict[str, int | float]:
        with self._lock:
            self._ensure_fresh()
            vec_bytes = sum(s.vectors.nbytes + (0 if s.coarse is s.vectors else s.coarse.nbytes)
                            for s in self._shards.values())
            lex_bytes = sum(s.nbytes for s in self._segments.values()) + self._alive.nbytes * 9 + self._idf.nbytes
            return {
                "documents": len(self._docs), "chunks": self._n_alive,
                "chunks_with_vectors": int(sum(s.alive.sum() for s in self._shards.values())),
                "segments": len(self._segments), "vector_shards": len(self._shards),
                "lexical_mb": round(lex_bytes / 2**20, 1), "vector_mb": round(vec_bytes / 2**20, 1),
                "load_seconds": round(self.load_seconds, 2),
            }

    def chunk(self, chunk_id: int) -> tuple[str, int | None]:
        row = self._db.execute("SELECT text, page FROM chunks WHERE chunk_id=?", (int(chunk_id),)).fetchone()
        if row is None:
            return "", None
        return row[0], (None if row[1] is None or row[1] < 0 else int(row[1]))

    def doc_chunk(self, doc_id: str, doc_chunk: int) -> tuple[str, int | None]:
        with self._lock:
            self._ensure_fresh()
            info = self._docs.get(doc_id)
        if info is None:
            return "", None
        row = self._db.execute("SELECT text, page FROM chunks WHERE doc_idx=? AND ord=?", (info[0], int(doc_chunk))).fetchone()
        if row is None:
            return "", None
        return row[0], (None if row[1] is None or row[1] < 0 else int(row[1]))

    def _texts(self, chunk_ids: np.ndarray) -> list[str]:
        ids = [int(c) for c in chunk_ids]
        found: dict[int, str] = {}
        for start in range(0, len(ids), 900):  # giới hạn số tham số của SQLite
            part = ids[start : start + 900]
            found.update(self._db.execute(
                f"SELECT chunk_id, text FROM chunks WHERE chunk_id IN ({','.join('?' * len(part))})", part).fetchall())
        return [found.get(i, "") for i in ids]

    # ---------- truy vấn ----------
    def _lookup(self, field_hash: str, field_owner: str, query: np.ndarray, cap: int, owner_map: str | None = None):
        """Tra ``query`` trên mọi segment, bỏ dấu xuất hiện > ``cap`` lần TRÊN TOÀN KHO (văn mẫu).

        Trả về (chỉ số query, chủ sở hữu) của các lần khớp và mặt nạ dấu văn mẫu.
        """
        segs = list(self._segments.values())
        bounds, total = [], np.zeros(len(query), dtype=np.int64)
        for s in segs:
            arr = getattr(s, field_hash)
            left = np.searchsorted(arr, query, "left")
            lengths = np.searchsorted(arr, query, "right") - left
            bounds.append((left, lengths))
            total += lengths
        boiler = total > cap
        hits, owners = [], []
        for s, (left, lengths) in zip(segs, bounds):
            keep = (lengths > 0) & ~boiler
            if not keep.any():
                continue
            pos = _expand_ranges(left[keep], lengths[keep])
            owner = getattr(s, field_owner)[pos]
            if owner_map is not None:
                owner = getattr(s, owner_map)[owner]
            hits.append(np.repeat(np.flatnonzero(keep), lengths[keep]))
            owners.append(owner)
        if not hits:
            return np.zeros(0, np.int64), np.zeros(0, np.int64), boiler
        return np.concatenate(hits).astype(np.int64), np.concatenate(owners).astype(np.int64), boiler

    def _exact_overlap(self, text: str) -> dict[int, float]:
        """Tỷ lệ 5-gram đặc trưng (ước lượng từ mẫu 1/4, bỏ văn mẫu) của ``text`` có trong từng tài liệu."""
        query = shingle_hashes(text, DOC_SHINGLE)
        if not len(query) or not self._segments:
            return {}
        _, owners, boiler = self._lookup("doc_hash", "doc_owner", query, DOC_FP_CAP)
        base = max(1, int((~boiler).sum()))
        docs, counts = np.unique(owners, return_counts=True)
        return {int(d): float(c) / base for d, c in zip(docs, counts)
                if c >= MIN_EXACT_SHARED and int(d) in self._doc_ids}

    def _span_matches(self, q_hash: np.ndarray, q_owner: np.ndarray, hit: np.ndarray, cids: np.ndarray,
                      n_q: int, flag: int) -> dict[tuple[int, int], tuple[int, int]]:
        """Trùng 3-gram rải rác: (đoạn mới, doc_idx) -> (số dấu chung, id đoạn kho khớp nhiều nhất).

        Với mỗi cụm ``SPAN_CHUNKS`` đoạn mới liên tiếp, đếm số dấu (giá trị hash phân biệt, vì các đoạn mới
        chồng lấn nhau) chung với từng tài liệu kho. Cụm đạt ``flag`` được gán cho đoạn mới góp nhiều dấu nhất.
        """
        keep = self._alive[cids]
        hit, cids = hit[keep], cids[keep]
        if flag <= 0 or not len(hit):
            return {}
        n_span = max(1, n_q - SPAN_CHUNKS + 1)
        owner = q_owner[hit]
        docs = self._chunk_doc[cids].astype(np.int64)
        values = q_hash[hit].view(np.int64)
        # Mỗi lần khớp thuộc các cụm bắt đầu từ owner - SPAN_CHUNKS + 1 đến owner.
        spans = np.concatenate([owner - k for k in range(SPAN_CHUNKS)])
        rep = np.tile(np.arange(len(hit)), SPAN_CHUNKS)
        ok = (spans >= 0) & (spans < n_span)
        spans, rep = spans[ok], rep[ok]
        triples = np.unique(np.stack([spans, docs[rep], values[rep]], axis=1), axis=0)
        pair, count = np.unique(triples[:, 0] * _PAIR + triples[:, 1], return_counts=True)
        result: dict[tuple[int, int], tuple[int, int]] = {}
        for p, c in zip(pair[count >= flag], count[count >= flag]):
            s, d = int(p // _PAIR), int(p % _PAIR)
            sel = (docs == d) & (owner >= s) & (owner < s + SPAN_CHUNKS)
            qs, q_counts = np.unique(owner[sel], return_counts=True)
            q = int(qs[np.argmax(q_counts)])
            gs, g_counts = np.unique(cids[sel & (owner == q)], return_counts=True)
            g = int(gs[np.argmax(g_counts)])
            if c > result.get((q, d), (0, -1))[0]:
                result[(q, d)] = (int(c), g)
        return result

    def _lexical_top(self, texts: Sequence[str], k: int, span_flag: int = SPAN_SHARED_FLAG):
        """Top-k đoạn kho cho mỗi đoạn mới: (TF-IDF cosine, tỷ lệ 3-gram chung, id đoạn, trùng rải rác)."""
        n_q = len(texts)
        best_s = np.zeros((n_q, k), dtype=np.float32)
        best_o = np.zeros((n_q, k), dtype=np.float32)
        best_i = np.full((n_q, k), -1, dtype=np.int64)
        q_hash, q_ptr = chunk_shingles(texts)
        if not len(q_hash) or not self._segments:
            return best_s, best_o, best_i, {}
        q_owner = np.repeat(np.arange(n_q, dtype=np.int64), np.diff(q_ptr))
        # Dấu vân tay xuất hiện ở quá nhiều đoạn (văn mẫu, cụm từ phổ biến) không giúp lọc -> bỏ qua.
        cap = CHUNK_FP_CAP
        hit, cids, boiler = self._lookup("fp_hash", "fp_row", q_hash, cap, owner_map="chunk_ids")
        # Mẫu số của tỷ lệ 3-gram chung: số dấu đặc trưng (không phải văn mẫu) của từng đoạn mới.
        base = np.maximum(np.bincount(q_owner[~boiler], minlength=n_q), MIN_OVERLAP_BASE).astype(np.float32)
        if not len(hit):
            return best_s, best_o, best_i, {}
        span = self._span_matches(q_hash, q_owner, hit, cids, n_q, span_flag)
        # Đếm số dấu chung cho từng cặp (đoạn mới, đoạn kho), bỏ đoạn đã xóa.
        pair, shared = np.unique(q_owner[hit] * _PAIR + cids, return_counts=True)
        qi, li = pair // _PAIR, pair % _PAIR
        keep = (shared >= MIN_SHARED) & self._alive[li]
        qi, li, shared = qi[keep], li[keep], shared[keep]
        if not len(qi):
            return best_s, best_o, best_i, {key: (c, g, 0.0, 0.0) for key, (c, g) in span.items()}
        # Giữ tối đa LEX_CANDIDATES ứng viên mỗi đoạn mới (nhiều dấu chung nhất).
        order = np.lexsort((-shared, qi))
        qi, li, shared = qi[order], li[order], shared[order]
        rank = np.arange(len(qi)) - np.searchsorted(qi, qi, "left")
        sel = rank < LEX_CANDIDATES
        qi, li, shared = qi[sel], li[sel], shared[sel]
        overlap = np.minimum(shared / base[qi], 1.0).astype(np.float32)
        # Chấm TF-IDF cosine chỉ trên các cặp ứng viên (văn bản lấy từ SQLite).
        uniq, pos = np.unique(li, return_inverse=True)
        cand = _tfidf(term_counts(self._texts(uniq)), self._idf)
        q = _tfidf(term_counts(texts), self._idf)
        scores = np.asarray(cand[pos].multiply(q[qi]).sum(axis=1)).ravel().astype(np.float32)
        order = np.lexsort((-np.maximum(scores, overlap), qi))
        qi, li, scores, overlap = qi[order], li[order], scores[order], overlap[order]
        rank = np.arange(len(qi)) - np.searchsorted(qi, qi, "left")
        top = rank < k
        best_s[qi[top], rank[top]] = scores[top]
        best_o[qi[top], rank[top]] = overlap[top]
        best_i[qi[top], rank[top]] = li[top]
        if span:
            # Gắn điểm TF-IDF / tỷ lệ 3-gram của cặp đoạn đại diện (nếu nó nằm trong các ứng viên đã chấm).
            scored = {(int(a), int(b)): (float(s), float(o)) for a, b, s, o in zip(qi, li, scores, overlap)}
            span = {key: (c, g, *scored.get((key[0], g), (0.0, 0.0))) for key, (c, g) in span.items()}
        return best_s, best_o, best_i, span

    def _semantic_top(self, qvec: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Top-k đoạn gần nghĩa nhất cho mỗi vector: (cosine, id đoạn, nền = TB láng giềng hạng 6-10)."""
        qvec = np.asarray(qvec, dtype=np.float32)
        qc = _head(qvec, self.coarse_dim)
        cand = max(COARSE_CANDIDATES if self.coarse_dim else k, SEMANTIC_BACKGROUND[1])
        best_s = np.full((len(qc), 0), -np.inf, dtype=np.float32)
        best_ids = np.zeros((len(qc), 0), dtype=np.int64)
        best_ref = np.zeros((len(qc), 0, 2), dtype=np.int64)  # (shard_id, hàng) cho tầng tinh
        for shard in self._shards.values():
            for start in range(0, len(shard.chunk_ids), SCAN_BLOCK):
                block = _scores_f16(qc, shard.coarse[start : start + SCAN_BLOCK])
                block[:, ~shard.alive[start : start + SCAN_BLOCK]] = -np.inf
                s, i = _topk(block, cand)
                rows = i + start
                best_s = np.concatenate([best_s, s], axis=1)
                best_ids = np.concatenate([best_ids, shard.chunk_ids[rows]], axis=1)
                ref = np.stack([np.full_like(rows, shard.shard_id), rows], axis=-1)
                best_ref = np.concatenate([best_ref, ref], axis=1)
                if best_s.shape[1] > cand:
                    best_s, keep = _topk(best_s, cand)
                    best_ids = np.take_along_axis(best_ids, keep, axis=1)
                    best_ref = np.take_along_axis(best_ref, keep[..., None], axis=1)
        if best_s.shape[1]:
            best_s, keep = _topk(best_s, cand)
            best_ids = np.take_along_axis(best_ids, keep, axis=1)
            best_ref = np.take_along_axis(best_ref, keep[..., None], axis=1)
        best_s = np.where(np.isfinite(best_s), best_s, np.nan)
        if not self.coarse_dim or not best_s.shape[1]:
            return best_s[:, :k], np.where(np.isnan(best_s), -1, best_ids)[:, :k], self._background(best_s)
        # Tầng tinh: cosine đủ số chiều trên ứng viên.
        fine = np.full(best_s.shape, -np.inf, dtype=np.float32)
        for q in range(len(qvec)):
            for c in range(best_s.shape[1]):
                if np.isfinite(best_s[q, c]):
                    shard_id, row = best_ref[q, c]
                    fine[q, c] = float(self._shards[int(shard_id)].vectors[row].astype(np.float32) @ qvec[q])
        fine_sorted, _ = _topk(fine, fine.shape[1])
        s, keep = _topk(fine, min(k, fine.shape[1]))
        ids = np.take_along_axis(best_ids, keep, axis=1)
        background = self._background(np.where(np.isfinite(fine_sorted), fine_sorted, np.nan))
        return np.where(np.isfinite(s), s, np.nan), np.where(np.isfinite(s), ids, -1), background

    @staticmethod
    def _background(sorted_scores: np.ndarray) -> np.ndarray:
        """Nền ngữ nghĩa của từng đoạn mới; NaN khi kho quá ít đoạn (khi đó chỉ dùng ngưỡng sàn)."""
        lo, hi = SEMANTIC_BACKGROUND
        part = sorted_scores[:, lo:hi]
        out = np.full(len(sorted_scores), np.nan, dtype=np.float32)
        if part.shape[1] == hi - lo:
            full = np.isfinite(part).all(axis=1)
            out[full] = part[full].mean(axis=1)
        return out

    def query(self, chunks: Sequence[Chunk], encoder: Encoder | None = None, top_k: int = 3,
              batch_size: int = 16, semantic_flag: float = SEMANTIC_FLAG,
              lexical_flag: float = LEXICAL_FLAG, overlap_flag: float = OVERLAP_FLAG,
              exact_flag: float = EXACT_FLAG, span_flag: int = SPAN_SHARED_FLAG,
              window_margin: float | None = WINDOW_MARGIN) -> QueryReport:
        """So khớp các đoạn mới với kho.

        ``span_flag=0`` tắt luật trùng 3-gram rải rác, ``window_margin=None`` tắt luật cửa sổ câu (để so với
        phiên bản trước).
        """
        if encoder is not None and encoder.name != self.vector_model:
            raise ValueError(f"Encoder {encoder.name} khác model của index ({self.vector_model}).")
        with self._lock:
            self._refresh(force=True)
            return self._query(chunks, encoder, top_k, batch_size, semantic_flag, lexical_flag, overlap_flag,
                               exact_flag, span_flag, window_margin)

    def _window_matches(self, windows: list[str], owners: list[int], wvec: np.ndarray,
                        window_margin: float) -> dict[tuple[int, int], tuple[float, str]]:
        """Luật cửa sổ câu: (đoạn mới, id đoạn kho) -> (cosine, cửa sổ) cho các cặp vừa gần nghĩa hơn nền của
        cửa sổ, vừa chung đủ từ đặc trưng (loại tin cùng thể loại mà e5-small chấm sát nhau)."""
        s, i, background = self._semantic_top(wvec, WINDOW_TOP)
        bg = np.where(np.isfinite(background), background, -np.inf)[:, None]
        ok = np.isfinite(s) & (i >= 0)
        ok[ok] = (s[ok] >= WINDOW_FLOOR) & ((s - bg)[ok] >= window_margin)
        rows, cols = np.nonzero(ok)
        if not len(rows):
            return {}
        cand = i[rows, cols]
        uniq, pos = np.unique(cand, return_inverse=True)
        lib = self._texts(uniq)
        contain = _containment([windows[r] for r in rows], [lib[p] for p in pos], self._idf)
        found: dict[tuple[int, int], tuple[float, str]] = {}
        for r, c, g, v in zip(rows, cols, cand, contain):
            if v < WINDOW_CONTAIN_FLAG:
                continue
            key = (owners[r], int(g))
            if float(s[r, c]) > found.get(key, (-1.0, ""))[0]:
                found[key] = (float(s[r, c]), windows[r])
        return found

    def _query(self, chunks, encoder, top_k, batch_size, semantic_flag, lexical_flag, overlap_flag,
               exact_flag, span_flag=SPAN_SHARED_FLAG, window_margin=WINDOW_MARGIN) -> QueryReport:
        timings: dict[str, float] = {}
        texts = [t for t, _ in chunks]
        n_q = len(texts)
        if not self._docs or not n_q:
            return QueryReport([], 0, n_q, False, 0, {})

        t = time.perf_counter()
        exact = self._exact_overlap("\n".join(texts))
        timings["5gram"] = time.perf_counter() - t

        t = time.perf_counter()
        lex_s, lex_o, lex_i, span = self._lexical_top(texts, top_k, span_flag)
        timings["lexical"] = time.perf_counter() - t

        sem_s = sem_i = None
        window_hits: dict[tuple[int, int], tuple[float, str]] = {}
        encoded = n_windows = 0
        use_semantic = encoder is not None and bool(self._shards)
        if use_semantic:
            need = np.flatnonzero(np.maximum(lex_s[:, 0], lex_o[:, 0]) < LEXICAL_SKIP_SEMANTIC)
            sem_s = np.full((n_q, top_k), np.nan, dtype=np.float32)
            sem_i = np.full((n_q, top_k), -1, dtype=np.int64)
            sem_thr = np.full(n_q, semantic_flag, dtype=np.float32)
            windows: list[str] = []
            owners: list[int] = []
            if window_margin is not None:
                for q in need:
                    ws = sentence_windows(texts[q])
                    if len(ws) > 1:  # đoạn chỉ có một cửa sổ thì cửa sổ chính là đoạn
                        windows += ws
                        owners += [int(q)] * len(ws)
            if len(need):
                t = time.perf_counter()
                # Encode riêng đoạn và cửa sổ: chung một batch thì cửa sổ ngắn bị đệm tới độ dài đoạn 180 từ.
                qvec = np.asarray(encoder.encode([texts[i] for i in need], batch_size=batch_size))
                wvec = np.asarray(encoder.encode(windows, batch_size=batch_size)) if windows else None
                timings["encode"] = time.perf_counter() - t
                encoded, n_windows = len(need), len(windows)
                t = time.perf_counter()
                s, i, background = self._semantic_top(qvec, top_k)
                sem_thr[need] = np.fmax(semantic_flag, background + SEMANTIC_MARGIN)
                sem_s[need, : s.shape[1]], sem_i[need, : i.shape[1]] = s, i
                timings["vector_search"] = time.perf_counter() - t
                if windows:
                    t = time.perf_counter()
                    window_hits = self._window_matches(windows, owners, wvec, window_margin)
                    timings["windows"] = time.perf_counter() - t

        t = time.perf_counter()
        per_doc: dict[int, list[Match]] = {}
        flagged_by_doc: dict[int, int] = {}
        best_lex: dict[int, float] = {}
        best_ovl: dict[int, float] = {}
        best_sem: dict[int, float] = {}
        best_span: dict[int, int] = {}
        span_by_q: dict[int, list[tuple[int, float, float]]] = {}   # đoạn mới -> [(id đoạn kho, tfidf, 3-gram)]
        for (q, d), (count, g, lex, ovl) in span.items():
            span_by_q.setdefault(q, []).append((g, lex, ovl))
            best_span[d] = max(best_span.get(d, 0), count)
        window_by_q: dict[int, list[tuple[int, float, str]]] = {}
        for (q, g), (sim, window) in window_hits.items():
            window_by_q.setdefault(q, []).append((g, sim, window))
        flagged_total = 0
        for q in range(n_q):
            candidates: dict[int, list] = {}   # id đoạn kho -> [tfidf, 3-gram, ngữ nghĩa, lý do, cửa sổ]
            for s, o, i in zip(lex_s[q], lex_o[q], lex_i[q]):
                if i >= 0 and (s > 0 or o > 0):
                    candidates[int(i)] = [float(s), float(o), None, [], None]
            if sem_s is not None:
                for s, i in zip(sem_s[q], sem_i[q]):
                    if i >= 0 and np.isfinite(s):
                        cand = candidates.setdefault(int(i), [0.0, 0.0, None, [], None])
                        cand[2] = float(s)
                        if s >= sem_thr[q]:
                            cand[3].append("ngữ nghĩa")
            for g, lex, ovl in span_by_q.get(q, []):
                cand = candidates.setdefault(g, [lex, ovl, None, [], None])
                cand[3].append("3-gram rải rác")
            for g, sim, window in window_by_q.get(q, []):
                cand = candidates.setdefault(g, [0.0, 0.0, None, [], None])
                cand[2] = max(cand[2] if cand[2] is not None else -1.0, sim)
                cand[3].append("câu")
                cand[4] = window
            flagged_docs: set[int] = set()
            for g, (lex, ovl, sem, reasons, window) in candidates.items():
                d = int(self._chunk_doc[g])
                if d not in self._doc_ids:
                    continue
                best_lex[d] = max(best_lex.get(d, 0.0), lex)
                best_ovl[d] = max(best_ovl.get(d, 0.0), ovl)
                if sem is not None:
                    best_sem[d] = max(best_sem.get(d, -1.0), sem)
                reasons = ["3-gram"] * (ovl >= overlap_flag) + ["tfidf"] * (lex >= lexical_flag) + reasons
                if reasons:
                    flagged_docs.add(d)
                    per_doc.setdefault(d, []).append(
                        Match(q, self._doc_ids[d], int(self._chunk_ord[g]), lex, sem, ovl, tuple(reasons), window))
            for d in flagged_docs:
                flagged_by_doc[d] = flagged_by_doc.get(d, 0) + 1
            flagged_total += bool(flagged_docs)

        # Chỉ báo tài liệu có bằng chứng: bị đánh dấu, trùng 5-gram, hoặc đoạn gần nhất đủ gần để đáng xem.
        relevant = set(exact) | set(per_doc) | {d for d, v in best_lex.items() if v >= 0.2} \
            | {d for d, v in best_ovl.items() if v >= 0.1} | {d for d, v in best_sem.items() if v >= SEMANTIC_REPORT}
        documents = []
        for d in relevant:
            doc_id = self._doc_ids[d]
            coverage = flagged_by_doc.get(d, 0) / n_q
            documents.append(DocResult(
                doc_id=doc_id, name=self._docs[doc_id][1],
                exact_5gram=exact.get(d, 0.0), best_lexical=best_lex.get(d, 0.0),
                best_semantic=best_sem.get(d), coverage=coverage, best_overlap=best_ovl.get(d, 0.0),
                flagged=coverage > 0 or exact.get(d, 0.0) >= exact_flag, span_shared=best_span.get(d, 0),
                matches=sorted(per_doc.get(d, []), key=lambda m: -max(m.lexical, m.overlap, m.semantic or 0.0)),
            ))
        documents.sort(key=lambda r: (r.flagged, r.coverage, r.exact_5gram, r.best_overlap, r.best_semantic or 0,
                                      r.best_lexical), reverse=True)
        timings["aggregate"] = time.perf_counter() - t
        return QueryReport(documents, flagged_total, n_q, use_semantic, encoded, timings, n_windows)
