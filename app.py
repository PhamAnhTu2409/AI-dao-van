from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import streamlit as st

from documents import ALLOWED_SUFFIXES, load_chunks
from indexer import (EXACT_FLAG, INDEX_DIR, LEXICAL_FLAG, OVERLAP_FLAG, SEMANTIC_FLAG, SEMANTIC_MARGIN,
                     SPAN_SHARED_FLAG, WINDOW_CONTAIN_FLAG, LibraryIndex)

ROOT = Path(__file__).resolve().parent
LIBRARY_DIR = ROOT / "data" / "library"
MATCHES_SHOWN = 10
# Chọn index khác (vd. kho báo 100k bài: data/index_vietnews) bằng biến môi trường.
ACTIVE_INDEX_DIR = Path(os.environ.get("DAOVAN_INDEX_DIR") or INDEX_DIR)
CORPUS_PREFIX = "corpus:"  # tài liệu nạp bằng script (không có tệp trong data/library)


def ensure_dirs() -> None:
    LIBRARY_DIR.mkdir(parents=True, exist_ok=True)


def get_files() -> list[Path]:
    ensure_dirs()
    return sorted(
        (p for p in LIBRARY_DIR.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_SUFFIXES),
        key=lambda p: display_name(p).casefold(),
    )


def display_name(path: Path) -> str:
    name_path = LIBRARY_DIR / f"{path.stem}.name"
    try:
        return name_path.read_text(encoding="utf-8") if name_path.exists() else path.name
    except OSError:
        return path.name


@st.cache_resource(show_spinner="Đang nạp index kho tài liệu…")
def get_index() -> LibraryIndex:
    return LibraryIndex(ACTIVE_INDEX_DIR)


@st.cache_resource(show_spinner="Đang nạp model ngữ nghĩa (multilingual-e5-small)…")
def load_encoder():
    """Trả về ``(encoder, None)`` hoặc ``(None, lý do)`` để app chạy tiếp chế độ từ vựng."""
    try:
        from embedder import FastEncoder

        return FastEncoder(), None
    except Exception as exc:  # thiếu file, thiếu thư viện, DLL bị chặn... đều rơi về từ vựng
        return None, f"{type(exc).__name__}: {exc}"


def index_document(index: LibraryIndex, encoder, path: Path) -> None:
    index.add_document(path.stem, display_name(path), load_chunks(path.name, path.read_bytes()), encoder)


def add_to_library(index: LibraryIndex, encoder, name: str, payload: bytes) -> bool:
    load_chunks(name, payload)  # kiểm tra đọc được trước khi lưu
    digest = hashlib.sha256(payload).hexdigest()
    target = LIBRARY_DIR / f"{digest}{Path(name).suffix.lower()}"
    if target.exists() and index.has(digest):
        return False
    target.write_bytes(payload)
    (LIBRARY_DIR / f"{digest}.name").write_text(Path(name).name, encoding="utf-8")
    index_document(index, encoder, target)
    return True


def remove_from_library(index: LibraryIndex, path: Path) -> None:
    path.unlink(missing_ok=True)
    (LIBRARY_DIR / f"{path.stem}.name").unlink(missing_ok=True)
    index.remove_document(path.stem)


def sync_index(index: LibraryIndex, encoder, files: list[Path]) -> None:
    """Index các tệp chưa có trong index (hoặc chưa có vector) và gỡ mục của tệp đã xóa."""
    on_disk = {p.stem for p in files}
    for doc in index.documents:
        if doc.doc_id not in on_disk and not doc.doc_id.startswith(CORPUS_PREFIX):
            index.remove_document(doc.doc_id)
    todo = [p for p in files if not index.has(p.stem) or (encoder is not None and not index.has_vectors(p.stem))]
    if not todo:
        return
    bar = st.progress(0.0, text="Đang index kho tài liệu…")
    try:
        for n, path in enumerate(todo, 1):
            bar.progress((n - 1) / len(todo), text=f"Index {n}/{len(todo)}: {display_name(path)}")
            try:
                index_document(index, encoder, path)
            except (ValueError, OSError) as exc:
                st.warning(f"Không index được {display_name(path)}: {exc}")
    finally:
        bar.empty()


def location(page: int | None) -> str:
    return "" if page is None else f" · trang {page}"


st.set_page_config(page_title="Kiểm tra tương đồng tài liệu", page_icon="📄", layout="wide")
st.title("📄 Kiểm tra tương đồng tài liệu")
st.caption("Đối chiếu tài liệu mới với kho tài liệu cũ trên máy này. Điểm số là chỉ dấu để rà soát, không phải kết luận đạo văn.")
ensure_dirs()

index = get_index()
encoder, semantic_error = load_encoder()
if encoder is None:
    st.warning(
        "Không nạp được model ngữ nghĩa nên **điểm Ngữ nghĩa đang tắt**; ứng dụng chạy chế độ từ vựng "
        f"(TF-IDF và 5-gram). Lý do: {semantic_error}"
    )

with st.sidebar:
    st.header("Thư viện tài liệu cũ")
    st.caption("Tệp lưu cục bộ trong `data/library`, index trong `data/index`.")
    old_uploads = st.file_uploader(
        "Thêm tài liệu nguồn", type=["pdf", "docx", "txt"], accept_multiple_files=True, key="library_upload"
    )
    if st.button("Lưu tài liệu vào thư viện", disabled=not old_uploads):
        added, skipped = 0, 0
        with st.spinner("Đang đọc và index tài liệu…"):
            for item in old_uploads or []:
                try:
                    was_added = add_to_library(index, encoder, item.name, item.getvalue())
                    added += int(was_added)
                    skipped += int(not was_added)
                except Exception as exc:
                    st.error(f"{Path(item.name).name}: {exc}")
        if added:
            st.success(f"Đã thêm {added} tài liệu.")
        if skipped:
            st.info(f"Bỏ qua {skipped} tài liệu có cùng nội dung.")
        st.rerun()

    files = get_files()
    sync_index(index, encoder, files)
    stats = index.stats()
    st.write(f"**{len(files)}** tệp thư viện · index: **{stats['documents']:,}** tài liệu, **{stats['chunks']:,}** đoạn")
    if files:
        selected = st.selectbox("Chọn tài liệu để xóa", files, format_func=display_name, key="delete_doc")
        if st.button("Xóa tài liệu đã chọn"):
            remove_from_library(index, selected)
            st.rerun()

    st.divider()
    if encoder is not None:
        st.caption(f"Ngữ nghĩa: {encoder.name} trên CPU — bật")
    else:
        st.caption("Ngữ nghĩa: tắt (chạy chế độ từ vựng)")
        if st.button("Thử nạp lại model"):
            load_encoder.clear()
            st.rerun()
    st.caption(f"Index: {stats['lexical_mb'] + stats['vector_mb']:.0f} MB RAM, nạp {stats['load_seconds']} s")

st.subheader("Tài liệu mới cần kiểm tra")
new_upload = st.file_uploader("Chọn một PDF, DOCX hoặc TXT", type=["pdf", "docx", "txt"], key="new_doc")
if st.button("Kiểm tra trùng lặp", type="primary", disabled=new_upload is None):
    try:
        started = time.perf_counter()
        new_chunks = load_chunks(new_upload.name, new_upload.getvalue())
        extract_seconds = time.perf_counter() - started
        if not index.documents:
            st.warning("Thư viện tài liệu cũ đang trống. Hãy thêm tài liệu nguồn ở thanh bên.")
        else:
            with st.spinner(f"Đang so khớp {len(new_chunks)} đoạn với {stats['chunks']:,} đoạn trong kho…"):
                report = index.query(new_chunks, encoder)
            total = time.perf_counter() - started
            sources = [r for r in report.documents if r.flagged]

            c1, c2, c3 = st.columns(3)
            c1.metric("Đoạn nghi trùng", f"{report.flagged_chunks}/{report.total_chunks}",
                      f"{report.flagged_chunks / max(report.total_chunks, 1):.0%} tài liệu", delta_color="off")
            c2.metric("Nguồn liên quan", len(sources))
            c3.metric("Thời gian kiểm tra", f"{total:.2f} s")
            parts = {"đọc tệp": extract_seconds, **report.timings}
            st.caption(
                " · ".join(f"{k} {v * 1000:.0f} ms" for k, v in parts.items())
                + (f" · đã encode {report.encoded_chunks} đoạn" if report.semantic_used else " · ngữ nghĩa tắt")
            )

            if not sources:
                st.success("Không tìm thấy đoạn nào vượt ngưỡng nghi trùng trong kho.")
            else:
                st.subheader("Nguồn có đoạn tương đồng")
                st.caption(
                    f"Phủ: tỷ lệ đoạn của tài liệu mới khớp với nguồn (Trùng 3-gram ≥ {OVERLAP_FLAG:.0%}, "
                    f"TF-IDF ≥ {LEXICAL_FLAG:.0%} hoặc Ngữ nghĩa ≥ {SEMANTIC_FLAG:.0%} và cao hơn nền của đoạn "
                    f"≥ {SEMANTIC_MARGIN * 100:.0f} điểm). Trùng 5-gram: tỷ lệ cụm 5 từ "
                    f"đặc trưng của cả tài liệu mới có trong nguồn (≥ {EXACT_FLAG:.0%} là nghi chép; cụm văn mẫu phổ "
                    "biến không tính). Trùng 3-gram: tỷ lệ cụm 3 từ đặc trưng của một đoạn mới có trong đoạn nguồn — "
                    "chịu được sửa lẻ tẻ. Ngữ nghĩa bắt được đoạn viết lại bằng từ khác; trong kho lớn cùng lĩnh vực, "
                    "đoạn chỉ cùng chủ đề đã đạt 0,92–0,94 nên ngưỡng tự nâng theo nền (láng giềng gần thứ 6–10). "
                    f"Ngoài ra một đoạn còn bị đánh dấu khi: hai đoạn mới liên tiếp chung ≥ {SPAN_SHARED_FLAG} dấu "
                    "3-gram đặc trưng với nguồn (\"3-gram rải rác\": chép rồi sửa nhiều, xáo câu); hoặc một cửa sổ "
                    "~60 từ gần nghĩa với đoạn nguồn hơn nền và có ≥ "
                    f"{WINDOW_CONTAIN_FLAG:.0%} từ đặc trưng (theo IDF) nằm trong đoạn nguồn (\"câu\")."
                )
                st.dataframe(
                    [
                        {
                            "Tài liệu": r.name, "Phủ": r.coverage, "Trùng 5-gram": r.exact_5gram,
                            "Trùng 3-gram": r.best_overlap, "TF-IDF": r.best_lexical, "Ngữ nghĩa": r.best_semantic,
                        }
                        for r in sources
                    ],
                    column_config={
                        name: st.column_config.ProgressColumn(name, min_value=0.0, max_value=1.0, format="percent")
                        for name in ("Phủ", "Trùng 5-gram", "Trùng 3-gram", "TF-IDF", "Ngữ nghĩa")
                    },
                    hide_index=True,
                    width="stretch",
                )
                for n, result in enumerate(sources):
                    with st.expander(
                        f"{result.name} — phủ {result.coverage:.0%} · trùng 5-gram {result.exact_5gram:.0%} "
                        f"· {len(result.matches)} cặp đoạn",
                        expanded=n == 0,
                    ):
                        if not result.matches:
                            st.write("Trùng nhiều cụm 5 từ với tài liệu mới nhưng không đoạn riêng lẻ nào vượt ngưỡng.")
                        for match in result.matches[:MATCHES_SHOWN]:
                            new_text, new_page = new_chunks[match.query_chunk]
                            old_text, old_page = index.doc_chunk(result.doc_id, match.doc_chunk)
                            sem = "—" if match.semantic is None else f"{match.semantic:.0%}"
                            st.markdown(f"**Trùng 3-gram {match.overlap:.0%} · TF-IDF {match.lexical:.0%} · Ngữ nghĩa {sem}**"
                                        + (f" — theo: {', '.join(match.reasons)}" if match.reasons else ""))
                            if match.window:
                                st.caption(f"Cửa sổ khớp: “{match.window}”")
                            left, right = st.columns(2)
                            with left:
                                st.markdown(f"*Tài liệu mới — đoạn {match.query_chunk + 1}{location(new_page)}*")
                                st.write(new_text)
                            with right:
                                st.markdown(f"*{result.name} — đoạn {match.doc_chunk + 1}{location(old_page)}*")
                                st.write(old_text)
                        if len(result.matches) > MATCHES_SHOWN:
                            st.caption(f"… và {len(result.matches) - MATCHES_SHOWN} cặp đoạn khác.")
    except Exception as exc:
        st.error(str(exc))

st.divider()
st.markdown("**Quyền riêng tư:** Tài liệu được xử lý và lưu cục bộ; ứng dụng không gửi nội dung tới API đám mây. Xóa tài liệu trong sidebar, hoặc xóa `data/library` và `data/index`.")
st.caption("PDF scan chưa hỗ trợ OCR. Giới hạn tải lên: 25 MB/tệp.")
