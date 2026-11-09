"""Kiểm thử đầu-cuối với văn bản tiếng Việt thật và model e5-small (index tạm, không đụng data/).

    .venv\\Scripts\\python.exe scripts\\e2e_check.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from documents import load_chunks  # noqa: E402
from embedder import FastEncoder  # noqa: E402
from indexer import LibraryIndex  # noqa: E402

LIBRARY = {
    "bhxh.txt": "Người sử dụng lao động có trách nhiệm đóng bảo hiểm xã hội bắt buộc cho người lao động theo quy định của pháp luật. Hằng tháng, doanh nghiệp phải trích từ tiền lương của người lao động một tỷ lệ nhất định và nộp cùng với phần đóng của doanh nghiệp vào quỹ bảo hiểm xã hội. Nếu chậm đóng, doanh nghiệp sẽ bị tính lãi và có thể bị xử phạt hành chính.",
    "khi_hau.txt": "Biến đổi khí hậu đang làm gia tăng tần suất và cường độ của các hiện tượng thời tiết cực đoan như bão, lũ lụt và hạn hán. Đồng bằng sông Cửu Long là một trong những khu vực chịu ảnh hưởng nặng nề nhất do nước biển dâng và xâm nhập mặn, đe dọa trực tiếp đến sản xuất nông nghiệp và sinh kế của hàng triệu người dân.",
    "hoc_may.txt": "Học máy là một nhánh của trí tuệ nhân tạo, cho phép máy tính tự học từ dữ liệu mà không cần được lập trình tường minh cho từng nhiệm vụ. Các mô hình học sâu sử dụng mạng nơ-ron nhiều lớp để trích xuất đặc trưng phức tạp, nhờ đó đạt kết quả vượt trội trong nhận dạng hình ảnh và xử lý ngôn ngữ tự nhiên.",
    "thue_tncn.txt": "Cá nhân có thu nhập từ tiền lương, tiền công vượt mức giảm trừ gia cảnh phải nộp thuế thu nhập cá nhân theo biểu thuế lũy tiến từng phần. Người nộp thuế có thể ủy quyền cho tổ chức trả thu nhập quyết toán thay nếu chỉ có một nguồn thu nhập trong năm.",
}
QUERIES = {
    # (văn bản, nguồn kỳ vọng hoặc None nếu không được đánh dấu)
    "chép nguyên văn": (LIBRARY["bhxh.txt"], "bhxh.txt"),
    "chép rồi sửa vài từ": (
        "Biến đổi khí hậu đang làm tăng tần suất và cường độ của những hiện tượng thời tiết cực đoan như bão, lũ "
        "và hạn hán. Đồng bằng sông Cửu Long là một trong các khu vực bị ảnh hưởng nặng nề nhất do nước biển dâng "
        "và xâm nhập mặn, đe dọa trực tiếp tới sản xuất nông nghiệp và sinh kế của hàng triệu người dân.",
        "khi_hau.txt",
    ),
    "viết lại bằng từ khác": (
        "Machine learning là lĩnh vực con của AI giúp máy tính có khả năng rút ra quy luật từ dữ liệu thay vì phải "
        "viết chương trình cụ thể cho mỗi việc. Deep learning dùng mạng nơ-ron có nhiều tầng để học ra các đặc trưng "
        "phức tạp, vì thế cho kết quả rất tốt trong các bài toán thị giác máy tính và xử lý ngôn ngữ.",
        "hoc_may.txt",
    ),
    "khác chủ đề": (
        "Phở là món ăn truyền thống nổi tiếng của Việt Nam, gồm bánh phở mềm, nước dùng ninh từ xương bò trong nhiều "
        "giờ cùng quế, hồi, gừng nướng, thường ăn kèm rau thơm, chanh và ớt tươi vào buổi sáng.",
        None,
    ),
}


def main() -> None:
    encoder = FastEncoder()
    with tempfile.TemporaryDirectory() as tmp:
        index = LibraryIndex(tmp)
        try:
            for name, text in LIBRARY.items():
                index.add_document(name, name, load_chunks(name, text.encode("utf-8")), encoder)
            failures = 0
            for label, (text, expected) in QUERIES.items():
                t = time.perf_counter()
                report = index.query(load_chunks("q.txt", text.encode("utf-8")), encoder)
                ms = (time.perf_counter() - t) * 1000
                flagged = [r for r in report.documents if r.coverage > 0]
                top = flagged[0].name if flagged else None
                ok = top == expected
                failures += not ok
                best = max(report.documents, key=lambda r: r.best_semantic or 0, default=None)
                detail = "" if best is None else (
                    f"gần nhất {best.name}: tfidf {best.best_lexical:.2f}, ngữ nghĩa {best.best_semantic or 0:.3f}, "
                    f"5-gram {best.exact_5gram:.0%}"
                )
                print(f"[{'OK ' if ok else 'SAI'}] {label}: đánh dấu {top!r} (kỳ vọng {expected!r}) — {detail} — {ms:.0f} ms")
            sys.exit(1 if failures else 0)
        finally:
            index.close()


if __name__ == "__main__":
    main()
