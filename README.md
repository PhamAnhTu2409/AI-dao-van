# Kiểm tra tương đồng tài liệu (local)

Ứng dụng Streamlit chạy trên máy Windows, nhận PDF/DOCX/TXT, lưu và index kho tài liệu nguồn cục bộ, rồi đối chiếu tài liệu mới với kho theo ba lớp: trùng nguyên văn, từ vựng và ngữ nghĩa. Kết quả là chỉ dấu để con người rà soát, không phải kết luận đạo văn hay xác định nguồn gốc văn bản.

## Máy đã được chuẩn bị

- Windows 11, Intel Core i7-14700, RAM 16 GiB, Intel UHD; cấu hình hiện tại dùng CPU, không dùng CUDA.
- Python 3.11 và môi trường ảo `.venv`.
- Các thư viện giao diện, đọc PDF/DOCX, TF-IDF đã cài theo `requirements.txt`.
- Model đã tải về bản revision cố định vào `models/`: **multilingual-e5-small** (~0.5 GB, model ngữ nghĩa mặc định), DeepX Embedding v1 và Qwen3-1.7B (hai model sau hiện không dùng trong luồng mặc định).
- PyTorch `2.14.1+cpu` trong `.venv` hiện **import và chạy suy luận được** (kiểm tra ngày 2026-10-19). Lỗi khởi tạo DLL `c10.dll` (WinError 1114) ghi nhận trước đây không còn tái hiện. 

## Cài đặt / chạy

Mở PowerShell tại thư mục dự án:

```powershell
.\scripts\setup.ps1
```

Chạy ứng dụng:

```powershell
.\scripts\run.ps1
```

Mở `http://127.0.0.1:8501`. Server chỉ bind localhost; không đưa lên mạng công cộng. Dừng bằng Ctrl+C trong terminal chạy app.

Tải lại model nếu cần:

```powershell
.\scripts\download_models.ps1
```

Index lại toàn bộ kho từ dòng lệnh (chạy lại an toàn, chỉ xử lý tài liệu mới; `--rebuild` để làm lại từ đầu, `--lexical` để bỏ qua vector):

```powershell
.\.venv\Scripts\python.exe scripts\build_index.py
```

Script tải các file tại revisions cố định, không chạy mã từ Hugging Face. Mã kiến trúc DeepX nằm ở `vendor/deepx-embed`; model ở `models/deepx-embedding-v1`. `semantic.py` thêm `vendor/deepx-embed` vào `sys.path` để dùng lớp mô hình, nhưng không dùng wrapper `DeepXEmbed` của upstream (xem ghi chú model).

## Cách hệ thống kiểm tra

Mọi phép tính nặng trên kho làm **một lần khi thêm tài liệu**; lúc kiểm tra chỉ xử lý tài liệu mới rồi tra index (`indexer.py`). Index nằm ở `data/index` (hoặc thư mục trong biến môi trường `DAOVAN_INDEX_DIR`):

- `catalog.sqlite` (chế độ WAL): danh sách tài liệu, văn bản từng đoạn, danh sách segment. SQLite lo khóa ghi nên script nạp kho chạy song song được với app. App chỉ nạp phần mới khi kho thay đổi, không dựng lại toàn bộ.
- `segments/<id>/*.npy`: dấu vân tay của một lô tài liệu, bất biến; các mảnh nhỏ (thêm từng tệp từ app) được gộp tự động.
- `vectors/<model>/<id>.npz`: vector đoạn (float16), encode được dần và tiếp tục nếu bị ngắt.
- `df.npy`: thống kê tần suất cho IDF.

Ba lớp so khớp:

1. **Trùng 5-gram** (cấp tài liệu): tỷ lệ cụm 5 từ *đặc trưng* của cả tài liệu mới có trong từng nguồn.
   - Chỉ lưu và tra mẫu 1/4 số cụm (theo giá trị hash, nên kho và truy vấn lấy cùng một mẫu và tỷ lệ vẫn đúng).
   - Cụm có trong hơn 25 tài liệu là văn mẫu ("theo thông tin từ…") nên không được tính.
2. **Từ vựng** (cấp đoạn 180 từ):
   - Lọc ứng viên: chỉ đoạn kho chung ít nhất 2 cụm 3 từ (đã lấy mẫu) với đoạn mới mới được xét; cụm có trong hơn 50 đoạn bị bỏ.
   - Chấm hai điểm cho mỗi ứng viên:
     - **Trùng 3-gram**: tỷ lệ cụm 3 từ đặc trưng của đoạn mới có trong đoạn kho. Điểm này chịu được sửa lẻ tẻ và không bị pha loãng khi chỉ một phần đoạn bị chép.
     - **TF-IDF cosine**: văn bản ứng viên lấy từ SQLite và vector hóa lại lúc truy vấn, nên không phải giữ ma trận TF-IDF toàn kho trong RAM.
   - **Trùng 3-gram rải rác**: đếm số dấu 3-gram đặc trưng (đã lấy mẫu, bỏ văn mẫu) mà **2 đoạn mới liên tiếp** (~325 từ) chung với **cả một tài liệu kho**. Đoạn chép rồi sửa nhiều và xáo câu thường vắt qua ranh giới đoạn ở cả hai phía, nên tỷ lệ theo từng cặp đoạn bị chia nhỏ; đếm theo cụm đoạn × tài liệu thì không. Vẫn là đếm cục bộ, nên tài liệu mới dài không làm tăng điểm.
3. **Ngữ nghĩa**: vector `multilingual-e5-small` (384 chiều) của từng đoạn kho lưu sẵn.
   - Quét chính xác toàn kho bằng phép nhân ma trận float16 qua PyTorch: ~20 ms cho 365k vector. Nếu không nạp được torch thì dùng numpy, chậm hơn nhiều.
   - Đoạn đã trùng từ vựng rõ (≥ 0.8) thì không encode.
   - **Cửa sổ câu**: ngoài cả đoạn 180 từ, mỗi đoạn mới còn được cắt thành các cửa sổ ~60 từ. Mỗi cửa sổ bắt đầu ở đầu câu, cửa sổ sau cách cửa sổ trước ≥ 30 từ, nên có ~6 cửa sổ mỗi đoạn. Mỗi cửa sổ được encode rồi so với **chính vector đoạn kho sẵn có**, không phải encode lại kho. Riêng e5-small ở cấp này vẫn không đủ phân biệt (tin cùng thể loại sát nền), nên một cặp (cửa sổ, đoạn kho) chỉ được tính khi:
     - cosine ≥ 0.85 và cao hơn nền của chính cửa sổ ≥ 0.03;
     - ≥ 50% trọng số IDF của các từ/cặp từ trong cửa sổ có mặt trong đoạn kho (xác minh từ vựng).

Một đoạn bị **đánh dấu nghi trùng** với một nguồn khi thỏa một trong các luật sau:
- Trùng 3-gram ≥ 0.25;
- TF-IDF ≥ 0.50;
- Ngữ nghĩa ≥ 0.90 **và** cao hơn nền của chính đoạn đó ≥ 0.04 (ngưỡng thích ứng, xem mục kho thử nghiệm);
- Trùng 3-gram rải rác ≥ 8 dấu (mẫu 1/4, tức ~32 cụm 3 từ đặc trưng thật). Cụm đoạn đạt ngưỡng được gán cho đoạn mới góp nhiều dấu nhất;
- có cửa sổ câu thỏa luật trên.

Mỗi cặp đoạn ghi lại luật đã đánh dấu (`Match.reasons`: `3-gram`, `tfidf`, `ngữ nghĩa`, `3-gram rải rác`, `câu`) và cửa sổ câu khớp (`Match.window`). App hiển thị cả hai để người rà soát biết vì sao một cặp bị đánh dấu. Muốn so với phiên bản trước, gọi `query(..., span_flag=0, window_margin=None)`.

Một nguồn bị **đánh dấu** khi có đoạn bị đánh dấu, hoặc Trùng 5-gram ≥ 10%. Kết quả cho mỗi nguồn gồm:
- **Phủ:** tỷ lệ đoạn của tài liệu mới bị đánh dấu với nguồn đó;
- Trùng 5-gram;
- điểm Trùng 3-gram, TF-IDF và Ngữ nghĩa cao nhất;
- các cặp đoạn kèm số trang PDF.

Các ngưỡng từ vựng đã hiệu chỉnh trên kho báo 100k bài (xem mục dưới); hằng số nằm trong `indexer.py`.

## Kho thử nghiệm 100k bài báo (VietNews)

Bộ [harouzie/vietnews](https://huggingface.co/datasets/harouzie/vietnews) (Apache-2.0) đã tải vào `data/corpus/`:
- train: 99,134 bài, dùng làm kho;
- validation: 22,184 bài ngoài kho, dùng làm văn bản "của người viết" và truy vấn âm tính.

Văn bản gốc đã tách từ (`Cơ_quan`), nên được trả về dạng viết thường trước khi index (`corpus.py`).

```powershell
.\.venv\Scripts\python.exe scripts\import_corpus.py             # nạp 99k bài vào data\index_vietnews (~2 phút)
.\.venv\Scripts\python.exe scripts\import_corpus.py --vectors   # thêm vector e5-small (~2.5 giờ, chạy lại để tiếp tục)
.\.venv\Scripts\python.exe scripts\eval_corpus.py               # đánh giá chỉ từ vựng
.\.venv\Scripts\python.exe scripts\eval_corpus.py --semantic    # đánh giá cả ngữ nghĩa
.\.venv\Scripts\python.exe scripts\eval_corpus.py --semantic --no-span --no-windows   # phiên bản trước, cùng bộ truy vấn
$env:DAOVAN_INDEX_DIR = "data\index_vietnews"; .\scripts\run.ps1   # mở app trên kho 100k bài
```

Tài liệu kho có id `corpus:vietnews:<guid>`, không có tệp trong `data/library`, nên app không tự gỡ chúng khỏi index.

`eval_corpus.py` ghép mỗi tài liệu truy vấn từ văn bản ngoài kho cộng một đoạn lấy từ bài trong kho, theo 6 kiểu:

| Kiểu | Cách tạo |
|---|---|
| Chép nguyên văn | ~150 từ chép nguyên |
| Sửa nhẹ | Chép rồi sửa ~12% số từ |
| Sửa nặng | Chép, sửa ~30% số từ và xáo trộn câu |
| Hai nguồn | Hai đoạn từ hai bài khác nhau |
| Tóm tắt | Sapo do tòa soạn viết lại của bài nguồn |
| Âm tính | Bài không có trong kho |

Biết trước nguồn nên đo được tỷ lệ phát hiện, top-1 và báo nhầm. Tài liệu không phải nguồn vẫn được tính là **trùng thật**, không phải báo nhầm, khi thỏa một trong hai điều kiện:
- chung ≥ 5% 5-gram với truy vấn;
- chung nguyên văn một câu ≥ 10 từ (≥ 3 cụm 8 từ, đếm đủ, không lấy mẫu, không bỏ văn mẫu).

Lý do: VietNews có nhiều tin cùng sự kiện đăng lại giữa các báo, và có cả bài trùng giữa train và validation. Điều kiện thứ hai mới thêm khi làm cửa sổ câu: soát tay cho thấy phần lớn "báo nhầm" trên truy vấn âm tính thực ra chép nguyên câu từ một bài trong kho. Các câu này bị bộ lọc văn mẫu của 5-gram bỏ qua vì lặp trong > 25 bài. Số của phiên bản trước dưới đây được **đo lại** bằng định nghĩa mới trên cùng bộ truy vấn (`--no-span --no-windows`).

Kết quả trên 99,134 bài / 365,140 đoạn, 360 truy vấn (60 mỗi kiểu). Ngưỡng của hai luật mới chỉnh trên seed 11. Bảng dưới là seed 23 (dữ liệu chưa thấy khi chỉnh), có ngữ nghĩa:

| Kiểu | Trước: phát hiện / top-1 | Nay: phát hiện / top-1 (top-1\*) | Báo nhầm (trước → nay) |
|---|---|---|---|
| Chép nguyên văn | 100% / 97% | 100% / 95% (100%) | 3% → 5% |
| Sửa nhẹ | 100% / 97% | 100% / 93% (100%) | 2% → 2% |
| Sửa nặng + xáo câu | **35%** / 27% | **72%** / 55% (78%) | 2% → 3% |
| Hai nguồn | 100% / 98% | 100% / 97% (100%) | 3% → 3% |
| Tóm tắt (sapo) | 2% / 2% | 2% / 2% | 2% → 3% |
| Âm tính | — | — | 0% → 0% |

Ghi chú bảng:
- top-1\*: bài đứng đầu là nguồn **hoặc** một bài trùng thật. Top-1 chặt giảm vài điểm vì luật mới thêm độ phủ cho các bài trùng nguyên văn thật với truy vấn: bản đăng lại của bài nguồn, hoặc bài trong kho trùng phần văn bản "của người viết" lấy từ validation.
- Chỉ từ vựng (luật 3-gram rải rác không cần model), seed 23: sửa nặng 25% → **57%**, báo nhầm 0% → 0–2%, các kiểu khác không đổi, p50 vẫn 8 ms.
- Seed 37: có ngữ nghĩa, sửa nặng 32% → **80%**; tóm tắt 5% → 5%; báo nhầm âm tính 3% → 3%, các kiểu khác ≤ 2%. Chỉ từ vựng, sửa nặng 20% → 58%.
- Seed 11 (dùng để chỉnh): sửa nặng 45% → 82%.
- Trong các ca sửa nặng bắt được, luật 3-gram rải rác góp nhiều nhất (33/60 ca ở seed 23), luật cửa sổ câu góp 27/60, nhiều ca được cả hai bắt.

Nhận xét:

- **Báo nhầm còn lại** chủ yếu là bài cùng khuôn mẫu (bản tin giá vàng, dự báo thời tiết) hoặc cùng một sự kiện do báo khác viết lại. Nhãn của bộ dữ liệu không phân biệt được trường hợp sau, nên tỷ lệ báo nhầm có ngữ nghĩa là mức trên.
  - Các báo nhầm mà luật mới thêm vào (seed 23/37) đều thuộc loại này. Ví dụ: cùng vụ ném chai trên cầu vượt km 89+340 cao tốc Hà Nội – Hải Phòng; cùng quyết định lập tổ liên ngành ở quận 1; cùng chỉ đạo chấn chỉnh cấp phép biểu diễn.
  - Đây đúng là kiểu diễn đạt lại mà luật cửa sổ câu nhắm tới.
- **Ngữ nghĩa cần ngưỡng thích ứng.** Với e5-small trên 365k đoạn tin tức, đoạn không liên quan gần nhất đã đạt cosine 0.93 (trung vị) đến 0.95 (p90): tin khác sự kiện nhưng cùng thể loại, như hai vụ trộm chó ở hai tỉnh. Cùng một sự kiện do hai báo viết đạt khoảng 0.965. Ngưỡng cố định 0.90 vì vậy đánh dấu nhầm gần như mọi truy vấn.
  - Ngưỡng hiện tại: cosine phải ≥ 0.90 **và** cao hơn "nền" của chính đoạn mới (trung bình láng giềng gần thứ 6–10) ít nhất 0.04 (`SEMANTIC_MARGIN`).
  - Kho nhỏ, nhiều chủ đề: nền thấp, ngưỡng giữ 0.90. Kho báo 100k bài: ngưỡng tự nâng lên khoảng 0.95.
- **Cửa sổ câu tự nó không đủ, phải kèm xác minh từ vựng.** Ở cấp cửa sổ ~60 từ, nguồn của ca sửa nặng đứng đầu danh sách láng giềng ở 100% ca; nguồn của ca tóm tắt đứng đầu ở ~45% ca và nằm trong top-40 ở ~90% ca.
  - Nhưng biên so với nền chỉ ~0.01–0.03, ngang tin cùng thể loại.
  - Seed 11, cộng với hệ trước (sửa nặng 45%): chỉ dùng ngưỡng biên ≥ 0.04 thì được 55%, báo nhầm 2.2%. Biên ≥ 0.03 kèm điều kiện ≥ 50% từ đặc trưng (theo IDF) của cửa sổ có trong đoạn kho thì được 60%, báo nhầm 1.1%. Thêm luật 3-gram rải rác thì lên 82%.
- **Tóm tắt (sapo) vẫn chưa bắt được** (2–5%, không đổi). Đã thử, ở mức báo nhầm ≤ 3%, đều không quá ~8%:
  - so cửa sổ 25/40/60 từ với vector đoạn;
  - so cửa sổ với vector trung bình của cả bài;
  - xác minh bằng TF-IDF, tỷ lệ từ chung, hay chỉ từ hiếm (tên riêng, số).

  Nguyên nhân: sapo là bản viết lại thật sự. Với bài nguồn, sapo chỉ chung 13–40% trọng số từ đặc trưng, cosine e5-small ~0.89–0.91 trong khi nền ~0.88–0.90. Cả hai mức này giống hệt một tin cùng thể loại. Muốn bắt kiểu này cần model embedding mạnh hơn, hoặc model phân loại cặp câu (cross-encoder/NLI) chấm lại top ứng viên. DeepX encode 365k đoạn trên CPU mất khoảng 12 ngày, không khả thi trên máy này.
- **Vector câu phía kho: đã ước lượng, chưa làm.** Thử trên 11,359 đoạn ứng viên (66,905 cửa sổ câu ~40 từ): e5-small encode ~115 cửa sổ/s.
  - Chi phí cho cả kho: ~2.15 triệu cửa sổ (cửa sổ bắt đầu ở mỗi câu), tức **~5 giờ** và **~1.65 GB RAM** vector float16. Cửa sổ không chồng lấn thì ~1.3 triệu cửa sổ: ~3 giờ, ~1 GB.
  - Lợi ích, đo với nền ước lượng trên tập ứng viên đó: so cửa sổ với cửa sổ nâng biên trung vị của nguồn sửa nặng từ 0.034 lên 0.060, nhưng tóm tắt gần như không đổi (0.011 → 0.013).
  - Hai luật đã chọn không tốn thêm RAM mà đã đưa sửa nặng lên 72–80%, trong khi máy thường thiếu RAM. Vì vậy chưa thêm lớp này.
- Bộ thử chỉ sửa từ ngẫu nhiên, không có bản viết lại bằng mô hình ngôn ngữ (máy không đủ RAM để chạy Qwen sinh văn bản cùng lúc). Tỷ lệ "sửa nặng" vì vậy là ước lượng thô cho đạo văn có diễn đạt lại.

### Hiệu năng (i7-14700, CPU, kho 99,134 bài / 365,140 đoạn)

| Hạng mục | Kết quả |
|---|---|
| Nạp kho từ parquet (`import_corpus.py`) | ~2 phút (~12 s / 10.000 bài) |
| Encode vector cả kho bằng e5-small | 148 phút (~41 đoạn/s), một lần, tiếp tục được |
| Mở index khi khởi động app | 0.4–3 s; RAM 308 MB (từ vựng) + 267 MB (vector float16) |
| Đĩa | SQLite 410 MB, segment 301 MB, vector 271 MB |
| Truy vấn chỉ từ vựng (~2.7 đoạn) | p50 8 ms, p95 14 ms |
| Truy vấn có ngữ nghĩa (~2.7 đoạn) | p50 240 ms, p95 303 ms (trước: 106 / 123 ms). Trong đó encode 148 ms (đoạn + ~6 cửa sổ câu/đoạn, trước 67 ms), quét 365k vector 21 ms, cửa sổ câu 56 ms (quét + xác minh từ vựng) |
| Kiểm tra trong app (tệp 385 từ, model đã nạp) | 0.12 s trước khi thêm cửa sổ câu; nay ước ~0.25 s theo số đo trên, chưa đo lại trong app |

Những thay đổi giữ thời gian truy vấn gần như không tăng theo kích thước kho:
- chỉ tra ứng viên qua index ngược (tìm nhị phân trên mảng hash đã sắp xếp);
- bỏ cụm từ quá phổ biến;
- chấm TF-IDF chỉ trên tối đa 30 ứng viên mỗi đoạn;
- nhân ma trận float16 bằng PyTorch. numpy phải đổi sang float32, mất ~0.7 s mỗi truy vấn ở 365k vector.

Phần tốn nhất còn lại là encode tài liệu mới: ~55 ms/đoạn tính cả cửa sổ câu, trước là ~25 ms/đoạn. Tài liệu 60 đoạn (~9.000 từ) vì vậy mất thêm khoảng 2 s. Đoạn và cửa sổ được encode thành hai lượt riêng: chung một batch thì cửa sổ ngắn bị đệm tới độ dài đoạn và chậm hơn ~35%. Đã thử lượng tử hóa int8 cho e5-small: chỉ nhanh hơn 1.3 lần mà vector lệch (cosine 0.96–0.97 so với float32), đủ làm sai ngưỡng, nên không dùng.

Đo lại bằng:
- `scripts/eval_corpus.py`: kho thật;
- `scripts/bench_index.py`: kho giả lập, không cần model;
- `scripts/e2e_check.py`: văn bản tiếng Việt thật + e5-small.

### Vận hành

- Thêm tài liệu ở sidebar: tài liệu được đọc, chia đoạn và index ngay. Tệp có sẵn trong `data/library` nhưng chưa có trong index sẽ tự được index khi mở app.
- **Chế độ dự phòng từ vựng:** nếu model ngữ nghĩa không nạp được, app cảnh báo, cột Ngữ nghĩa để trống và vẫn chạy 5-gram/TF-IDF. Sau khi khắc phục, bấm "Thử nạp lại model" (tài liệu thiếu vector sẽ được bổ sung).
- **Giới hạn đã biết:**
  - Phía kho vẫn chỉ có vector đoạn 180 từ. Cửa sổ câu của tài liệu mới bù được một phần pha loãng, nhưng bản tóm tắt/diễn đạt lại sâu vẫn khó bắt (xem mục kho thử nghiệm).
  - Nếu cần độ nhạy cao hơn, có thể giảm `CHUNK_WORDS` trong `documents.py` rồi chạy `build_index.py --rebuild`. Đổi lại kho lớn hơn và encode lâu hơn.
- PDF scan không có lớp text được báo rõ cần OCR; hiện chưa tích hợp OCR. Giới hạn 25 MB/tệp.
- Không gọi API từ xa khi so sánh. Các thao tác tải model và pip cài package cần Internet.
- Windows Application Control có lúc chặn DLL của scipy/torch ("An Application Control policy has blocked this file"); chạy lại thường hết. Không tắt chính sách bảo mật để né lỗi này.

## Ghi chú về model

- **intfloat/multilingual-e5-small** (MIT, revision `614241f6`): nạp bằng `embedder.py` (`FastEncoder`), mean pooling, tiền tố `query: ` cho cả hai phía vì so sánh đối xứng, tối đa 512 token.
- **Qwen/Qwen3-1.7B** là model sinh văn bản, không phải model embedding hay detector đạo văn. Bản tải gốc gồm weights BF16 khoảng 4.07 GB; để chạy FP32 trên CPU có thể cần vài GB RAM ngoài footprint của ứng dụng. Tính năng phân tích giải thích Qwen chưa được nối vào luồng so khớp.
- **dxtech-asia/deepx-embedding-v1** (không dùng mặc định vì quá chậm trên CPU; trên máy có GPU có thể dùng lại qua `indexer.LibraryIndex(vector_model="deepx-embedding-v1", coarse_dim=256)` — tìm 2 tầng Matryoshka) gồm checkpoint backbone khoảng 0.97 GB (fp16), token embedding khoảng 0.57 GB (bf16) và tokenizer/remap. Model được nạp bằng `semantic.py` (`SemanticEncoder`), chạy CPU float32, đầu ra là vector 1536 chiều đã chuẩn hóa L2. Wrapper `DeepXEmbed.from_pretrained` của upstream không dùng được trên máy này: nó ép `.half()` và `autocast(device_type="cuda")`, còn `AutoTokenizer` lỗi với transformers 4.57.1. `semantic.py` xử lý khác như sau:
  - **Cấu hình:** shape trong checkpoint khớp `DeepXConfig()` mặc định của vendor, không khớp `models/deepx-embedding-v1/config.json`. `load_state_dict` không thiếu và không thừa key nào (0 missing, 0 unexpected).
  - **RoPE/RoDE:** các tham số `rope_*` và `depth_rotary_dims` chỉ dùng ở nhánh softmax. Nhánh này bị tắt bằng `_alpha_override = 0.0` như upstream (checkpoint cũng cho thấy `path_mix_logit` chưa từng được huấn luyện), nên các tham số này không ảnh hưởng đầu ra.
  - **Nhánh GDN-2:** viết lại bằng PyTorch thuần theo chunk, cùng công thức với kernel FLA mà model dùng khi huấn luyện, và đã đối chiếu với bản tính tuần tự. Fallback tuần tự của upstream dùng gate theo kênh, khác lúc huấn luyện và chậm.
  - **LoRA theo vòng lặp:** toàn bộ bằng 0 trong checkpoint v1.0 nên được bỏ qua. Kết quả không đổi.
  - **Tokenizer:** nạp trực tiếp `tokenizer.json`, offline, không `trust_remote_code`, đệm phải. Token đã bị cắt khỏi vocab (`id_remap = -1`) được quy về `<unk>`.
  - **Chất lượng:** hai câu tiếng Việt gần nghĩa đạt cosine ≈ 0.69, câu không liên quan ≈ 0.0–0.11.
  - **Hiệu năng (i7-14700, 20 luồng):** xem mục "Hiệu năng DeepX trên CPU".
- Cả hai model weights, tài liệu cá nhân và môi trường Python đều nằm trong các thư mục bị `.gitignore` loại trừ.

## Hiệu năng DeepX trên CPU

Đo ngày 2026-10-19 trên i7-14700, PyTorch 2.14.1+cpu, 20 luồng. Mỗi đoạn 180 từ tiếng Việt ≈ 250 token.

| Cấu hình | Thời gian / đoạn |
|---|---|
| float32, batch 4 (mặc định), máy rảnh | ~2.8 s |
| float32, 1 đoạn, máy rảnh | ~3.5 s |
| float32, batch 4 / 8 / 16, máy thiếu RAM (đang chạy Photoshop, Chrome...) | 5.5 / 5.7 / 6.3 s |
| bfloat16, 1 đoạn | ~21.6 s — CPU không có lệnh bf16 nên chậm hơn, không dùng |

- **RAM:** model chiếm khoảng 3.5 GB sau khi nạp, đỉnh khoảng 3.7 GB. Nạp mất 4–8 giây.
- **Khi máy thiếu RAM:** Windows phân trang làm encode chậm gấp đôi. Nên đóng bớt ứng dụng nặng khi mã hóa thư viện lớn.

## Cấu trúc

- `app.py` — giao diện Streamlit.
- `documents.py` — đọc PDF/DOCX/TXT (TXT: UTF-8, UTF-16, Windows-1258), chia đoạn giữ số trang.
- `indexer.py` — index kho (SQLite + segment `.npy`) và truy vấn 3 lớp (`LibraryIndex`).
- `corpus.py` — đọc bộ báo VietNews trong `data/corpus`.
- `embedder.py` — e5-small trên CPU (`FastEncoder.encode(list[str]) -> (N, 384)`).
- `semantic.py` — DeepX Embedding trên CPU (`SemanticEncoder`, tùy chọn).
- `scripts/setup.ps1`, `run.ps1`, `download_models.ps1` — môi trường, chạy, tải model; `scripts/build_index.py` — index kho từ dòng lệnh; `scripts/bench_index.py`, `scripts/e2e_check.py` — đo và kiểm thử; `scripts/import_corpus.py`, `scripts/eval_corpus.py` — nạp và đánh giá trên kho báo 100k bài.
- `models/` — checkpoint cục bộ (không commit).
- `data/library/` — tài liệu nguồn; `data/index/` — index; `data/corpus/`, `data/index_vietnews/` — kho báo thử nghiệm (không commit).
- `vendor/deepx-embed/` — mã kiến trúc DeepX upstream, chỉ `semantic.py` dùng.
- `docs/kien-truc-he-thong.md` — mô hình kiến trúc hệ thống (thành phần, luồng, mô hình dữ liệu, thuộc tính chất lượng) kèm sơ đồ Mermaid.
