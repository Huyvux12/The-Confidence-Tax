# RB-Cal — tiến độ và phạm vi

File này là nguồn sự thật về việc đã làm / sẽ làm. Nhật ký quyết định chi tiết (có ngày, hash):
[data_gen/PROTOCOL_LOG.md](data_gen/PROTOCOL_LOG.md). Đăng ký trước: [PREREGISTRATION.md](PREREGISTRATION.md).

## Đã làm

### Giai đoạn 1 — Cài đặt 5 hướng M1–M5 (toy data)
- [x] Khung chung `calibration_extensions/` (numerics, data_adapter, source_splits, features,
      calibrators, source_objectives, evaluator, run_experiment, toy_data).
- [x] M1 (β-field minimax + chẩn đoán common descent), M2, M3 (top-gap), M4 (label budget), M5 (active certificate).
- [x] Test: 79 test pass (`.venv/Scripts/python -m pytest calibration_extensions/tests -q`).
- [x] Deep research 2 lượt về công trình liên quan.

### Giai đoạn 2 — Dữ liệu thật
- [x] 31 task trắc nghiệm, 15.037 câu, 2 thứ tự đáp án, 2 prompt (`data_gen/build_items.py`).
- [x] 5 model 2025–2026 (Qwen3.5-9B, Gemma-4-12B, Granite-4.2-8B, Ministral-3-8B, Olmo-3-7B),
      suy luận trên Colab A100 (~17,5 CU) → 300.740 dòng xác suất (`data_gen/raw/`).
- [x] Chia dữ liệu (`data_gen/make_splits.py`): 25 task nguồn (train/select/audit 60/20/20 theo case),
      3 task target dev (`data/real/dev`), 3 task target confirm niêm phong (`data/real/confirm_sealed`).
- [x] Dọn cache dataset HF trên máy local (không có model nào được tải về máy local).

### Giai đoạn 3 — Phương pháp RB-Cal và thí nghiệm
- [x] Go/no-go: M1 chặt bất khả thi với 25 nguồn xung đột → chuyển sang RB-Cal (ngân sách regret ε).
- [x] `rb_calibration.py`: RB-Cal hằng / RBF / top-gap, LOSO chọn cấu hình, cổng select, shrinkage.
- [x] Dev z=0 (đăng ký trước): cổng chặn ở chế độ đồng thuận → sửa thành ràng buộc UCB (z=1,645), có ghi log.
- [x] Dev đầy đủ 10 đơn vị (7 đơn vị fit trên Colab CPU): E1, E2, E3 pass.
- [x] Tối ưu tốc độ (kết quả giống từng bit) + `--reuse-selection` (kiểm chứng trên dev: mọi quyết định cổng/cấu hình khớp).
- [x] Chốt code (hash `b0e2f24c6e5d169d`) → **chạy confirm đúng 1 lần**: E1, E2, E3 pass, cả trong từng chế độ.

| Confirm (30 task×đơn vị) | ΔNLL | Gây hại (>0,01) | Tệ nhất |
|---|---|---|---|
| MRTS | −0,201 | 0 | +0,000 |
| Pooled TS | −0,257 | 7 | +0,045 |
| **RB-Cal-UCB-RBF ε=0,01** | **−0,230** | **0** | +0,006 |

## Phạm vi cố định còn lại (đã chốt 2026-11-01, không mở rộng thêm)

Cả hai mục là **phân tích phụ, sau confirm** — được ghi nhãn như vậy trong paper; không thay đổi claim chính.

### Mục 1 — Thêm baseline Thermometer và ATS
Dữ liệu chỉ có xác suất chữ cái (không có hidden state), nên cả hai được chuyển thể dùng đặc trưng xác suất
(entropy, top-1, margin → RBF J=8, σ=0,5 + hằng số); tên gọi ghi rõ hậu tố `-P`.
- [x] `THERMO_P`: theo Thermometer (Shen et al., ICML 2024) — mỗi dòng cho τ_i = softplus(θ·φ_i);
      nhiệt độ của một task = trung bình τ_i trên các input không nhãn của task đó; huấn luyện bằng
      NLL trung bình đều theo task nguồn; L2 = 1e-3.
- [x] `ATS_P`: theo Adaptive Temperature Scaling (Xie et al., 2024) — β_i = exp(θ·φ_i) theo từng dòng,
      loss "selective smoothing" (dòng đúng: NLL; dòng sai: kéo về phân phối đều; trọng số bằng nhau),
      gộp mọi nguồn; L2 = 1e-3.
- [x] Test gradient (sai phân hữu hạn) + test khởi tạo = identity (`tests/test_extra_baselines.py`; tổng 84 test pass).
- [x] Chạy trên target dev và confirm (fit chỉ trên nguồn), so với RB-Cal-UCB-RBF: ΔNLL, CI bootstrap
      ghép cặp, số lần gây hại, tệ nhất, thắng/thua.

### Mục 3 — Leave-one-task-out (LOTO) trên 25 task nguồn
- [x] Với mỗi đơn vị (10) × mỗi task nguồn t (25): fit trên 24 task còn lại (train; select cho cổng),
      đánh giá trên mọi dòng của t → 250 cặp target thay vì 30.
- [x] Phương pháp: IDENTITY, POOLED_TS, MRTS, SHRINKAGE_LOSO, RBCAL_UCB_CONST_e0.01,
      RBCAL_UCB_RBF_e0.01, THERMO_P, ATS_P.
- [x] Cấu hình RBF của RB-Cal lấy từ lượt chọn trên đủ 25 nguồn (chọn lại lồng nhau tốn ~250 giờ CPU);
      rò rỉ nhỏ này (t có mặt khi chọn 4 siêu tham số rời rạc) được ghi rõ.
- [x] Báo cáo: ΔNLL macro, số gây hại, tệ nhất, thắng/thua, CI bootstrap **theo task** (lấy mẫu lại 25 task),
      tách theo chế độ.

### Định nghĩa "xong"
- [x] Hai mục trên chạy hết, kết quả nằm ở `runs/extra_baselines/` và `runs/loto/`.
- [x] Kết quả (thắng, hòa, thua — báo đúng như số liệu) được ghi vào PROTOCOL_LOG.md và bảng ở cuối file này.
- [x] Test vẫn pass toàn bộ.

Sau khi xong: báo cáo theo đúng các mục trên. Không thêm việc mới trừ khi bạn yêu cầu.

## Kết quả mục 1 và 3

### Mục 1 — baseline mới (`runs/extra_baselines/report.json`)
Confirm, 30 task×đơn vị (ΔNLL macro / số gây hại):

| | Tất cả | Xung đột (9) | Đồng thuận (21) |
|---|---|---|---|
| Pooled TS | −0,257 / 7 | −0,023 / 4 | −0,357 / 3 |
| THERMO_P | −0,262 / 5 | −0,027 / 3 | −0,362 / 2 |
| ATS_P | −0,263 / 7 | −0,021 / 5 | −0,367 / 2 |
| **RB-Cal-UCB-RBF** | −0,230 / **0** | −0,027 / **0** | −0,318 / **0** |

- Trung bình: THERMO_P và ATS_P **tốt hơn** RB-Cal (chênh +0,031 và +0,033 nats, CI không chứa 0); thắng/thua theo cặp 15/15 và 16/14.
- An toàn: hai baseline gây hại 5 và 7 lần trên 30; RB-Cal 0 lần.
- Chế độ xung đột: RB-Cal ngang THERMO_P về trung bình (+0,0002, CI [−0,005; +0,006]) và ngang/nhỉnh hơn ATS_P (−0,006, CI chứa 0), với 0 lần gây hại so với 3 và 5.
- Dev cho cùng bức tranh (baseline mới hơn ~0,06 về trung bình; gây hại 1–2 so với 0).
- Kết luận cho paper: RB-Cal **không** thắng các baseline gộp về NLL trung bình; nó đổi ~10–12% mức lợi lấy việc không gây hại trên task mới.

### Mục 3 — LOTO (`runs/loto/report.json`)
250 cặp (10 đơn vị × 25 task nguồn); CI bootstrap lấy mẫu lại 25 task.

| | ΔNLL | Gây hại (>0,01) | Tệ nhất | ECE |
|---|---|---|---|---|
| MRTS | −0,451 | 0 | +0,001 | 0,146 |
| Pooled TS | −0,605 | 22 | +0,073 | 0,068 |
| Shrinkage LOSO | −0,306 | 2 | +0,033 | 0,150 |
| THERMO_P | −0,607 | 19 | +0,062 | 0,065 |
| ATS_P | −0,600 | 32 | +0,140 | 0,069 |
| RB-Cal-UCB hằng | −0,509 | 2 | +0,013 | 0,118 |
| **RB-Cal-UCB-RBF** | **−0,524** | **2** | **+0,011** | 0,106 |

RB-Cal-UCB-RBF trừ phương pháp khác (âm = RB-Cal tốt hơn), CI theo task, thắng/hòa/thua:
- vs MRTS −0,073 [−0,094; −0,054], 198/0/52 — tốt hơn về trung bình, nhưng gây hại 2 so với 0
  (cả hai ca chỉ vừa vượt ngưỡng: +0,0102 và +0,0108, đều Qwen·letter_colon).
- vs RB-Cal hằng −0,015 [−0,021; −0,011], 161/61/28 — lớp RBF có ích (E3 lặp lại).
- vs Pooled TS +0,080 [+0,054; +0,108]; vs THERMO_P +0,083; vs ATS_P +0,076 — thua về trung bình,
  đổi lại gây hại 2 so với 22 / 19 / 32 và tệ nhất +0,011 so với +0,07 / +0,06 / +0,14.
- Chế độ xung đột (75 cặp): RB-Cal −0,070 / gây hại 2; Pooled −0,102 / 15; THERMO_P −0,104 / 13;
  ATS_P −0,090 / 26; MRTS 0,000 / 0.

### Tổng kết sau mục 1 + 3 (dùng cho paper)
- Lặp lại được trên 250 target: RB-Cal-RBF > RB-Cal hằng; RB-Cal > MRTS về trung bình; RB-Cal gây hại
  ít hơn hẳn mọi phương pháp gộp (Pooled TS, THERMO_P, ATS_P) với khoảng 85–87% mức lợi của chúng.
- Không đúng: RB-Cal "không bao giờ gây hại" — LOTO có 2/250 ca vượt ngưỡng 0,01 (MRTS 0/250).
- Không đúng: RB-Cal tốt hơn các baseline gộp về NLL trung bình.

## Giai đoạn 4 — Viết paper (chốt 2026-11-01; venue để sau, định dạng trung tính dễ chuyển sang TMLR)
- [x] `scripts/make_paper_assets.py`: mọi bảng, hình, con số trong paper sinh tự động từ `runs/` (không gõ tay số liệu).
- [x] `paper/main.tex` (tiếng Anh): Introduction, Related work, Setup, Method, 2 mệnh đề + chứng minh,
      Experiments (dev → confirm → baseline mới → LOTO), Limitations, phụ lục giao thức (pre-registration + lệch kế hoạch).
- [x] Trích dẫn chỉ dùng tài liệu có thật; chỗ nào chưa chắc (vd. trích dẫn gốc của MRTS) để `TODO` cho bạn điền.
- [x] Xong khi: `paper/main.pdf` biên dịch không lỗi, mọi con số đến từ `paper/generated/`.

### Kết quả giai đoạn 4
- `paper/main.pdf`: 13 trang (9 trang nội dung + tài liệu tham khảo + phụ lục giao thức). Build lại: `sh paper/build.sh`.
- 3 việc còn lại sau bản nháp (cập nhật 2026-11-01):
  1. [x] MRTS: nguồn gốc là bản thảo CALPORT của bạn (repo private `Huyvux12/calport-calibration-transfer`,
     "Anonymous review draft, 23 October 2026"). Cài đặt của mình khớp định nghĩa (lưới log 241 điểm trên [0,25; 4],
     phá hòa về |log T| nhỏ nhất). CALPORT đã chứng minh phần vô hướng của Mệnh đề 1 → paper đã ghi công,
     chỉ nhận phần mở rộng convex hull + Mệnh đề 2, 3. Trích dẫn: khóa `calport2026` trong refs.bib.
  2. [x] Đã đối chiếu Thermometer (arXiv 2403.08819) và ATS (EMNLP 2024, PDF gốc); mô tả trong paper đã sửa theo bản gốc
     và nêu rõ phần chuyển thể khác ở đâu.
  3. [ ] (bạn) Tên tác giả, venue/template, và cách trích dẫn CALPORT khi nó công bố (đang để "Anonymous").

## Giai đoạn 5 — Rà soát (chốt 2026-11-01): tính mới + dữ kiện + tái lập; báo cáo rồi sửa lỗi rõ ràng
Báo cáo: `paper/RECHECK.md` (mọi phát hiện có nguồn). Lỗi rõ ràng sửa luôn; điểm cần quyết định thì hỏi bạn.
- [x] A. Tính mới: tìm paper/repo (arXiv, ACL, OpenReview, GitHub, HF) trùng ý tưởng: ngân sách regret theo từng nguồn,
      ràng buộc UCB, chẩn đoán bao lồi / xung đột gradient cho calibration, calibration nhiều nguồn an toàn cho LLM.
- [x] B. Dữ kiện: 5 model trên Hugging Face (id, ngày phát hành, kích thước, giấy phép), id dataset,
      từng mục refs.bib (tác giả, năm, venue), các con số gõ tay trong paper (không qua `generated/`).
- [x] C. Tái lập: tính lại độc lập các số chính từ dữ liệu thô (số dòng, hash, endpoint), chạy lại test + build,
      fit lại đầy đủ 1 đơn vị confirm KHÔNG dùng `--reuse-selection` rồi so với kết quả đã chấm,
      kiểm tra repo có đủ để người khác chạy lại (README, phụ thuộc, dữ liệu).
- [x] Xong khi: RECHECK.md đầy đủ, lỗi rõ ràng đã sửa, paper build lại sạch.

### Kết quả giai đoạn 5 (chi tiết: `paper/RECHECK.md`)
- Tính mới: không tìm thấy công trình trùng; thêm Gong et al. 2021, Choi et al. 2024 vào Related work.
- Dữ kiện: 3 lỗi trích dẫn + 2 con số gõ tay đã sửa; model/dataset đúng; revision + giấy phép đã ghi (`data_gen/revisions.json`).
- Tái lập: 18/18 số liệu tính lại khớp; fit lại từ đầu 1 đơn vị confirm khớp (33/33 cùng cấu hình/cổng, lệch ≤ 8e-12);
  thêm README, requirements, `--revision`. 84/84 test pass, paper 14 trang build sạch.
