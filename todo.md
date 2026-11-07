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

## Giai đoạn 6 — Gate mới để giữ cải thiện ngang pooled mà vẫn 0 hại (chốt 2026-11-06, không mở rộng thêm)
Lý do: bản fit RB-Cal-RBF (ε=0.01, z=0) TRƯỚC gate đã đạt ngang pooled với 0 hại trên cả dev (Δ −0.529 vs pooled −0.529)
lẫn confirm (NLL 0.679 vs 0.686). Gate hiện tại (max_w Δ̂_select ≤ ε, không tính nhiễu, 25 nguồn) từ chối phần lớn
→ mất cải thiện. Cần một gate có cơ sở thống kê, chọn trên dev, không chọn trên confirm.
Biến thể (khai báo trước khi xem kết quả): ứng viên {C0: z=0, C1: z=1.645 (hiện tại)} × gate {G-mean: hiện tại,
G-sig: chỉ từ chối khi ∃w: Δ̂_select,w − 1.645·SE_select,w > ε (vi phạm có ý nghĩa)} × fallback {identity, MRTS} = 8.
Quy tắc chọn trên dev: trong các biến thể có 0 unit hại (>0.01) trên dev, lấy cải thiện macro lớn nhất;
hòa → ưu tiên fallback identity, rồi G-mean, rồi C1.
- [x] 6.1 `scripts/gate_study.py`: fit lại cấu hình đã chọn (không tìm lại menu), lưu Δ̂_select,w + SE theo cụm, áp 8 biến thể.
- [x] 6.2 Chạy trên dev → bảng 8 biến thể; ghi lựa chọn đóng băng vào `data_gen/PROTOCOL_LOG.md` TRƯỚC khi chấm confirm.
- [x] 6.3 Áp biến thể đã đóng băng lên confirm và LOTO (250 job) → số liệu ghi là "v2, sau khi đã mở confirm; thứ cấp".
- [ ] 6.4 Thêm v2 vào paper (Bảng 5/6 + đoạn công bố) — KHÔNG thay phương pháp chính.
Xong khi: 6.1–6.4 xong, test pass, paper build sạch, báo cáo cả khi v2 không tốt hơn.
Ngoài phạm vi: bộ confirm mới (cần để v2 thành phương pháp chính) — quyết định riêng sau.
### Kết quả giai đoạn 6 (`runs/gate_study/`, chi tiết trong PROTOCOL_LOG 2026-11-06)
| Biến thể | Dev Δ / hại | Confirm Δ / hại / NLL | LOTO Δ / hại / worst |
|---|---|---|---|
| Chính hiện tại (C1, G-mean, identity) | −0.472 / 0 | −0.230 / 0 / 0.712 | −0.524 / 2 / +0.011 |
| v2 đóng băng (C0, G-sig, MRTS) | −0.512 / 0 | −0.249 / 0 / 0.693 | −0.572 / **7** / +0.044 |
| C1, G-sig (thấy sau, chưa đóng băng) | −0.498 / 0 | −0.251 / 0 / 0.692 | −0.560 / 2 / +0.011 |
| Pooled TS (so sánh) | −0.529 / 1 | −0.257 / 7 / 0.686 | −0.605 / 22 / +0.073 |
- v2 không được dùng: LOTO tăng từ 2 lên 7 unit hại. C1 + G-sig tốt hơn bản chính trên cả 3 tập với cùng số hại,
  nhưng chọn sau khi đã xem → cần bộ confirm mới. 6.4 (sửa paper) chờ bạn quyết vì kết quả đổi hướng viết.

## Giai đoạn 7 — Bộ confirm thứ hai (confirm2) cho biến thể v3 = C1 + G-sig (chốt 2026-11-06)
Tiền đăng ký: PROTOCOL_LOG 2026-11-06 (task, endpoint P1–P3, viết trước khi có dữ liệu).
- [x] 7.1 `data_gen/build_items_confirm2.py` → `data_gen/items_confirm2.jsonl` + revision dataset.
- [x] 7.2 Inference 5 model trên Modal A100 (acc3) → `data_gen/raw_confirm2/`.
- [x] 7.3 `data/real/confirm2_sealed` (nguồn giữ nguyên từng byte) + SHA256.
- [x] 7.4 Chấm: run_rbcal --reuse-selection, gate_study (v3), Thermometer-P/ATS-P; endpoint P1–P3.
- [x] 7.5 Báo cáo + ghi PROTOCOL_LOG (kể cả khi thất bại).
Xong khi: 7.1–7.5 xong, không còn app Modal nào chạy.
### Kết quả giai đoạn 7 (confirm2, 60 unit; `runs/rbcal_confirm2/confirm2_report.json`)
| Phương pháp | Δ | hại | worst | NLL | Brier | ECE |
|---|---|---|---|---|---|---|
| Identity | 0 | 0 | 0 | 0.941 | 0.355 | 0.136 |
| Pooled TS | −0.319 | 9 | +0.051 | 0.622 | 0.314 | 0.071 |
| Thermometer-P | −0.323 | 7 | +0.031 | 0.619 | 0.313 | 0.067 |
| ATS-P | −0.315 | 12 | +0.068 | 0.626 | 0.315 | 0.072 |
| MRTS | −0.266 | 0 | 0 | 0.676 | 0.326 | 0.088 |
| v1 (bản chính cũ) | −0.296 | 0 | +0.009 | 0.645 | 0.318 | 0.073 |
| **v3 (C1 + G-sig)** | **−0.310** | **0** | +0.009 | **0.631** | **0.315** | **0.068** |
P1, P2, P3 đều PASS. Bỏ Gemma (post hoc): v3 −0.256 / 0 hại, pooled −0.253 / 9 hại.

## Giai đoạn 8 — Đặc trưng nhất quán theo thứ tự đáp án (v4), chốt 2026-11-06, không mở rộng thêm
Lý do: trên source_train, câu nhất quán khi đảo đáp án ở arc_easy/openbookqa/hellaswag đã chuẩn (gap +0.007..+0.022),
câu không nhất quán lệch nặng (+0.21..+0.25); chính các nguồn này chặn budget của v3.
Biến thể (khai báo trước): v4a = v3 + đặc trưng theo câu tv (TV giữa 2 thứ tự đáp án, quy về nhãn ngữ nghĩa);
v4b = v4a + đặc trưng theo task: tv trung bình của batch (nguồn: theo task nguồn; target: theo task target, không nhãn).
Cả hai: ứng viên C1 (z=1.645, eps=0.01), menu RBF như cũ trên đặc trưng mở rộng, chọn bằng LOSO, gate G-sig, fallback identity.
Quy tắc chọn trên dev: 0 unit hại, rồi cải thiện lớn nhất; hòa -> v4a. Cổng LOTO: v4 chỉ đi tiếp confirm3 nếu
LOTO hại <= 2 (v3) VÀ cải thiện LOTO >= v3 (-0.5599). Không đạt -> dừng, báo cáo.
- [x] 8.1 Code đặc trưng + test (tv ghép cặp theo case, bất biến hoán vị, cận beta vẫn đúng).
- [x] 8.2 Dev: chọn cấu hình LOSO (Modal CPU, acc3) cho v4a, v4b; bảng dev; chốt vào PROTOCOL_LOG.
- [x] 8.3 LOTO 250 job cho biến thể đã chốt; áp cổng LOTO.
- [x] 8.4 (nếu qua cổng) Tiền đăng ký confirm3 -> build 5-6 task mới -> inference Modal A100 -> chấm.
- [x] 8.5 Báo cáo + PROTOCOL_LOG, kể cả khi thất bại; không còn app Modal nào chạy.
### Kết quả giai đoạn 8
| Phương pháp | Dev Δ/hại | LOTO Δ/hại | Confirm3 Δ/hại | Confirm3 NLL/Brier/ECE | ECE giảm (C3) |
|---|---|---|---|---|---|
| Pooled TS | −0.529/1 | −0.605/22 | −0.305/7 | 0.492/0.257/0.051 | 45/60 |
| Thermometer-P | — | −0.607/19 | −0.306/5 | 0.490/0.256/0.049 | 45/60 |
| ATS-P | — | −0.600/32 | −0.304/9 | 0.492/0.257/0.052 | 43/60 |
| v3 | −0.498/0 | −0.560/2 | −0.279/0 | 0.518/0.262/0.062 | 50/60 |
| **v4b** | −0.505/0 | −0.573/**0** | −0.285/0 | 0.512/0.261/0.061 | 47/60 |
Q1–Q3 PASS. v4b nhích hơn v3 (+0.006 nats, CI loại 0; LOTO 0/250 hại) nhưng chưa xóa trade-off với SOTA.

## Giai đoạn 9 — Điểm vận hành "nới" (v5): ngang SOTA về cải thiện, ít hại hơn hẳn (chốt 2026-11-06)
Biến thể (khai báo trước): đặc trưng v4b (rbf_oct), KHÔNG margin (z=0), gate G-sig với ngưỡng = ε, fallback identity;
v5a: ε=0.01, v5b: ε=0.02. Chọn cấu hình bằng LOSO trên dev (ngưỡng xếp hạng 0.01 như cũ).
Quy tắc chốt: biến thể đủ điều kiện nếu dev hại <= 1/30 VÀ LOTO hại <= 7/250 (2.8%); trong đó lấy cải thiện LOTO lớn nhất;
hòa -> ε nhỏ hơn. Không biến thể nào đủ -> dừng, báo cáo.
- [x] 9.1 Code: ε/z theo biến thể trong v4_study, ngưỡng gate = ε.
- [x] 9.2 LOSO + fit dev + LOTO cho v5a, v5b (Modal CPU, acc3) -> chốt theo quy tắc.
- [-] 9.3 (bỏ: v5a thua v4b trên dev và LOTO, xem PROTOCOL_LOG) Tiền đăng ký confirm4 (6 task mới) -> build -> inference Modal A100 -> chấm.
- [x] 9.4 Đường đánh đổi đầy đủ trên confirm2/confirm3 (mọi biến thể, phân tích phụ) + báo cáo + PROTOCOL_LOG.

## Giai đoạn 10 — Theo báo cáo `reports/Hướng hiệu chỉnh chuyển giao mới.md` (chốt 2026-11-06)
- [x] 10.0 Chẩn đoán (local CPU): cận oracle của bộ chọn trên LOTO; AUROC của tín hiệu cấp task (tv trung bình,
      TV giữa 2 template, đồng thuận peer, entropy prior vị trí kiểu PriDe, entropy trung bình) cho 2 nhãn:
      "pooled hại" (Δ_pooled > 0.01) và "pooled hơn v4b > 0.01". ĐI TIẾP nếu oracle hơn v4b >= 0.01 nats
      VÀ có tín hiệu AUROC >= 0.75; không thì dừng và báo cáo.
- [x] 10.1 Hướng 3: base = trung bình xác suất 2 thứ tự; chạy lại identity/pooled/Thermo-P/ATS-P/v4b trên dev + LOTO.
- [x] 10.2 (cổng LOTO KHÔNG đạt) Hướng 1: v6 (peer feature) trên base đã chốt; LOSO dev -> LOTO; nhánh công bằng (baseline + peer); biến thể 1 peer.
      Cổng LOTO khai báo trước: hại <= 2/250 VÀ lợi <= -0.588.
- [-] 10.3 (dừng theo khai báo) Hướng 2: bộ chọn + LTT, frontier lợi-hại.
- [-] 10.4 (dừng theo khai báo) Tiền đăng ký confirm4 (8-12 task mới) -> inference Modal A100 -> chấm.
- [x] 10.5 Báo cáo + PROTOCOL_LOG; dừng ở bất kỳ cổng nào không đạt.
### Kết quả giai đoạn 10 (LOTO 250 unit)
| | Base gốc (1 thứ tự): Δ / hại | Base trung bình 2 thứ tự: Δ so với identity cùng base / hại | Δ so với identity gốc |
|---|---|---|---|
| Identity | 0 / 0 | 0 / 0 | −0.290 |
| Pooled TS | −0.605 / 22 | −0.362 / 22 | −0.652 |
| Thermometer-P (+peer) | −0.607 / 19 (−0.607 / 18) | −0.364 / 17 (−0.364 / 18) | −0.654 |
| ATS-P (+peer) | −0.600 / 32 (−0.636 / 19) | −0.351 / 44 (−0.382 / 29) | −0.641 |
| v4b | −0.573 / 0 | −0.298 / 0 | −0.587 |
| **v6 (peer)** | −0.583 / 1 | −0.316 / 0 | **−0.605 / 0 hại** |
Cổng v6 không đạt -> dừng trước confirm4 (xem PROTOCOL_LOG).

## Giai đoạn 11 — RB-Switch (chuyển theo task giữa pooled TS và v4b), chốt 2026-11-06
- [x] 11.0 Kiểm tra khả thi trên LOTO (lồng nhau): -0.607 / 2 hại / ECE 0.063 (pooled -0.605/22/0.068).
- [x] 11.1 Chốt quy tắc + tiền đăng ký confirm4 (10 task, S1-S3) trong PROTOCOL_LOG.
- [x] 11.2 Build items confirm4 -> inference Modal A100 (acc3) -> data/real/confirm4_sealed.
- [x] 11.3 (S1, S2 KHÔNG đạt; S3 đạt) Fit (pooled TS, Thermo-P, ATS-P, MRTS, v4b) -> tín hiệu target không nhãn -> áp quy tắc -> chấm S1-S3 + biểu đồ ECE.
- [x] 11.4 Báo cáo + PROTOCOL_LOG; không còn app Modal nào chạy.

## Giai đoạn 12 — Thành phần EC cho EvoLearn 2027 (hạn 16/11/2026)
- [x] 12.0 CMA-ES solver (evo_calibration.py) + khai báo PROTOCOL_LOG.
- [x] 12.1 Front: CMA-ES vs SLSQP theo eps, mặt Pareto (10 unit dev).
- [x] 12.2 (quy tắc đạt: Evo-RB tốt hơn v4b) Evo-RB (tiến hóa tâm/độ rộng RBF): dev, LOTO 250, confirm1-4; áp quy tắc quyết định.
- [x] 12.3 Paper LNCS 14 trang: thêm phần EC + bảng 500 unit + roster confirm4, ẩn danh, khai báo AI.
