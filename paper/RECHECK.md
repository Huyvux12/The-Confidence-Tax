# Rà soát bản thảo (2026-11-01)

Phạm vi (bạn chọn): **tính mới**, **dữ kiện trong paper**, **tái lập**. Cách xử lý: sửa luôn lỗi rõ ràng, liệt kê điểm cần bạn quyết định.

## Tóm tắt

| Hạng mục | Kết quả | Đã sửa |
|---|---|---|
| A. Tính mới | Không tìm thấy công trình trùng (ngân sách regret theo từng nguồn / chẩn đoán bao lồi cho calibration). Thiếu 2 công trình liên quan. | Thêm Gong et al. 2021, Choi et al. 2024 vào Related work |
| B. Model, dataset | 5 model, 14 dataset tồn tại; mô tả trong paper đúng | Ghi revision + giấy phép |
| B. Trích dẫn | 3 lỗi: tác giả Yu et al. 2022, tác giả Zhao et al. 2021, LTT đã đăng tạp chí | Sửa cả 3 |
| B. Số gõ tay | 1 sai: thời gian chọn cấu hình "1.5–4.7 CPU h" → thực tế 42–285 phút; ước tính LOTO lồng nhau "250 CPU h" bị đánh giá thấp | Sửa cả 2 |
| C. Tái lập số liệu | 18/18 kiểm tra độc lập khớp | — |
| C. Fit lại từ đầu 1 đơn vị confirm | 33/33 phương pháp: cùng cấu hình, cùng cổng; lệch ≤ 8e-12 | — |
| C. Repo | Thiếu README, thiếu thư viện trong requirements, revision model không được ghim | Thêm README, requirements, `data_gen/revisions.json`, `--revision` |

## A. Tính mới

Tìm trên web (arXiv, ACL Anthology, OpenReview, AAAI/ICCV), GitHub (`gh search repos`) và Hugging Face với các truy vấn: ngân sách
regret / ràng buộc theo từng nguồn cho calibration; calibration nhiều miền cho miền chưa thấy; temperature scaling minimax / group-DRO;
Seldonian / cận tin cậy cho calibration; hướng giảm chung (MGDA) trong calibration; calibration LLM chuyển task.

**Không tìm thấy** paper hoặc repo nào: (i) đặt ràng buộc regret cho *từng* nguồn khi hiệu chỉnh xác suất,
(ii) đặt ràng buộc đó lên cận tin cậy, hay (iii) dùng hình học bao lồi / MGDA làm chẩn đoán khi nào calibration an toàn có thể dịch chuyển.
Đây là bằng chứng vắng mặt, không phải bảo đảm.

Công trình gần nhất (đã có hoặc đã thêm vào paper):
- CALPORT (bản thảo của bạn): MRTS, mệnh đề dấu vô hướng — đã ghi công.
- Yu et al. 2022 (multi-domain TS), Thermometer, ATS — đã có.
- **Gong et al., ICCV 2021** — calibration cho domain generalization từ nhiều miền calibration — *đã thêm*.
- **Choi et al., AAAI 2024 (CTS)** — temperature scaling chỉ dùng miền nguồn cho miền chưa thấy — *đã thêm*.
- Không cần thêm: Kobalczyk & van der Schaar 2026 (đã có; không bàn ràng buộc theo nguồn), Jin et al. 2026 (calibration lúc huấn luyện, COLM 2026),
  CalArena 2026 (benchmark tabular/vision), các paper MGDA (đã trích bản gốc).

## B. Dữ kiện

**Model (Hugging Face API).** Cả 5 tồn tại, đều là bản post-train, Apache-2.0, 7,3–12,0 tỷ tham số, tạo 2025-10 → 2026-08
→ "open-weight instruction models released in 2025–2026", "7–12B" đúng.

**Dataset.** 14 repo tồn tại; commit cuối của tất cả đều trước ngày tạo dữ liệu (31/10).
Giấy phép: MIT (MMLU, MMLU-Pro, CommonsenseQA), CC BY-SA (ARC, BoolQ), Apache-2.0 (MedMCQA, TruthfulQA, AQuA-RAT),
**CC BY-NC 3.0 (SciQ — phi thương mại)**, không ghi (OpenBookQA, HellaSwag, WinoGrande, PIQA, LogiQA).

**Trích dẫn** (đối chiếu Semantic Scholar + Crossref; DBLP bị chặn từ máy này). 37/40 mục khớp. Đã sửa:
1. `yu2022multidomain`: "Ma, Yi" → **"Ma, Yi-An"**.
2. `zhao2021calibrate`: "Zhao, Zihao" → **"Zhao, Tony Z."**.
3. `angelopoulos2021ltt`: arXiv → **The Annals of Applied Statistics 19(2):1641–1662, 2025**.
Ghi chú (không sửa): Balanya et al. — bản arXiv 2022, bản tạp chí 2024 (giữ 2024); Sagawa et al. — ICLR 2020 (bản arXiv 2019).

**Số gõ tay trong paper** (không đi qua `paper/generated/`): kiểm bằng `scripts/recheck_numbers.py`.
Đúng: 15.037 câu; 31 task; 60.148 dòng/model; 2–11 lựa chọn; median letter mass 0,45 / 0,999; 37 cấu hình; 3 đơn vị xung đột;
"13–26 trên 75" (LOTO xung đột); refit 6–15 phút. Sai, đã sửa: thời gian chọn cấu hình đầy đủ (42–285 phút, không phải 1,5–4,7 giờ);
chi phí LOTO lồng nhau (vài trăm giờ CPU, không phải ~250).

## C. Tái lập

**C1. Tính lại độc lập** (`scripts/recheck_numbers.py`, không dùng `evaluator.py`; kết quả `runs/recheck/numbers.json`): 18/18 khớp —
số dòng, hash, dữ liệu nguồn dev = confirm (từng byte), macro ΔNLL / số gây hại / tệ nhất của 4 phương pháp chính trên confirm
(khớp đến 1e-9), E1 và CI bootstrap (CI tính lại bằng seed khác lệch < 0,0003), hash dự đoán.

**C2. Fit lại từ đầu 1 đơn vị confirm** (`scripts/recheck_full_refit.py`, Qwen3.5-9B · paren, chọn LOSO đầy đủ, không `--reuse-selection`):
chạy trên VM CPU Colab (25,6 phút; lượt đầu ở máy local bị lỗi hết RAM do máy đang chạy nhiều ứng dụng khác).
Cả 33 phương pháp: **cấu hình được chọn giống hệt, quyết định cổng giống hệt**, chế độ giống hệt; dự đoán target lệch tối đa **8,0e-12**
(do khác máy: lượt chấm confirm chạy ở local). Vậy việc dùng `--reuse-selection` cho confirm không làm thay đổi kết quả
(kết quả: `runs/recheck/full_refit.json`).

**C3. Repo.** Đã thêm `README.md` (cách chạy lại từng bước), bổ sung `requirements.txt` (matplotlib, datasets, huggingface_hub),
`data_gen/requirements-inference.txt`, `data_gen/revisions.json` (commit của 5 model + 14 dataset, kèm kiểm tra chưa đổi từ ngày chạy),
tùy chọn `--revision` cho `infer_mcq.py` và ghim tự động trong `run_all.py`. 84/84 test pass; paper build sạch (14 trang).

**Giới hạn còn lại (không sửa được):** phiên bản torch trên Colab lúc chạy inference không được ghi lại; suy luận GPU có thể lệch
ở các bit cuối giữa phần cứng/thư viện; chưa chạy lại inference.

## Cần bạn quyết định
1. Không có điểm nào bắt buộc. Nếu muốn công bố dữ liệu xác suất cho mục đích thương mại thì phải loại SciQ (CC BY-NC).
