# MRTS — Đặc tả 5 hướng phát triển phương pháp dành cho nhóm code

Ngày: 23/10/2026. Phiên bản đặc tả: 1.0.

**Đối tượng đọc:** người hoặc AI đã có code và dữ liệu của dự án, cần triển khai thêm phương pháp. Tài liệu tập trung vào cơ chế mới, input/output, thuật toán và phép thử; không tổng hợp lại kết quả cũ.

**Trạng thái:** đây là các đề xuất nghiên cứu và đặc tả triển khai. Chưa có kết quả benchmark cho năm phương pháp này. Các tên `M1`–`M5` là mã công việc nội bộ, không phải tên phương pháp đã được công bố. Không tự đưa chúng vào paper như đóng góp đã được xác nhận.

**Phạm vi bàn giao:** triển khai thành năm nhánh có thể bật/tắt độc lập. Không ghép cả năm thành một hệ thống trước khi biết từng cơ chế có tác dụng. Phương pháp chính là M1; M2–M5 là bốn phương án dự phòng với nguyên nhân tạo lợi ích khác nhau.

## Mục lục

1. [Bản đồ năm hướng và thứ tự triển khai](#1-bản-đồ-năm-hướng-và-thứ-tự-triển-khai)
2. [Quy ước toán học và hợp đồng dữ liệu chung](#2-quy-ước-toán-học-và-hợp-đồng-dữ-liệu-chung)
3. [M1 — Hiệu chỉnh hàm có chẩn đoán xung đột nguồn](#3-m1--hiệu-chỉnh-hàm-có-chẩn-đoán-xung-đột-nguồn)
4. [M2 — Dự báo lợi ích hiệu chỉnh trên target không nhãn](#4-m2--dự-báo-lợi-ích-hiệu-chỉnh-trên-target-không-nhãn)
5. [M3 — Tách hiệu chỉnh khoảng cách đáp án đầu và phân phối còn lại](#5-m3--tách-hiệu-chỉnh-khoảng-cách-đáp-án-đầu-và-phân-phối-còn-lại)
6. [M4 — Minimax thay đổi NLL dưới ngân sách nhiễu nhãn nguồn](#6-m4--minimax-thay-đổi-nll-dưới-ngân-sách-nhiễu-nhãn-nguồn)
7. [M5 — Chủ động hỏi nhãn để xác nhận lợi ích trên một batch](#7-m5--chủ-động-hỏi-nhãn-để-xác-nhận-lợi-ích-trên-một-batch)
8. [Giao diện code và các tệp phải xuất](#8-giao-diện-code-và-các-tệp-phải-xuất)
9. [Giao thức đánh giá và quyết định tiếp tục/dừng](#9-giao-thức-đánh-giá-và-quyết-định-tiếp-tụcdừng)
10. [Kiểm tra đúng/sai bắt buộc](#10-kiểm-tra-đúngsai-bắt-buộc)
11. [Nghiên cứu liên quan và ranh giới đóng góp](#11-nghiên-cứu-liên-quan-và-ranh-giới-đóng-góp)
12. [Yêu cầu giao việc có thể gửi nguyên văn cho bên code](#12-yêu-cầu-giao-việc-có-thể-gửi-nguyên-văn-cho-bên-code)

## 1. Bản đồ năm hướng và thứ tự triển khai

| Mã | Hướng | Giới hạn muốn giải quyết | Dữ liệu được dùng khi quyết định | Phép thử đầu tiên |
|---|---|---|---|---|
| M1 | Học hàm hiệu chỉnh từ xung đột nguồn | Nhiệt độ chung giữ nguyên dù có thể tồn tại cách sửa có điều kiện cùng có lợi | Nguồn có nhãn; xác suất từng mẫu lúc triển khai | Chẩn đoán hướng chung, sau đó fit một bộ hiệu chỉnh kernel ít thành phần |
| M2 | Dự báo thay đổi NLL trên target | Một quy tắc chọn từ nguồn không biết mức hiệu chỉnh nào hợp với batch mới | Nguồn có nhãn + batch target không nhãn | Ước lượng khoảng NLL ban đầu của target, suy ra khoảng lợi ích của từng nhiệt độ |
| M3 | Tách khoảng cách top-1/top-2 khỏi phân phối các đáp án còn lại | Một nhiệt độ buộc hai loại điều chỉnh phải dùng cùng độ mạnh | Nguồn có nhãn; xác suất từng mẫu lúc triển khai | Fit hai hệ số tuyến tính trong log-probability, bảo toàn thứ hạng |
| M4 | Tối ưu dưới các nhãn nguồn có thể bị sai | Một ít trạng thái có nhãn sai có thể làm lệch hướng hoặc cường độ hiệu chỉnh | Nguồn có nhãn quan sát và ngân sách nhiễu giả định | Tối ưu thay đổi NLL xấu nhất khi tối đa một số trạng thái bị đổi nhãn |
| M5 | Hỏi ít nhãn target để xác nhận lợi ích | Xác suất không nhãn có thể không đủ để quyết định an toàn | Nguồn + batch target không nhãn + oracle nhãn có ngân sách | Thu hẹp cận trên/dưới chính xác của thay đổi NLL bằng truy vấn nhãn |

**Ba giao thức phải tách riêng:**

- `source_only`: M1, M3, M4. Không lấy thống kê từ toàn bộ target để chọn tham số.
- `target_unlabeled`: M2. Được đọc xác suất của batch target, không được đọc nhãn target.
- `target_label_budget`: M5. Được đọc đúng những nhãn đã truy vấn; mọi nhãn còn lại bị che khỏi phương pháp.

Mọi hướng có thể thử bằng CPU từ probabilities hiện có. Riêng M5 cần nhãn target đáng tin cậy cho các truy vấn: benchmark có thể mô phỏng oracle bằng nhãn có sẵn; triển khai thực cần một nguồn nhãn thật. Tài liệu không yêu cầu inference lại LLM hoặc fine-tune LLM.

**Thứ tự ưu tiên:** dựng bộ khung chung → M1 bản ít chiều → M3 → M4 → M2 → M5. Có thể triển khai M3 trước phần kernel đầy đủ của M1 để kiểm tra nhanh bộ tối ưu hai tham số.

**Lý do các nhánh độc lập:** M1 sửa giới hạn của lớp hàm; M2 dùng thêm thông tin phân phối target; M3 thay đổi hình dạng phép biến đổi; M4 xử lý bất định của nhãn nguồn; M5 mua thêm thông tin bằng truy vấn nhãn. Một nhánh thất bại không tự động phủ định các nhánh còn lại.

## 2. Quy ước toán học và hợp đồng dữ liệu chung

### 2.1. Đối tượng cần hiệu chỉnh

Một mẫu có xác suất trên đúng các đáp án được phép:

\[
p=(p_1,\ldots,p_K),\qquad K\ge2,\qquad \sum_kp_k=1.
\]

Nhãn đúng là chỉ số `y` trong chính thứ tự đáp án của mẫu đó. Không giả định nhãn vị trí 0 có cùng ý nghĩa ngữ nghĩa giữa các tác vụ.

Nguồn thứ \(w\) có tập \(D_w\). Bộ hiệu chỉnh trả phân phối \(q_\theta(p)\). Định nghĩa:

\[
\ell_\theta(p,y)=-\log q_{\theta,y}(p),\quad
L_w(\theta)=\frac1{n_w}\sum_{i\in D_w}\ell_\theta(p_i,y_i),
\]
\[
\Delta_w(\theta)=L_w(\theta)-L_w(\theta_0).
\]

\(\theta_0\) là phép giữ nguyên xác suất. `delta_nll < 0` là cải thiện. `gain_nll = -delta_nll`; tuyệt đối không trộn hai quy ước dấu trong cùng output.

Chỉ số harm chính:

\[
\operatorname{harm}_{0.01}(d)=\mathbf1[\Delta_d>0.01].
\]

Xuất thêm `harm_any = 1[delta_nll > 1e-8]`. Điều kiện nguồn không tăng NLL và chỉ số harm target trên ngưỡng 0.01 là hai khái niệm khác nhau.

### 2.2. Chuẩn hóa xác suất và số học

1. Kiểm tra xác suất hữu hạn, không âm, tổng dương và số phần tử bằng số nhãn.
2. Chuẩn hóa tổng bằng 1.
3. Đặt `p_work = maximum(p, 1e-12)` rồi chuẩn hóa lại. Đây là phân phối tham chiếu chung cho mọi phương pháp trong phép thử mới.
4. Dùng `float64`, `log_p = log(p_work)` và `logsumexp`.
5. Tính NLL từ `log_q`, không lấy `log(exp(log_q))` và không chặn NLL riêng cho một phương pháp.
6. Không đưa các ô padding vào entropy, top-2 hoặc mẫu số softmax. Có thể dùng ragged arrays hoặc mask rõ ràng.
7. Lưu cả thông tin số mẫu có xác suất đầu vào bị chặn sàn. Không coi sàn số học là một thành phần học được.

Với nhiệt độ nghịch đảo \(\beta=1/T\):

\[
\log q_k=\beta\log p_k-\operatorname{LSE}_j(\beta\log p_j).
\]

Vùng mặc định cho phép thử so cùng miền nhiệt độ là \(\beta\in[0.25,4]\), tương đương \(T\in[0.25,4]\). Lưới rộng `[0.05,20]` là phép thử độ nhạy riêng, chạy cho mọi đối chứng phù hợp. Không mở lưới sau khi nhìn target chỉ cho phương pháp mới.

### 2.3. Input JSONL chuẩn

Adapter của bên code chuyển định dạng hiện có sang các tệp logic sau. Không cần đổi tên hoặc ghi đè dữ liệu gốc.

```text
source_train.jsonl
source_select.jsonl
source_audit.jsonl
target_inputs.jsonl
target_labels_private.jsonl
manifest.json
```

Ví dụ **minh họa schema**, không phải mẫu thực hay số đo của dự án:

```json
{"sample_id":"s_001_q1","case_id":"state_001","model_id":"model_A","prompt_id":"prompt_A","source_id":"workflow_A","task_id":"task_A","probs":[0.7,0.2,0.1],"label_ids":["A","B","C"],"y":1}
```

Trường bắt buộc:

| Trường | Kiểu | Ý nghĩa |
|---|---|---|
| `sample_id` | string | ID duy nhất của một dự đoán |
| `case_id` | string | Đơn vị gốc để chia tập và bootstrap; các câu từ cùng một trạng thái/text phải cùng nhóm |
| `model_id` | string | Checkpoint tạo xác suất |
| `prompt_id` | string | Prompt/định dạng tạo xác suất |
| `source_id` | string | Nhóm nguồn dùng trong mục tiêu minimax; target có thể để null |
| `task_id` | string | Tác vụ để đánh giá theo từng task |
| `probs` | list[float] | Xác suất trên các lựa chọn hợp lệ |
| `label_ids` | list[string] | Thứ tự lựa chọn; cần để ánh xạ nhãn khi thứ tự thay đổi |
| `y` | int | Chỉ xuất hiện trong nguồn có nhãn; không có trong input target |

`target_labels_private.jsonl` chứa `sample_id`, `case_id`, `y` và ánh xạ ngữ nghĩa cần thiết. Chỉ evaluator và oracle của M5 được mở tệp này.

Metadata tùy chọn: `label_provenance`, `parent_case_id`, `question_type`, `annotation_cost`, `split_id`. Nguồn do LLM gán nhãn phải ghi đúng nguồn gốc; không tự gọi là nhãn sạch.

### 2.4. Chia tập và chọn tham số

- Dùng các split hiện có nếu đã bảo đảm tách `case_id`; nếu cần split mới, tạo manifest mới trước khi chạy target.
- Tất cả câu, bản lặp, thứ tự đáp án và prompt của cùng case phải cùng phía khi chúng được coi là lặp của cùng ví dụ.
- `source_train`: fit tham số. Chọn cấu hình bằng leave-one-source-out bên trong tập này.
- `source_select`: kiểm tra quy tắc chấp nhận/fallback cuối cùng, đã cố định trước.
- `source_audit`: chỉ đo sau khi chốt. Không sửa phương pháp từ kết quả audit rồi vẫn gọi đó là audit độc lập.
- Không tự refit trên `source_select` sau khi dùng nó làm cổng chấp nhận. Muốn refit phải có giao thức và bộ kiểm tra độc lập mới.
- Mặc định fit một calibrator cho mỗi `model_id × prompt_id × bộ nguồn`. Không tự gộp các model để tăng cỡ mẫu.

Nguồn train của các đối chứng phải giống nhau. Cổng kiểm tra nguồn của phương pháp mới cũng cần đối chứng dùng cùng cổng để tách lợi ích của gate khỏi lợi ích của cơ chế.

### 2.5. Đặc trưng từ xác suất

Với xác suất đã sắp giảm dần \(p_{(1)}\ge p_{(2)}\):

\[
h(p)=H(p)/\log K,\quad c(p)=p_{(1)},\quad m(p)=p_{(1)}-p_{(2)}.
\]

Đặc trưng mặc định cho M1 là `z = [h, c, m]`, thuộc `[0,1]^3`. Chúng không phụ thuộc tên hay thứ tự nhãn. Không dùng `y`, correctness, loss hoặc nhãn target để tính đặc trưng lúc predict.

Khả năng tính cùng một vector đặc trưng khi K thay đổi không chứng minh quan hệ hiệu chỉnh bất biến theo K. Phải đánh giá chuyển K riêng; M2 bản đầu yêu cầu K khớp giữa nguồn và target.

## 3. M1 — Hiệu chỉnh hàm có chẩn đoán xung đột nguồn

### 3.1. Câu hỏi nghiên cứu và đóng góp kỳ vọng

**RQ-M1:** Có trường hợp nhiệt độ chung không có hướng làm giảm đồng thời NLL của mọi nguồn, nhưng một hàm nhiệt độ phụ thuộc xác suất có hướng như vậy và chuyển giao tốt hơn sang task mới hay không?

Phương pháp phải làm được hai việc:

1. Chẩn đoán giới hạn của lớp hàm đang xét.
2. Học hàm mở rộng với mức phức tạp nhỏ, rồi kiểm tra độc lập xem mở rộng có đáng dùng không.

Đóng góp kỳ vọng nằm ở vấn đề xung đột nguồn và cách lựa chọn lớp hàm; riêng kernel, gradient đa mục tiêu hoặc nhiệt độ theo mẫu đều đã có tiền lệ.

### 3.2. Input/output riêng

**Fit input:** các nguồn train có nhãn; nguồn select; seed; miền beta; danh sách số tâm kernel và bandwidth cố định.

**Predict input:** probabilities của từng mẫu, không cần biết nó thuộc nguồn nào.

**Output:**

- Calibrator `beta(p)` và `log_q(p)`.
- Cấu hình đã chọn: số tâm, bandwidth, regularization, hệ số.
- Chẩn đoán tại phép giữ nguyên cho từng lớp hàm.
- Quyết định chấp nhận hoặc fallback cùng lý do máy đọc được.
- Bảng hiệu quả nguồn train/select/audit và kết quả target do evaluator tạo.

### 3.3. Lớp hàm triển khai mặc định

Không bắt đầu bằng mạng neural hoặc kernel đầy đủ với một tâm cho mỗi mẫu. Dùng số tâm nhỏ để kiểm soát cỡ mẫu.

Chọn các tâm \(c_1,\ldots,c_J\) chỉ từ đặc trưng nguồn train. Tâm đầu là điểm gần trung bình đặc trưng nhất; các tâm tiếp theo dùng farthest-point traversal. Thứ tự phá hòa theo `sample_id`. Dùng cùng tập tâm cho mọi nguồn trong một calibrator.

\[
\psi_j(z)=\frac1{\sqrt J}\exp\left(-\frac{\|z-c_j\|_2^2}{2\sigma^2}\right),\quad J>0.
\]

Khi đó \(\|\psi(z)\|_2\le1\) với mọi z. Đặt:

\[
\beta_\theta(p)=b_0+v^\top\psi(z(p)).
\]

Ràng buộc toàn cục:

\[
\|v\|_2\le b_0-\beta_{\min},\qquad
\|v\|_2\le\beta_{\max}-b_0.
\]

Hai ràng buộc bảo đảm beta nằm trong miền cho phép với mọi input, không chỉ input train. Khi `J=0`, dùng `beta(p)=b0`; khi `b0=1,v=0`, giữ nguyên xác suất.

**Không dùng `clip(beta(p), ...)` trong objective rồi vẫn gọi bài toán lồi.** Đặc trưng phải cố định trước khi tối ưu hệ số; với lớp hàm và ràng buộc trên, NLL lồi theo `(b0,v)`.

Mỗi cấu hình J có chuẩn hóa riêng `1/sqrt(J)`. Không chuyển hệ số giữa hai cấu hình J mà giữ nguyên hệ số một cách máy móc, vì hàm sẽ thay đổi. Chạy fit độc lập hoặc chuyển đổi sao cho beta thực sự được giữ nguyên.

### 3.4. Chẩn đoán hình học tại beta = 1

Đặt \(\phi(p)=[1,\psi(z(p))]\). Tại phép giữ nguyên:

\[
r_i=-\log p_{i,y_i}-H(p_i),\qquad
g_w=\frac1{n_w}\sum_{i\in D_w}r_i\phi(p_i).
\]

`g_w` là gradient NLL của nguồn w theo `(b0,v)`. Tìm:

\[
\alpha^*=\arg\min_{\alpha\in\Delta_W}
\tfrac12\alpha^\top G\alpha,
\quad G_{uv}=g_u^\top g_v,
\quad \bar g=\sum_w\alpha_w^*g_w.
\]

`Delta_W` là simplex không âm, tổng bằng 1. Nếu \(\rho=\|\bar g\|_2>0\), lấy \(d=-\bar g/\rho\). Tại nghiệm chính xác:

\[
g_w^\top d\le-\rho<0\qquad\forall w.
\]

Vì phép giữ nguyên nằm trong miền khả thi, một bước đủ nhỏ theo hướng đó giảm NLL của tất cả nguồn train.

Nếu \(\rho\) bằng 0, không có hướng làm giảm **nghiêm ngặt đồng thời mọi nguồn** trong không gian đặc trưng hiện tại. Điều này không loại trừ hướng giữ nguyên một nguồn và cải thiện nguồn khác; cũng không phải chứng minh mọi bộ hiệu chỉnh đều vô ích.

**Phạm vi chẩn đoán:** hình học của loss nguồn quan sát và lớp hàm đã chọn. Không gắn nhãn `population_safe` hoặc `target_safe` cho kết quả này.

Ghi `rho`, `alpha`, mọi `g_w`, mọi `g_w_dot_d`, sai số KKT và gradient kiểm tra số. `rho` phụ thuộc chuẩn hóa đặc trưng; không dùng độ lớn rho để so mô hình khác thang đo như một metric hiệu quả.

### 3.5. Mở rộng lớp hàm

Menu cấu hình ban đầu:

```yaml
centers: [0, 1, 2, 4, 8]
bandwidth: [0.2, 0.5, 1.0]
field_l2: [0.0, 0.01, 0.1]
beta_min: 0.25
beta_max: 4.0
```

Với `centers=0`, bỏ bandwidth và field regularization trùng lặp.

Chạy chẩn đoán riêng cho các lớp sau để thấy cơ chế:

1. `constant`: chỉ nhiệt độ chung.
2. `entropy_affine`: beta = a + b h, với beta ở hai đầu h=0 và h=1 nằm trong miền cho phép.
3. `rbf_J`: lớp hàm hữu hạn ở trên.

Đây là menu được chỉ định trước, không phải thuật toán sinh ra đặc trưng ngữ nghĩa mới. Bản triển khai đầu dùng mở rộng số tâm hữu hạn. Sinh tâm theo witness/column generation chỉ là extension sau khi bản này có tín hiệu; không ghi rằng đã triển khai column generation nếu chỉ chọn từ menu.

Chọn cấu hình bằng LOSO trên `source_train`: lần lượt giữ lại một nguồn; fit trên các nguồn còn lại; đo thay đổi NLL trên nguồn giữ lại. Sắp xếp cấu hình theo `max heldout delta`, rồi `mean heldout delta`, rồi độ phức tạp nhỏ hơn. Dùng cùng cách chọn cho ablation cùng lớp hàm. Phá hòa số học ở mức `1e-8`: ưu tiên ít tâm hơn, norm hệ số field nhỏ hơn, rồi thứ tự cấu hình cố định. Không tối ưu bandwidth liên tục trên tập select.

### 3.6. Hai objective phải triển khai tách biệt

**M1-MINIMAX — phiên bản chính ban đầu:**

\[
\min_{b_0,v,t}\ t+\lambda\|v\|_2^2
\quad\text{s.t.}\quad
\Delta_w(b_0,v)\le t\ \forall w,
\quad (b_0,v)\text{ thỏa ràng buộc beta}.
\]

**M1-MEAN-SAFE — ablation mục tiêu:**

\[
\min_{b_0,v}\sum_w\pi_w L_w(b_0,v)+\lambda\|v\|_2^2
\quad\text{s.t.}\quad\Delta_w(b_0,v)\le0\ \forall w.
\]

Mặc định \(\pi_w=n_w/\sum_un_u\); thử trọng số đều chỉ dưới tên cấu hình khác. Với MINIMAX, nghiệm giữ nguyên có objective 0. Nghiệm tối ưu được chấp nhận phải có objective không lớn hơn 0 trong sai số solver; suy ra max thay đổi NLL train không dương vì penalty không âm.

Đối chứng bắt buộc `M1-POOLED`: cùng features, số tâm, ràng buộc và regularization, chỉ tối ưu NLL trung bình, không có ràng buộc các nguồn.

### 3.7. Solver và gradient

Có thể dùng CVXPY với `log_sum_exp` và ràng buộc norm, hoặc SciPy có gradient tường minh. Không yêu cầu GPU. Nếu SciPy không đạt residual quy định, xuất lỗi/invalid, không dùng nghiệm chưa khả thi.

Với \(\beta_i=b_0+v^\top\psi_i\):

\[
\frac{\partial\ell_i}{\partial\beta_i}
=\sum_k q_{ik}\log p_{ik}-\log p_{i,y_i},
\qquad
\nabla_\theta\ell_i=
\frac{\partial\ell_i}{\partial\beta_i}\phi_i.
\]

Fit nguồn baseline và loss của mọi candidate phải dùng cùng `log_p` và cùng trọng số. Khởi tạo bằng phép giữ nguyên; có thể thêm warm start nhiệt độ MRTS vô hướng. Chẩn đoán hướng chung là phép phân tích tại identity; không mặc nhiên áp dụng chứng minh hướng không ràng buộc tại điểm đang nằm trên biên.

QP simplex mặc định: điểm đầu alpha đều, Jacobian `G @ alpha`, tối đa 1000 vòng, tolerance objective khoảng `1e-12`. Kiểm tra `sum(alpha)`, dấu alpha và bất đẳng thức hướng. Không tự thêm ridge `G + lambda I` rồi dùng kết quả như nghiệm đúng của QP ban đầu; nếu cần ridge phải xuất biến thể và bỏ diễn giải certificate chính xác tương ứng.

Trong số học hữu hạn, không kết luận “chứng minh không tồn tại hướng” chỉ vì rho nhỏ. Đặt ngưỡng chẩn đoán `rho_tol = 1e-6 * max(1, max_w ||g_w||_2)`; chỉ công nhận hướng sử dụng được khi rho vượt ngưỡng, mọi `g_w_dot_d < -1e-8` và một bước nhỏ khả thi thực sự giảm loss của từng nguồn. Nếu chưa xác nhận được, xuất `NO_VERIFIED_COMMON_DESCENT` hoặc `NUMERICAL_UNRESOLVED`; định lý không-tồn-tại hướng ở mục 3.4 là phát biểu toán học cho rho=0 chính xác. Hướng chung đã kiểm tra vẫn có thể dùng dù QP chỉ đạt nghiệm gần đúng; phải tách kiểm tra dấu đạo hàm khỏi mức tối ưu của QP.

### 3.8. Cổng chấp nhận và fallback

Sau khi chọn hyperparameter và refit trên `source_train`, áp dụng lên `source_select`:

- Chấp nhận khi max thay đổi NLL của mọi nguồn select không vượt `epsilon_select`; mặc định 0 với tolerance số `1e-8`.
- Nếu không qua, giữ nguyên xác suất. Không chọn lại bandwidth từ kết quả select.
- Cổng này là kiểm tra thực nghiệm trên một tập hữu hạn, không phải bảo đảm thống kê cho target.
- Xuất kết quả cả trước và sau cổng. Chạy baseline cùng cổng để đo phần lợi ích riêng của gate.

### 3.9. Pseudocode

```python
def fit_m1(source_train, source_select, cfg):
    validate_sources(source_train, source_select)
    candidates = build_predeclared_candidates(cfg)
    loso_scores = []
    for candidate in candidates:
        fold_scores = []
        for heldout_source in source_ids(source_train):
            train_fold, validation_fold = leave_one_source_out(
                source_train, heldout_source
            )
            feature_map = fit_centers(train_fold, candidate)
            head = solve_m1_objective(train_fold, feature_map, candidate)
            fold_scores.append(evaluate_delta(validation_fold, head))
        loso_scores.append(aggregate_loso(candidate, fold_scores))
    selected = select_by_worst_then_mean_then_complexity(loso_scores)
    feature_map = fit_centers(source_train, selected)
    diagnostic = diagnose_at_identity(source_train, feature_map)
    fitted = solve_m1_objective(source_train, feature_map, selected)
    gate = evaluate_fixed_gate(source_select, fitted, cfg.epsilon_select)
    deployed = fitted if gate.accept else identity_calibrator()
    return deployed, fitted, diagnostic, gate, loso_scores
```

### 3.10. Phép thử cơ chế và tình huống thất bại

Bắt buộc có toy data tự dựng: (a) mọi nguồn cần cùng một nhiệt độ; (b) nguồn xung đột nhưng khác nhau theo entropy; (c) dạng sai lệch phi tuyến, affine entropy chưa đủ; (d) cùng phân phối xác suất nhưng quan hệ nhãn mâu thuẫn, kiểm tra giới hạn quan sát; (e) nguồn nhỏ với nhãn nhiễu để phát hiện học thuộc.

Trên dữ liệu thật, đo riêng các trường hợp `constant_has_common_descent=false` và hàm mở rộng có hướng chung. Sau đó đo nguồn audit và target; không dùng sự tồn tại của hướng trên train thay cho kết quả chuyển giao.

M1 có thể thất bại vì xác suất không chứa thông tin để phân biệt hai kiểu sai lệch; kernel fit nhiễu; quan hệ nguồn/target thay đổi; hoặc cổng từ chối quá nhiều. Output phải cho biết loại thất bại nào xuất hiện.

## 4. M2 — Dự báo lợi ích hiệu chỉnh trên target không nhãn

### 4.1. Ý tưởng và giao thức

Thay vì chọn một nhiệt độ rồi hy vọng chuyển giao tốt, ước lượng khoảng thay đổi NLL của từng nhiệt độ trên batch target. Đây là hướng dùng `target_unlabeled`, không gộp chung với kết quả `source_only` mà bỏ qua khác biệt thông tin.

**RQ-M2:** Có thể dự báo đúng hướng và mức lợi ích của hiệu chỉnh từ nguồn có nhãn và xác suất target không nhãn, đủ tốt để chọn nhiệt độ có đánh đổi tốt hơn hay không?

### 4.2. Đẳng thức chính xác giúp đơn giản hóa bài toán

Với beta **chung cho cả batch**, đặt \(s_i=-\log p_{i,y_i}\) và \(A_i(\beta)=\log\sum_kp_{ik}^{\beta}\):

\[
\ell_i(\beta)-\ell_i(1)=(\beta-1)s_i+A_i(\beta).
\]

Trên batch N mẫu:

\[
\Delta_T(\beta)=(\beta-1)M_T+\bar A_T(\beta),
\quad M_T=\frac1N\sum_i-\log p_{i,y_i}.
\]

Toàn bộ \(\bar A_T(\beta)\) tính được từ xác suất không nhãn. Đại lượng chưa biết chỉ là **NLL ban đầu trung bình của target**, \(M_T\).

Nếu dự báo được \(M_T\in[L_T,U_T]\), cận trên tương ứng là:

\[
B_T(\beta)=\bar A_T(\beta)+
\begin{cases}
(\beta-1)U_T,&\beta\ge1,\\
(\beta-1)L_T,&\beta<1.
\end{cases}
\]

Chọn beta trên lưới cố định để tối thiểu hóa B, luôn có beta=1. Khi khoảng bao phủ đúng M của chính batch này và `B <= 0`, thay đổi NLL thực cũng không dương. **Điểm khó là xây được khoảng bao phủ đáng tin dưới shift, không phải phép tìm beta.**

Không dùng công thức một M chung cho `beta(p)` thay đổi theo mẫu: khi đó cần mô hình các moment có trọng số/conditional khác, không thể tách beta ra khỏi kỳ vọng.

### 4.3. Input/output riêng

Input fit: nguồn có nhãn, source select, cấu hình bin và quy tắc dựng khoảng. Input adapt: `target_inputs` không y. Predict: beta đã chọn cho batch và probabilities của mẫu.

Output bổ sung:

```json
{"m_hat":1.2,"m_lower":0.8,"m_upper":1.6,"interval_type":"empirical_source_loso","guarantee_type":"conditional_on_interval_coverage","target_label_access":false,"unsupported_mass":0.0,"selected_beta":1.0}
```

Các số trên chỉ minh họa schema. Xuất `predicted_upper_delta` của mọi beta; evaluator thêm `true_m`, `interval_covered`, `true_delta` sau khi lựa chọn đã đóng băng.

### 4.4. Bộ dự báo khoảng bản đầu — quy định cụ thể

1. Chỉ thử trước trên nhóm nguồn/target có cùng K. Nếu K target chưa được hỗ trợ, trả khoảng thô và identity; không tự ngoại suy NLL giữa K khác nhau.
2. Chia normalized entropy thành `B in {2,4}` bin theo quantile nguồn train. Biên ngoài cùng là 0 và 1. Chọn B bằng LOSO nguồn, không bằng target.
3. Trong mỗi bin b của mỗi nguồn w, tính trung bình surprisal quan sát \(m_{wb}\). Số đơn vị để kiểm tra hỗ trợ là số `case_id` riêng, không phải số dòng.
4. Với tối thiểu 5 case nguồn trong bin, dùng cluster bootstrap nguồn để có dải bất định thực nghiệm của trung bình: lấy lại mẫu theo case, giữ toàn bộ dòng của case rồi tính mean surprisal của các dòng thuộc bin. Mặc định 500 draws, lấy percentile 2.5%/97.5%. Draw không có dòng trong bin là invalid; nếu dưới 80% draws hợp lệ, coi bin đó thiếu hỗ trợ. Log seed, số case và số draws hợp lệ.
5. Lấy envelope của các nguồn đủ hỗ trợ trong bin: lower nhỏ nhất, upper lớn nhất. Với mỗi nguồn w giữ lại trong LOSO, tính khoảng dự báo mean NLL của chính nguồn đó từ các nguồn còn lại, rồi lấy lỗi vượt khoảng `e_w = max(0, lower_w - true_M_w, true_M_w - upper_w)`. Chọn B bằng MAE của midpoint dự báo trong LOSO, phá hòa bằng lỗi lớn nhất rồi B nhỏ hơn. Sau khi chọn B, cố định inflation `c = max_w e_w`; mở khoảng dự báo batch cuối thành `[lower - c, upper + c]`. Đây là hiệu chỉnh kinh nghiệm trên nguồn, không phải mức coverage target được chứng minh. Không tính e_w từ target labels.
6. Với target, tính tỷ trọng mẫu ở các bin rồi cộng trung bình có trọng số các khoảng conditional. Đây dựa trên giả thuyết conditional mean surprisal target nằm trong envelope; phải ghi giả thuyết trong artifact.
7. Bin không đủ hỗ trợ dùng cận không nhãn trực tiếp của từng mẫu: \(-\log\max_kp_k\le s_i\le-\log\min_kp_k\). Phần này không giả định conditional invariance, nhưng có thể rất rộng.
8. Clamp khoảng dự báo cuối vào các cận trung bình thô vừa nêu chỉ khi giao không rỗng. Nếu giao rỗng, đánh dấu lỗi khoảng và fallback; không đảo hai đầu cho tiện.
9. Tính B trên mọi beta; chọn min B, phá hòa bằng gần beta=1 theo log. Nếu mọi phương án khác không vượt được identity theo B, giữ nguyên.

Để tránh trùng ký hiệu, trong code dùng `n_entropy_bins` cho số bin và `upper_delta` cho cận B_T(beta). Midpoint dự báo dùng cho metric MAE là `(m_lower + m_upper)/2`, sau inflation và giao với cận thô. Kiểm tra source_select như những batch mới, với interval estimator đã cố định: nếu bất kỳ khoảng nào bỏ sót mean NLL thực của batch nguồn select thì đánh dấu `source_interval_gate=false` và bản gated triển khai identity. Xuất riêng bản trước gate; không dùng source_select để tăng c rồi gọi đó là kiểm tra độc lập.

Cluster bootstrap và LOSO ở đây cho dải thực nghiệm, **không tạo ra bảo đảm hữu hạn mẫu cho target bất kỳ**. Cần đo coverage trên task giữ lại. Không gắn tên conformal hoặc distribution-free nếu chưa xây và chứng minh giao thức phù hợp.

### 4.5. Baseline, ablation và quyết định

Đối chứng: MRTS/pooled TS nguồn; cùng bộ dự báo nhưng dùng point estimate M; cận thô không nhãn; envelope không có bootstrap; phương pháp hiệu chỉnh dùng target không nhãn đã có nếu đáp ứng đúng input. Phương pháp tái triển khai khác paper phải khai báo là adaptation.

Metric cơ chế: MAE của dự báo M; coverage/độ rộng khoảng theo task; tỷ lệ dự đoán đúng dấu delta; tỷ lệ chấp nhận sai `predicted_upper_delta <= 0` nhưng `true_delta > 0`; tỷ lệ fallback và unsupported mass.

M2 không có ích nếu khoảng gần như luôn rộng đến mức chỉ chọn identity, hoặc khoảng hẹp nhưng bỏ sót M target thường xuyên. Giả định về shift là phần cần nghiên cứu; xác suất không nhãn không nhận diện được mọi thay đổi quan hệ nhãn.

## 5. M3 — Tách hiệu chỉnh khoảng cách đáp án đầu và phân phối còn lại

### 5.1. Ý tưởng khác M1

M1 thay độ mạnh của temperature theo từng mẫu nhưng mỗi mẫu vẫn dùng phép lũy thừa chung cho tất cả đáp án. M3 thay chính hình dạng phép biến đổi: tách khoảng cách giữa top-1 và top-2 khỏi độ dốc giữa các đáp án còn lại.

Ở đây “phần còn lại” là các đáp án ngoài top-1 của **cùng một mẫu**, không phải các lớp hiếm trong bài toán long-tailed classification.

### 5.2. Công thức hai tham số, bảo toàn toàn bộ thứ hạng

Đặt \(z_k=\log p_k\), \(k^*=\arg\max_kp_k\), \(d=z_{(1)}-z_{(2)}\ge0\). Với \(a>0,b>0\):

\[
u_k=a z_k+(b-a)d\,\mathbf1[k=k^*],
\qquad q_k=\operatorname{softmax}(u)_k.
\]

- Khoảng cách top-1/top-2 sau biến đổi bằng \(bd\).
- Khoảng cách giữa hai đáp án không phải top-1 được nhân a.
- `a=b` khôi phục temperature scaling với beta=a.
- `a=b=1` giữ nguyên xác suất.
- Vì a,b dương, toàn bộ thứ hạng được giữ nguyên. Khi top bị hòa, d=0 và phép biến đổi không tự phá hòa.
- Số tham số không tăng theo K.

**Trường hợp K=2:** phân phối chỉ phụ thuộc b; a không nhận diện được độc lập. Khi fit chỉ trên dữ liệu nhị phân, cố định a=1 và ghi `effective_dof=1`. Không gọi đây là lợi thế hai tham số ở binary. Khi dữ liệu có K>2 mới có thể học phần còn lại.

### 5.3. Vì sao có thể hữu ích?

Đặt \(C=\mathbf1[y=k^*]\), \(r_j=q_j/(1-q_{k^*})\) với j khác top-1. Có phân rã chính xác:

\[
-\log q_y=
-C\log q_{k^*}-(1-C)\log(1-q_{k^*})
-(1-C)\log r_y.
\]

Hai số hạng đầu là top-versus-all BCE; số cuối là loss phân phối đáp án còn lại trên các mẫu dự đoán sai. Một temperature buộc chúng thay đổi cùng nhau. M3 cho thêm khả năng điều chỉnh hai cấu trúc, với objective và đạo hàm vẫn đơn giản.

### 5.4. Objective và triển khai

Box mặc định `a,b in [0.25,4]`. Fit:

\[
\min_{a,b,t}\ t+\lambda(a-b)^2
\quad\text{s.t.}\quad\Delta_w(a,b)\le t\ \forall w.
\]

Chạy thêm `M3-POOLED` cùng tham số/box/penalty và `M3-MEAN-SAFE` với ràng buộc delta nguồn không dương. Chọn lambda `[0,0.01,0.1]` bằng nguồn LOSO. Dùng cùng cổng select như M1 và xuất cả trước/sau gate.

Các u tuyến tính theo a,b, vì z, top-index và d đã cố định từ p. NLL lồi theo a,b. Gradient:

\[
\partial_a\ell=\sum_kq_k(z_k-d\mathbf1[k=k^*])
-(z_y-d\mathbf1[y=k^*]),
\]
\[
\partial_b\ell=d(q_{k^*}-\mathbf1[y=k^*]).
\]

Không thay top-index bằng argmax sau hiệu chỉnh ở bên trong vòng tối ưu. Không cần sort toàn bộ: top-2 và mask top là đủ.

Input: nguồn train/select có nhãn, p từng mẫu khi predict. Output: `a`, `b`, `effective_dof`, log_q, quyết định gate, thống kê top-BCE và tail-NLL.

### 5.5. Đối chứng và metric riêng

Đối chứng bắt buộc: TS; bộ hiệu chỉnh entropy affine cùng số tham số; M3-POOLED; M3-MINIMAX; TvA-TS theo paper; ETS và phương pháp order-preserving phù hợp nếu có code chuẩn. Phải phân biệt mục tiêu top-BCE của TvA với NLL đa lớp của M3.

Ngoài NLL/Brier/ECE/harm, xuất:

- Top-versus-all BCE trên mọi mẫu.
- Tail conditional NLL trên mẫu dự đoán sai; nếu không có mẫu sai, ghi `null` và count=0.
- Hai đóng góp mean của phân rã NLL, tổng phải bằng NLL đa lớp.
- Tỷ lệ đổi argmax/thứ hạng phải bằng 0 trong tolerance.
- a,b, tỷ lệ chạm biên, và độ khác nhau của a,b.

Nếu a≈b và kết quả gần như TS, không có bằng chứng cần tách cấu trúc. Nếu chỉ ECE tốt hơn nhưng NLL/harm xấu đi, báo cáo đánh đổi; không đổi metric chính hậu nghiệm.

## 6. M4 — Minimax thay đổi NLL dưới ngân sách nhiễu nhãn nguồn

### 6.1. Ý tưởng

Đặt bất định trực tiếp lên nhãn nguồn, rồi tối ưu thay đổi loss so với giữ nguyên. Không suy đoán rằng nhãn đa số hoặc một LLM khác chắc chắn là nhãn thật. Ngân sách nhiễu là giả định nhạy cảm được khai báo, không phải mức nhiễu đã nhận diện từ dữ liệu.

**RQ-M4:** Tối ưu đối phó với một số trạng thái nguồn có thể bị gán nhãn sai có giúp giảm tác động của nhiễu lên quyết định hiệu chỉnh và kết quả target không?

### 6.2. Input/output

Input: nguồn có nhãn quan sát \(\tilde y\), `case_id` nhóm trạng thái, ngân sách \(\rho_w\) cho mỗi nguồn và lớp calibrator. Không cần nhiều annotator hoặc nhãn sạch để chạy optimizer.

Output: calibrator; `rho_assumed`; ngân sách số trạng thái; regret quan sát; phần tăng do đối thủ; các case đang chi phối cận xấu nhất; quyết định gate. Không xuất những case đối thủ chọn dưới tên “nhãn được phát hiện sai”.

### 6.3. Tập nhiễu và lời giải đối thủ chính xác

Với bất kỳ nhãn k:

\[
\delta_i(\theta;k)=-\log q_{\theta,k}(p_i)+\log p_{ik}.
\]

Nguồn w có S trạng thái. Cho phép đối thủ chọn tối đa
\(B_w=\min(S,\lceil\rho_w S\rceil)\) trạng thái, rồi thay nhãn của bất kỳ câu lựa chọn nào trong các trạng thái đó bằng một nhãn hợp lệ bất kỳ. Những trạng thái khác giữ nhãn quan sát.

Tính:

\[
R_w^{\rm obs}=\frac1{n_w}\sum_i\delta_i(\theta;\tilde y_i),
\qquad
A_s=\sum_{i\in s}\left[\max_k\delta_i(\theta;k)
-\delta_i(\theta;\tilde y_i)\right].
\]

Mỗi \(A_s\ge0\). Cận xấu nhất chính xác cho tập nhiễu đã định nghĩa là:

\[
R_w^{\rho}(\theta)=R_w^{\rm obs}
+\frac1{n_w}\sum_{s\in\operatorname{TopB}(A)}A_s.
\]

`B=0` cho tổng bổ sung bằng 0. `rho=0` khôi phục regret quan sát. Tại identity, mọi delta, A và R đều bằng 0.

Ngân sách ở đây tính theo **trạng thái**, không phải dòng. Nếu muốn budget theo dòng phải viết biến thể riêng. Không dùng 10% dòng như thể luôn tương đương 10% trạng thái.

### 6.4. Objective

\[
\min_{\theta,t}\ t+\lambda\Omega(\theta)
\quad\text{s.t.}\quad R_w^{\rho}(\theta)\le t\ \forall w.
\]

Hai lớp hàm cần thử:

- `M4-SCALAR`: beta chung, tìm trên cùng lưới với MRTS; bản rẻ và dễ xác minh.
- `M4-FIELD`: cùng lớp hàm hữu hạn, cùng norm bound và regularization như M1; tách lợi ích xử lý nhãn khỏi lợi ích tăng khả năng biểu diễn.

Penalty phải bằng 0 tại identity: bản scalar mặc định không penalty; bản field dùng `Omega=||v||_2^2`. Sau fit, bản gated tính lại đúng R_w^rho trên source_select với budget suy ra từ số state của source_select; chỉ chấp nhận nếu mọi R_w^rho không dương trong tolerance. Bản rho=0 và các baseline gate tương ứng phải dùng cùng quy tắc. Không thay robust gate bằng gate trên nhãn quan sát mà không ghi biến thể.

Menu rho giả định: `[0,0.05,0.10,0.20]`. Báo cáo cả đường nhạy cảm, không chọn mức tốt nhất từ target. Muốn một quy tắc chọn rho để triển khai phải định nghĩa nguồn thông tin và selector nguồn trước; nếu chưa có, ghi rõ đây là sensitivity analysis theo rho.

Với score softmax tuyến tính theo theta, bài toán đối thủ là cực đại hữu hạn của tổng các NLL cộng hằng số; phần robust vẫn lồi. Top-B có thể biểu diễn bằng epigraph/auxiliary variables hoặc tính trực tiếp cho scalar grid. Không dùng solver gradient trơn mà bỏ qua các điểm đổi thứ tự Top-B.

### 6.5. Bảo đảm có điều kiện và giới hạn

Nếu nhãn sạch thật của chính tập nguồn nằm trong ngân sách trạng thái đã giả định và \(R_w^\rho\le0\), thay đổi NLL trên tập đó với nhãn sạch không dương. Đây là bảo đảm theo tập bất định hữu hạn đã định nghĩa, không phải khôi phục nhãn sạch và không phải bảo đảm target.

Nếu nhãn thật nằm ngoài budget, hoặc noise đổi nghĩa tác vụ/lựa chọn ngoài label set, phát biểu trên không áp dụng. Một tỷ lệ bất đồng giữa hai lượt LLM không tự bằng tỷ lệ nhãn sai.

### 6.6. Thí nghiệm và đối chứng

Tạo noise có kiểm soát trên các nguồn có nhãn tham chiếu trong thí nghiệm; giữ nguyên probabilities và target labels. Chạy cùng seed, cùng case bị tác động cho mọi method.

- Noise theo state: chọn 5/10/20% state, đổi các nhãn choice trong state đó.
- Noise theo dòng: phép thử riêng; xuất cả tỷ lệ dòng và tỷ lệ state thực sự bị thay đổi.
- Ghi rõ nhãn trước khi thêm noise chỉ là nhãn tham chiếu nếu chưa được xác minh sạch.

Đối chứng: MRTS rho=0; shrinkage cùng miền; M1 hoặc M3 cùng capacity; TransTS khi tái lập được đúng giả định/phương pháp. Không thay TransTS bằng một phép smooth nhãn rồi gọi là kết quả paper.

Metric: thay đổi beta/hàm giữa các mức noise; tỷ lệ đổi quyết định giữ nguyên; harm và mean delta target; tỷ lệ fallback; độ rộng penalty robust; kết quả khi rho giả định thấp hơn/cao hơn corruption state thực tế.

M4 có thể quá bảo thủ và gần như luôn giữ nguyên. Harm bằng 0 với mức cải thiện cũng bằng 0 không đủ chứng minh cơ chế mới có giá trị.

## 7. M5 — Chủ động hỏi nhãn để xác nhận lợi ích trên một batch

### 7.1. Giao thức khác và giá trị kỳ vọng

M5 dùng một ngân sách nhãn target nhỏ. Đây là lựa chọn khi quan hệ xác suất–nhãn trên target không nhận diện được từ nguồn. Không gọi M5 là zero-shot/no-target-label và không so như thể nó dùng cùng thông tin với M1.

**RQ-M5:** Có thể dùng ít truy vấn nhãn hơn chọn ngẫu nhiên để xác nhận một calibrator cải thiện NLL của batch đang triển khai hay không?

Bản đầu dùng cận tất định cho chính batch hữu hạn, không cần giả định iid, nguồn–đích giống nhau hoặc độ đúng của một surrogate. Cận có thể rộng và cần nhiều nhãn; đó là điều phải đo.

### 7.2. Input/output

Input:

- Batch target p không nhãn.
- Danh sách calibrator ứng viên **đã fit và đóng băng từ nguồn**; luôn có identity.
- `label_oracle.query(case_id)` và ngân sách theo số case hoặc chi phí ghi rõ.
- ID batch và quy tắc truy vấn cố định.

Output:

- Candidate được chọn, tập case đã hỏi, tổng cost.
- Cận lower/upper của mỗi candidate sau từng query.
- `certified`, `certificate_scope="current_finite_batch"`.
- Lý do dừng: `CERTIFIED`, `BUDGET_EXHAUSTED`, `NO_POTENTIAL`, `ALL_LABELS_QUERIED` hoặc `ORACLE_FAILURE`.
- Predictions và metric trên toàn batch; metric riêng trên phần chưa truy vấn để thấy tác động của việc chọn có giám sát.

### 7.3. Cận chính xác từ các nhãn chưa biết

Với candidate a, tính trước trên tất cả nhãn hợp lệ:

\[
\delta_{ia}(k)=-\log q_{ia,k}+\log p_{ik},
\quad l_{ia}=\min_k\delta_{ia}(k),
\quad u_{ia}=\max_k\delta_{ia}(k).
\]

Gọi Q là tập dòng đã biết nhãn qua các case đã truy vấn. Khi đó:

\[
L_a(Q)=\frac1N\left[\sum_{i\in Q}\delta_{ia}(y_i)
+\sum_{i\notin Q}l_{ia}\right],
\]
\[
U_a(Q)=\frac1N\left[\sum_{i\in Q}\delta_{ia}(y_i)
+\sum_{i\notin Q}u_{ia}\right].
\]

Luôn có \(L_a(Q)\le\Delta_a\le U_a(Q)\) cho chính batch này, với nhãn query đúng và q được tính đúng. Các cận đúng đồng thời cho mọi candidate đã cho và mọi cách chọn query thích nghi.

Nếu \(U_a\le-\eta\), có cải thiện NLL trung bình ít nhất eta trên batch. Chọn mặc định `eta=1e-6` để tránh nhận sai số số học là cải thiện. Identity có cận 0 nhưng không được tính là “đã tìm được cải thiện”.

Đây không phải cận NLL của batch tương lai. Không diễn giải `certified=true` thành mọi task hoặc mọi mẫu đều được cải thiện.

### 7.4. Quy tắc hỏi nhãn mặc định

1. Tính cận cho toàn bộ candidates từ probabilities.
2. Nếu có candidate không phải identity đạt `U <= -eta`, dừng; chọn candidate có U thấp nhất.
3. Loại khỏi bước chọn query các candidate có `L >= -eta`: ngay cả gán nhãn có lợi nhất cũng chưa thể đạt mức cải thiện cần tìm.
4. Trong các candidate còn lại, chọn candidate a* có U nhỏ nhất; phá hòa theo ID ổn định.
5. Với mỗi case chưa hỏi, tính tổng `u_ia* - l_ia*` của các dòng thuộc case đó, chia cho annotation cost nếu cost khác nhau.
6. Query case có điểm cao nhất. Oracle trả nhãn cho đúng các dòng/label order của case; cập nhật bounds cho tất cả candidates.
7. Hết budget mà không có certificate cải thiện: trả identity, vẫn xuất candidate tốt nhất theo cận và lý do thất bại để phân tích.

Quy tắc chọn độ rộng lớn nhất là heuristic mặc định, không có chứng minh nó tối thiểu số nhãn. Cần so với query ngẫu nhiên và active testing phù hợp.

### 7.5. Candidate set và ranh giới truy cập nhãn

Candidate mặc định gồm identity, pooled TS, MRTS và các calibrator M1/M3/M4 đã fit từ nguồn; có thể thêm lưới temperature cố định nhỏ nếu đã khai báo. Không refit các candidate bằng nhãn query trong bản đầu.

Nếu cùng một text có nhiều thứ tự đáp án/prompt, oracle trả nhãn ngữ nghĩa rồi ánh xạ về từng dòng; một annotation của text tính một lần nhưng phải cập nhật mọi dòng hợp lệ liên quan. Chỉ evaluator giữ toàn bộ target y; method gọi oracle để nhận từng phần.

### 7.6. Đánh giá và trường hợp thất bại

Budget đề xuất: `0,8,16,32,64` case và các mốc `1%,5%,10%` batch khi kích thước cho phép. Không vượt tổng số case. Random-query baseline dùng tối thiểu 20 seed; không ngụ ý phương pháp tất định cần 20 seed nếu không có ngẫu nhiên.

Metric riêng: số nhãn đến certificate đầu; tỷ lệ batch được certified trong budget; guaranteed gain `-U`; true gain toàn batch; delta phần chưa query; runtime; tỷ lệ certificate bị sai phải bằng 0 trong tolerance khi oracle đúng.

M5 không có ích nếu cần hỏi gần hết nhãn hoặc phần lớn batch không chứng nhận được trong budget. Tuy vậy, không được làm hẹp cận bằng giả định surrogate không khai báo để tăng tỷ lệ certified.

## 8. Giao diện code và các tệp phải xuất

### 8.1. Cấu trúc module đề nghị

```text
calibration_extensions/
  data_adapter.py
  numerics.py
  source_splits.py
  features.py
  source_objectives.py
  m1_functional.py
  m2_target_gain.py
  m3_top_gap.py
  m4_label_budget.py
  m5_active_certificate.py
  evaluator.py
  run_experiment.py
  tests/
```

Tên thư mục là gợi ý tích hợp, không yêu cầu di chuyển code gốc. Giữ nguyên các baseline đang có và output cũ.

### 8.2. API tối thiểu

```python
fit(source_train, source_select, config) -> FitResult
predict_log_proba(fitted, probs_batch) -> list[np.ndarray]
diagnose(fitted, labelled_sources) -> dict
save_fitted(fitted, destination) -> None
load_fitted(destination) -> FittedCalibrator

# Chỉ M2:
adapt_unlabeled(fitted, target_inputs) -> AdaptedBatchCalibrator

# Chỉ M5:
select_with_oracle(candidates, target_inputs, oracle, budget) -> SelectionResult

# Chỉ evaluator được mở target_labels_private:
score(predictions, private_labels, protocol) -> EvaluationResult
```

`FitResult` tối thiểu chứa `method_id`, `protocol`, `parameters`, `feature_map`, `selected_hyperparameters`, `solver_status`, `source_gate`, `fallback_reason`, `diagnostic_scope`, `input_fingerprints`.

Trả identity khi fallback là trạng thái hợp lệ của phương pháp; lỗi optimizer là lỗi triển khai phải xuất riêng. Không gộp `SOLVER_FAILURE` vào `NO_COMMON_DESCENT`.

### 8.3. Các output bắt buộc cho mỗi run

| Tệp | Nội dung tối thiểu |
|---|---|
| `resolved_config.json` | Mọi tham số thực dùng, seed, protocol, nguồn dữ liệu, feature map, solver, quy tắc selection/gate |
| `fit_artifact.json` + `fit_arrays.npz` | Hệ số và các tâm/ma trận cần để predict lại mà không fit |
| `diagnostics.json` | Chẩn đoán nguồn; scope bảo đảm; lý do fallback/invalid |
| `source_metrics.csv` | Theo method/model/prompt/source/split: count case/dòng, NLL, delta, các metric liên quan |
| `predictions.jsonl` | ID, method, log_q, argmax, beta hoặc tham số biến đổi, trạng thái fallback; target predictions không chép y |
| `target_metrics.csv` | Evaluator tạo; mỗi dòng là method × model × prompt × task × seed |
| `paired_comparisons.csv` | Chênh lệch ghép cặp với các đối chứng bắt buộc, số đơn vị, khoảng bất định |
| `mechanism_metrics.csv` | Metric riêng của M1/M2/M3/M4/M5, không chỉ một bảng NLL |
| `runtime.json` | Fit time, predict time, CPU, peak memory, số mẫu, số task, số lần solve; GPU/inference LLM mới nếu có phải ghi thật |
| `checks.json` | Kết quả kiểm tra số học, leakage, tham số, thứ hạng và tính lặp lại |
| `decision.json` | PASS/FAIL/INVALID/NOT_RUN cho các câu hỏi tương ứng, lý do và đường dẫn evidence |

M2 thêm `target_gain_bounds.csv`; M4 thêm `adversary_budget.csv`; M5 thêm `query_trace.jsonl` và `batch_certificates.csv`.

### 8.4. Schema `target_metrics.csv`

```text
method_id,protocol,model_id,prompt_id,task_id,split_id,seed,
n_rows,n_cases,nll,brier,ece_15,accuracy,
delta_nll,gain_nll,harm_001,harm_any,
fallback,solver_valid,target_labels_used,label_query_cases,fit_seconds,predict_seconds
```

Thêm trường thì được; không đổi dấu hoặc đơn vị các trường cũ. `accuracy` dùng tỷ lệ `[0,1]` trong dữ liệu, chuyển sang % chỉ khi trình bày. NLL dùng log tự nhiên, đơn vị nats. Brier là tổng bình phương trên K lớp rồi lấy trung bình mẫu, không tự chia thêm K giữa các task.

ECE mặc định dùng 15 bin đều trên [0,1], confidence=max(q), correctness của argmax q, tổng có trọng số số mẫu của `abs(mean_confidence - mean_correctness)` trong từng bin. Chốt quy tắc đóng biên; bin rỗng đóng góp 0. Khi tổng hợp chính, tính metric theo task trước rồi macro-average các đơn vị model–prompt–task đã khai báo; xuất cả micro-average dưới tên riêng. Không thay đổi trọng số để một task lớn hoặc K nhỏ chi phối kết luận sau khi xem kết quả.

### 8.5. Ví dụ `decision.json`

```json
{
  "method_id": "M1_MINIMAX",
  "implementation": "PASS",
  "mechanism": "FAIL",
  "target_benefit": "FAIL",
  "reason": "Schema minh hoa: khong dien PASS chi vi fit loss giam",
  "claim_scope": "no_benchmark_claim_before_evaluation",
  "evidence_files": ["checks.json", "mechanism_metrics.csv", "paired_comparisons.csv"]
}
```

Ví dụ trên chỉ quy định cấu trúc, không phải đánh giá thực tế của M1.

## 9. Giao thức đánh giá và quyết định tiếp tục/dừng

### 9.1. Ba lớp quyết định

1. `implementation`: công thức, ràng buộc, cận, serialization và quyền đọc nhãn có đúng không?
2. `mechanism`: cơ chế đề xuất có xuất hiện trên dữ liệu giữ lại và khác đối chứng đúng capacity không?
3. `target_benefit`: khác biệt có ích trên target theo endpoint đã chốt hay không?

Một phương pháp có thể PASS triển khai nhưng FAIL cơ chế hoặc lợi ích. `NOT_RUN` dùng cho giao thức không đủ input, ví dụ M5 chưa có oracle; không đổi thành PASS/FAIL giả. Trong báo cáo phải trả lời trực tiếp phần đã chạy là đạt hay không đạt, kèm evidence.

### 9.2. Các nhóm đối chứng bắt buộc

- **Tham chiếu chung:** identity, pooled TS, MRTS, sign consensus cùng miền, heuristic nguồn gần một có kiểm tra dấu, shrinkage chọn từ nguồn.
- **Baseline paper:** HTS/ATS phù hợp, TvA-TS, ETS hoặc order-preserving theo từng nhánh; chỉ nhận là tái lập khi khớp giao thức và công thức. Nếu không đủ input thì ghi NOT_RUN + lý do, không giả lập bằng một công thức khác.
- **Cùng capacity:** M1-POOLED so M1-MINIMAX/MEAN-SAFE; M3-POOLED so M3-MINIMAX; M4-rho0 so rho>0 trong cùng lớp hàm.
- **Cùng thông tin target:** M2 cần đối chứng được xem cùng batch không nhãn; M5 cần random query và active query với cùng budget/candidate set/oracle.
- **Cùng gate:** có phiên bản baseline dùng cùng quy tắc kiểm tra nguồn khi phân tích lợi ích riêng của gate.

Không loại một baseline vì nó hòa hoặc thắng phương pháp mới. Đó là kết quả cần để xác định cơ chế có ích hay không.

### 9.3. Endpoint và tiêu chí có lợi ích

Trước khi chạy target, chọn đúng một endpoint chính cho mỗi giao thức và ghi vào config. Mặc định:

- So với MRTS cùng giao thức, muốn mean delta NLL giảm với khoảng bất định ghép cặp 95% không chứa 0, đồng thời harm không tăng trong roster đã cố định.
- So với đối chứng cùng capacity, cần có bằng chứng rằng cơ chế bảo vệ nguồn cải thiện phần rủi ro theo task hoặc đánh đổi mean–harm; chỉ thắng baseline ít tham số chưa đủ quy cho cơ chế minimax.
- Nếu hướng muốn chấp nhận đổi một ít mean để giảm harm, phải khai báo mức dung sai mean trước; không tạo một điểm số có trọng số mới sau khi nhìn bảng.
- Với M2, thêm endpoint dự báo coverage/false acceptance; với M5, thêm label budget và tỷ lệ certified.

Đây là gate nghiên cứu đề xuất để quyết định đầu tư, không phải tiêu chuẩn đảm bảo paper được nhận. Khi số đơn vị độc lập quá ít, không biến khoảng bootstrap hẹp do dùng sai đơn vị thành bằng chứng thắng ổn định.

### 9.4. Bất định và đơn vị phụ thuộc

- Bootstrap/hoán vị ghép cặp giữ nguyên cùng text/state giữa các phương pháp.
- Phép thử độ ổn định calibrator phải resample nguồn theo state và **fit lại**, gồm cả bước chọn cấu hình khi bước đó thuộc thuật toán.
- Báo cáo riêng biến thiên do nguồn, task, model và prompt. Nhiều task dùng chung calibrator không phải nhiều lần fit độc lập.
- Nếu bootstrap theo model, số model nhỏ phải được ghi; nếu source tasks chồng lấn phải giữ cấu trúc chồng lấn trong resampling.
- Không dùng target để chọn phương pháp trong năm nhánh rồi trình bày khoảng bất định của nhánh thắng như thể đã được chỉ định trước. Các lần xem target dùng để phát triển là exploratory; xác nhận cuối cần task/model/prompt chưa dùng để chọn.

### 9.5. Đồ thị cần xuất

1. Mean delta NLL – harm, giữ mọi đối chứng cần thiết và phân biệt ba giao thức.
2. M1: tỷ lệ có hướng chung theo lớp hàm; train/select/audit/target; mức độ dùng field so với scalar.
3. M2: M target thật so với khoảng dự báo; predicted vs true delta; tỷ lệ chấp nhận/fallback.
4. M3: head BCE và đóng góp tail NLL; a so với b, chia theo K.
5. M4: mean–harm và đổi quyết định theo corruption thực tế/rho giả định.
6. M5: certified batch fraction và guaranteed/true gain theo số nhãn; so random cùng budget.

Không dùng riêng một hình ECE để kết luận thắng NLL hoặc giảm negative transfer.

## 10. Kiểm tra đúng/sai bắt buộc

### 10.1. Toàn bộ hệ thống

- Identity cho log_q=log_p và delta=0 trong sai số số học.
- Mỗi q có tổng 1, không NaN; log_q hữu hạn trên nhãn hợp lệ.
- Hoán vị lựa chọn đầu vào rồi hoàn nguyên phải cho cùng phân phối; xử lý tie nhất quán.
- K=2, K=3, K=16; probabilities gần one-hot, uniform và có số 0 đầu vào.
- Đổi ngẫu nhiên nhãn target private không được làm đổi fit/predict của M1–M4. Với M2, beta được chọn từ batch p cũng không được đổi.
- Với M5, đổi những nhãn chưa query không làm đổi quá trình cho đến khi chính case đó được query.
- Save/load cho cùng log_q; seed cố định lặp lại kết quả trong tolerance.
- Lỗi optimizer/thiếu dữ liệu có trạng thái riêng; không âm thầm thay bằng một baseline mà vẫn giữ tên method.

### 10.2. M1

- Gradient analytic so finite difference ở identity và một điểm nội miền.
- Lớp constant khớp nhiệt độ vô hướng khi objective, miền và cách tối ưu khớp; phân biệt sai số lưới và tối ưu liên tục.
- Norm bound bảo đảm beta toàn cục; kiểm tra thêm nhiều z ngẫu nhiên trong `[0,1]^3`.
- QP simplex và bất đẳng thức hướng đạt tolerance; rho=0 chỉ được diễn giải trong lớp hàm tương ứng.
- Với toy affine-resolvable, scalar không có hướng chung nhưng affine có. Với toy không thể phân biệt bằng p, không được tuyên bố luôn giải được xung đột.
- Khi không qua gate, artifact phải cho biết lý do, không ghi rằng mất common descent trên train.

### 10.3. M2

- Đẳng thức delta theo beta đúng cho mọi nhãn, nhiều K và beta hai phía 1.
- Với một khoảng chứa đúng M, max hai endpoint bằng cận B theo công thức rẽ nhánh.
- Đảo endpoint ở beta<1 phải bị test phát hiện.
- Unsupported K/bin kích hoạt đúng fallback/cận thô.
- Coverage dự báo được đo sau freeze; không được dùng true M để sửa khoảng trong adapt.

### 10.4. M3

- a=b khớp TS; a=b=1 khớp identity.
- K=2 không đổi q khi đổi a mà giữ b cố định.
- Mọi cặp thứ hạng được giữ với a,b>0; top tie giữ đúng tie.
- Tổng đóng góp top-BCE và wrong-only tail loss bằng NLL toàn phần, với trọng số đúng số mẫu.
- Gradient hai tham số đúng finite difference; kiểm tra tính lồi số học trên các đoạn ngẫu nhiên.

### 10.5. M4

- rho=0 khớp objective không robust cùng lớp hàm.
- Trên toy rất nhỏ, liệt kê toàn bộ cách đổi nhãn/case trong budget; max brute force phải khớp công thức Top-B.
- R robust không giảm khi tăng B tại cùng theta.
- Identity cho R=0 với mọi rho.
- Ngân sách và corruption cùng tính theo state hoặc được khai báo biến thể theo dòng.

### 10.6. M5

- Trên toy, liệt kê mọi nhãn chưa biết và kiểm tra tất cả true delta nằm trong `[L,U]`.
- Query thêm nhãn làm L không giảm, U không tăng cho candidate cố định.
- Biết hết nhãn thì L=U=true delta.
- Candidate được certified với oracle đúng không được có true delta > -eta ngoài tolerance.
- Đếm cost theo case, không theo số lần đọc nhãn cache hoặc số dòng cùng một text.
- Nhãn oracle phải đáng tin cho phát biểu certificate; oracle noisy cần giao thức mới, không tái dùng bảo đảm này.

## 11. Nghiên cứu liên quan và ranh giới đóng góp

Các liên kết dưới đây là nguồn gốc để nhóm code đọc công thức và triển khai baseline. Những đề xuất M1–M5 là thiết kế nghiên cứu trong tài liệu này, không phải tên đổi của các paper dưới đây. Không khẳng định chưa ai từng làm một cơ chế tương đương chỉ vì chưa tìm thấy paper trùng nguyên văn.

| ID | Nguồn gốc | Liên quan trực tiếp | Điều không được nhận nhầm là đóng góp mới |
|---|---|---|---|
| R01 | [Guo et al., On Calibration of Modern Neural Networks, ICML 2017](https://proceedings.mlr.press/v70/guo17a.html) | Temperature scaling | Một scalar nhiệt độ và fit NLL |
| R02 | [Liu et al., Conflict-Averse Gradient Descent for Multi-task Learning, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/9d27fdf2477ffbff837d73ef7ae23db9-Abstract.html) | M1; gradient xung đột, đa mục tiêu | Hình học gradient đa mục tiêu đã có; cần phân biệt đóng góp đặc thù hiệu chỉnh |
| R03 | [Long et al., Kernel Multiaccuracy, FORC 2025](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.FORC.2025.7) | M1; kernel và witness sửa sai | Học hàm sửa từ residual bằng kernel đã có |
| R04 | [Kobalczyk & van der Schaar, Discovery of Hidden Miscalibration Regimes, arXiv 2026](https://arxiv.org/abs/2605.13484) | M1; phát hiện cấu trúc sai lệch ẩn, hiệu chỉnh cục bộ | Riêng việc phát hiện các vùng over/underconfidence trong LLM đã có nghiên cứu gần |
| R05 | [Balanya et al., Adaptive Temperature Scaling for Robust Calibration of Deep Neural Networks, Neural Computing and Applications 2024](https://link.springer.com/article/10.1007/s00521-024-09505-4) | M1; HTS | Nhiệt độ theo entropy đã có |
| R06 | [Xie et al., Calibrating Language Models with Adaptive Temperature Scaling, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.1007/) | M1; ATS cho LLM | Nhiệt độ thích nghi cho từng token đã có; cần đúng input của bản tái lập |
| R07 | [Yu et al., Robust Calibration with Multi-domain Temperature Scaling, NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/b054fadf1ccd80b37d465f6082629934-Abstract-Conference.html) | M1/M2; nhiều miền và chuyển giao | Nhiều nguồn và temperature theo đầu vào đã có |
| R08 | [Garg et al., Leveraging Unlabeled Data to Predict Out-of-Distribution Performance, ICLR 2022](https://arxiv.org/abs/2201.04234) | M2; ATC và giới hạn nhận diện hiệu quả không nhãn | Dự báo hiệu quả từ target không nhãn đã có; M2 cần phân biệt dự báo lợi ích NLL của hiệu chỉnh |
| R09 | [Lu et al., Characterizing Out-of-Distribution Error via Optimal Transport, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/38fd51cf36f28566230a93a5fbeaabbf-Paper-Conference.pdf) | M2; COT/COTT | Ước lượng hiệu quả OOD có điều kiện/giả định; không có bảo đảm shift tùy ý từ p không nhãn |
| R10 | [Wang et al., Transferable Calibration with Lower Bias and Variance in Domain Adaptation, NeurIPS 2020](https://arxiv.org/abs/2007.08259) | M2; TransCal | Dùng target không nhãn để hiệu chỉnh chuyển miền đã có |
| R11 | [Le Coz et al., Confidence Calibration of Classifiers with Many Classes, NeurIPS 2024](https://arxiv.org/abs/2411.02988) | M3; Top-versus-All | Cải thiện confidence qua bài toán correctness nhị phân đã có |
| R12 | [Rahimi et al., Intra Order-preserving Functions for Calibration of Multi-Class Neural Networks, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/9bc99c590be3511b8d53741684ef574c-Abstract.html) | M3; biến đổi giữ thứ hạng | Giữ nguyên thứ hạng khi tăng expressiveness đã có |
| R13 | [Zhang et al., Mix-n-Match: Ensemble and Compositional Methods for Uncertainty Calibration in Deep Learning, ICML 2020](https://proceedings.mlr.press/v119/zhang20k.html) | ETS; đối chứng expressiveness | Trộn probabilities/ensemble calibrator không tự là ý tưởng mới |
| R14 | [Wu et al., TransTS: An Adaptive Post-hoc Method for Probability Calibration under Label Noise, Science China Information Sciences 2026](https://link.springer.com/article/10.1007/s11432-025-4576-9) | M4; calibration dưới nhãn nhiễu | Hiệu chỉnh có xét noise đã có; M4 cần phân biệt tập nhiễu và source-relative regret |
| R15 | [Hanada & Futami, Noisy-Label Calibration with Multiple Unreliable Labels, arXiv 2026](https://arxiv.org/abs/2610.19543) | M4; giả định nhận diện noise/annotator | Nhiều nhãn không tự cho nhãn thật; cần giả định để hiệu chỉnh clean-risk |
| R16 | [Kossen et al., Active Testing: Sample-Efficient Model Evaluation, ICML 2021](https://proceedings.mlr.press/v139/kossen21a.html) | M5; chọn nhãn để đánh giá hiệu quả | Active query để đánh giá model đã có |
| R17 | [Kossen et al., Active Surrogate Estimators: An Active Learning Approach to Label-Efficient Model Evaluation, NeurIPS 2022](https://arxiv.org/abs/2202.06881) | M5; active surrogate | Dùng surrogate để tiết kiệm nhãn đánh giá đã có |
| R18 | [Angelopoulos et al., Learn then Test: Calibrating Predictive Algorithms to Achieve Risk Control](https://arxiv.org/abs/2110.01052) | Risk control và lựa chọn có kiểm định | Chọn tham số dưới ràng buộc rủi ro đã có; bootstrap thường không tự biến thành certificate |

Ghi chú mức đọc: TransTS được đối chiếu qua abstract/trang xuất bản chính thức; PDF toàn văn không tải được trong lượt nghiên cứu này. Không khẳng định tài liệu đã audit code hoặc toàn bộ công thức của TransTS. Các mục ghi arXiv là preprint theo nguồn đã đối chiếu, không tự gán venue chấp nhận.

Các đẳng thức NLL của M2, phân rã loss M3, đối thủ Top-B M4 và cận hữu hạn batch M5 là suy luận đại số của thiết kế này. Chúng cần được kiểm tra bằng test và rà soát tính mới; một đẳng thức đúng không tự là một định lý mới trong literature.

## 12. Yêu cầu giao việc có thể gửi nguyên văn cho bên code

> Triển khai năm module nghiên cứu trong tài liệu này trên probabilities hiện có, ưu tiên M1, M3 và M4 dùng nguồn; M2 dùng target không nhãn và M5 dùng oracle có ngân sách phải tách protocol. Không cần đọc lại kết quả cũ để suy ra kết quả mong muốn. Không đổi code baseline hiện có hoặc ghi đè artifact cũ.
>
> M1 phải có chẩn đoán gradient tại identity, lớp hàm kernel hữu hạn có bound beta toàn cục, hai objective MINIMAX/MEAN-SAFE và đối chứng POOLED cùng capacity. M3 phải đúng biến đổi top-gap hai tham số, có binary-collapse test. M4 phải dùng đối thủ Top-B theo state và xác minh bằng brute force. M2 phải dùng đúng đẳng thức một moment cho beta chung, ghi rõ khoảng dự báo thực nghiệm và kiểm tra coverage. M5 phải che target y, chỉ truy cập oracle khi query, dùng cận tất định cho batch và đếm đúng label budget.
>
> Trước hết chạy toy tests để xác minh cơ chế và công thức. Sau đó khóa config, split, endpoint và menu hyperparameter bằng nguồn. Chạy toàn bộ roster target đã chỉ định, xuất cả kết quả thắng/hòa/thua và mọi fallback. Các lần phát triển có xem target phải đánh dấu exploratory. Không chọn model, prompt, nguồn hoặc baseline theo việc kết quả có lợi cho phương pháp mới.
>
> Bàn giao code, cấu hình tái lập, fitted artifacts, predictions, metrics, so sánh ghép cặp, runtime, checks và decision.json. Kết luận riêng ba câu: triển khai có đúng không; cơ chế có xuất hiện ngoài tập fit không; có lợi ích target so với đối chứng phù hợp không. Không kết luận thành công chỉ vì loss train giảm hoặc harm bằng 0 khi luôn giữ nguyên.

**Kết thúc đặc tả.**
