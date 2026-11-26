# Báo cáo nghiên cứu adaptive embedding fusion

26/11/2026 · **Đã xác nhận test R52 qua ba seed**

Giữ kiến trúc AG-CMF hiện tại và dùng **training-time Logit Adjustment τ=1** cho R52. Trên test, cấu hình này đạt **96,04% Accuracy, 80,24% Macro-F1 và 61,09% Rare F1**, vượt cả AG-CMF + CE lẫn MLP cân bằng tham số theo cả ba chỉ số ở từng seed. Fusion phân cấp đã bị loại ở pilot. Kết luận về khả năng chuyển miền còn để ngỏ vì cache FP16 Ohsumed vẫn từ chối truy cập.

## Kết quả lịch sử đã xác minh

Các số dưới đây được tính lại từ 30 hàng metrics của teammate, đối chiếu với ablation CSV và manifest, trên source test split. Mỗi ô dùng mean ± sample SD của seeds 42/43/44, theo %. Mỗi dataset dùng dữ liệu và cache FP16 chung theo báo cáo nguồn, với 100 epochs và checkpoint chọn theo validation accuracy. Fingerprint dữ liệu khớp khi tái tạo. Bộ số lịch sử không có feature manifests để xác nhận bytes và revisions độc lập.

| Dataset | Phương pháp | Accuracy | Macro-F1 | Rare F1 |
|---|---|---:|---:|---:|
| R52 | #11 Avg+Cat | 91.91 ± 0.08 | 42.54 ± 0.68 | 0.00 ± 0.00 |
| R52 | #15 Exp 3 | 93.22 ± 0.04 | 49.85 ± 0.54 | 0.00 ± 0.00 |
| R52 | #15 + Weighted Loss | 91.43 ± 0.38 | 74.09 ± 0.99 | 54.35 ± 3.63 |
| R52 | Parameter-Matched MLP | 95.56 ± 0.04 | 72.77 ± 0.48 | 39.09 ± 1.38 |
| R52 | AG-CMF | 95.81 ± 0.21 | 76.94 ± 0.80 | 50.72 ± 1.33 |
| Ohsumed | #11 Avg+Cat | 67.66 ± 2.07 | 50.75 ± 5.59 | 0.00 ± 0.00 |
| Ohsumed | #15 Exp 3 | 69.83 ± 0.56 | 58.08 ± 1.96 | 0.00 ± 0.00 |
| Ohsumed | #15 + Weighted Loss | 65.45 ± 0.35 | 59.71 ± 0.62 | 11.99 ± 2.56 |
| Ohsumed | Parameter-Matched MLP | 73.06 ± 0.20 | 66.73 ± 0.63 | 20.46 ± 7.25 |
| Ohsumed | AG-CMF | 72.58 ± 0.59 | 65.70 ± 0.65 | 9.89 ± 8.58 |

| So sánh ghép theo seed | Dataset | Δ Accuracy | Δ Macro-F1 | Δ Rare F1 |
|---|---|---:|---:|---:|
| AG-CMF − MLP | R52 | +0.25 ± 0.18 | +4.16 ± 0.33 | +11.63 ± 1.42 |
| AG-CMF − MLP | Ohsumed | −0.49 ± 0.78 | −1.03 ± 1.25 | −10.57 ± 14.95 |
| Weighted #15 − #15 | R52 | −1.79 ± 0.34 | +24.24 ± 1.48 | +54.35 ± 3.63 |
| Weighted #15 − #15 | Ohsumed | −4.38 ± 0.49 | +1.63 ± 1.46 | +11.99 ± 2.56 |

AG-CMF thắng MLP về Macro-F1 trên cả ba seed R52 (+4.36, +4.34, +3.78 điểm %) và một trong ba seed Ohsumed (+0.21, −2.30, −1.00 điểm %). MLP đã đạt 72.77% Macro-F1 trên R52 trong khi #15 đạt 49.85%, nên năng lực học đặc trưng của head là một giải thích cạnh tranh cho phần cải thiện từ #15. MLP thiếu 5.494 tham số so với AG-CMF, khoảng 0,019%, và dùng kiến trúc khác. Weighted Loss trong raw chỉ có cho #15, với trọng số nghịch đảo tần suất train. Kết quả đó chưa đo tác động của logit adjustment hoặc bất kỳ objective cân bằng nào trên chính AG-CMF.

[Bảng gốc của teammate](https://github.com/ThanhNosaur/LLMEmbed/blob/88f1089d9275d942f22736ad01a256ecde49fd66/kaggle_experiment/reports/README.md) · [Biểu đồ theo seed](audit/historical_fp16_contrasts.png)

## Audit kiến trúc và giao thức

| Cấu hình | Điều được xác minh từ code |
|---|---|
| LLMEmbed ACL 2024 | Mã gốc trích năm tầng cuối LLaMA-2-7B-Chat, mean trên chiều token đã pad, BERT pooler và token đầu RoBERTa. #15 dùng Gram 7×7 sau chiếu, `2 sigmoid(σG) − 1`, nối với mean LLaMA. `main.py` dùng Adam và head xuất softmax trước CE. 96.20% trong paper là accuracy trung bình năm bộ trên giao thức paper. |
| Exp 0 nội bộ | Controlled dùng LLaMA-2 base, mean che padding, xuất logits cho CE, AdamW/cosine và validation nội bộ. Chế độ `author` cố gắng khớp đường tính gốc, nhưng dữ liệu và môi trường còn khác. |
| Exp 1 → Exp 3 | Exp 1 chọn các tầng 32/28/24/20/16. Exp 2 đổi cả prompt lẫn pooling theo cosine, không có sinh chuỗi suy luận. Exp 3 thêm cửa sổ token 256 chồng 32 và learnable chunk pooling. |
| Exp 4 | Cùng extraction/cache với Exp 3. ChunkPool riêng mỗi source, projection riêng mỗi vector, MHA có residual, shared scalar gate trên bảy vector, Hadamard giữa mean hai nhóm, rồi MLP classifier. CE chuẩn và checkpoint validation accuracy. |
| Exp 5B / 5C | Exp 5B thay BERT bằng DeBERTa-v3-large. Exp 5C thay thêm LLaMA-2 bằng Qwen2.5-7B. Thay backbone nên nguồn cache và embedding thay đổi. Đây là kết quả thăm dò lịch sử. |
| Strong controls | #11 gộp mean LLaMA rồi concat với BERT/RoBERTa. MLP concat năm tầng và hai encoder rồi dùng MLP có số tham số gần AG-CMF. Weighted #15 giữ fusion #15. Có thể gán chênh lệch cho cả thiết kế head, chưa tách từng projection, attention, gate và Hadamard. |

[Paper ACL 2024](https://aclanthology.org/2024.acl-long.433/) · [Mã tác giả](https://github.com/ChunLiu-cs/LLMEmbed-ACL2024) · [Mã và báo cáo teammate tại commit audit](https://github.com/ThanhNosaur/LLMEmbed/tree/88f1089d9275d942f22736ad01a256ecde49fd66)

### Split và độ tin cậy

| Dataset | Train / validation / test | Rare / Medium / Frequent theo train | Validation coverage | Trùng chính xác với train |
|---|---:|---:|---|---|
| R52 | 5.870 / 662 / 2.568 | 18 / 22 / 12 | 51/52. `platinum` vắng mặt, 17 lớp Rare còn lại có một mẫu/lớp. | 19 validation, 6 test |
| Ohsumed | 3.021 / 336 / 4.043 | 1 / 7 / 15 | 23/23. Rare chỉ có C22, với 8 train, 1 validation, 10 test. | 0 validation, 0 test |

R52 giữ benchmark gốc và ghi một phép sensitivity trên 643 mẫu validation và 2.562 mẫu test còn lại. Trong 19 mẫu validation trùng văn bản train, có hai mẫu mà nhãn không trùng bất kỳ bản sao nào trong train. Việc báo cáo thêm tập lọc cần đi kèm quy tắc checkpoint gốc và kiểm tra theo support. Fingerprint khớp CSV xác nhận văn bản, split, label mapping và thứ tự dữ liệu tạo bằng loader hiện tại. Nó không xác nhận đầu ra backbone. Kết quả tác giả ACL 2024 dùng giao thức khác và có dữ liệu tiền xử lý chưa đối chiếu từng mẫu.

[Audit R52](audit/r52_data_audit.json) · [Audit Ohsumed](audit/ohsumed_data_audit.json)

## Chẩn đoán baseline trên validation R52

Baseline AG-CMF + CE v6 chạy đủ 100 epochs với seed 42 trên cache FP16 mới. Epoch 26 là epoch đầu đạt validation accuracy cao nhất. Các giá trị Macro-F1 trong bảng tính trên lớp có support. Rare có 17 lớp, mỗi lớp một mẫu.

| Quy tắc dự đoán trên cùng checkpoint | Accuracy gốc | Macro-F1 gốc | Accuracy lọc trùng | Macro-F1 lọc trùng | Rare F1 | Rare đúng |
|---|---:|---:|---:|---:|---:|---:|
| Logits gốc | 96,07% | 68,52% | 96,11% | 68,55% | 27,45% | 5/17 |
| Dịch hậu nghiệm τ=1 | 94,41% | 76,21% | 94,40% | 76,20% | 53,64% | 11/17 |

Điều chỉnh hậu nghiệm giảm lỗi rare → frequent từ 4 xuống 2 và tăng Macro-F1 7,69 điểm %, với Accuracy giảm 1,66 điểm %. Trong 589 mẫu frequent, số dự đoán đúng giảm từ 579 xuống 561. Hiệu ứng vẫn hiện diện khi bỏ 19 mẫu trùng. Đây là bằng chứng rằng decision rule đóng góp vào trade-off trên checkpoint cố định. Mức hiệu quả của **training-time** adjustment cần đo riêng theo cùng protocol.

Trọng số trung bình trên nhánh pooled là LLaMA **90,61%**, BERT 1,51% và RoBERTa 7,87%. LLaMA nhận 89,00% trên 17 mẫu rare. Cosine trung bình của các cặp layer LLaMA trước MHA là 0,203, so với 0,083 giữa LLaMA và các encoder. Đây là thống kê mô tả của một checkpoint. Attention và Hadamard còn duy trì đường ảnh hưởng của các nguồn ngoài nhánh gate.

Trong lịch sử training, epoch 72 đạt cùng accuracy 96,07% và full-label Macro-F1 71,44%, so với 67,21% ở epoch 26. Checkpoint selection là một giải thích cạnh tranh cho một phần chênh lệch. Pilot tiếp tục giữ tie-break theo epoch đầu để so sánh nhất quán. [Raw metrics](audit/kaggle_downstream_v6/study/study_metrics.csv) · [Biểu đồ hội tụ và gate](audit/r52_baseline_validation.png)

## Pilot A trên cùng cache R52

| Seed 42, validation | Checkpoint | Accuracy gốc | Macro-F1 có support | Accuracy lọc trùng | Macro-F1 có support lọc trùng | Rare F1 có support |
|---|---:|---:|---:|---:|---:|---:|
| AG-CMF + CE | 26 | 96,07% | 68,52% | 96,11% | 68,55% | 27,45% |
| Hierarchical + CE | 12 | 96,22% | 68,02% | 96,27% | 68,01% | 27,45% |

Đổi gate giảm tỷ trọng LLaMA từ 90,61% xuống 85,88% và tăng entropy theo backbone từ 0,342 lên 0,477 nats. Trên riêng nhóm rare, tỷ trọng LLaMA giảm từ 89,00% xuống 87,21%. Chênh lệch dự đoán ghép theo mẫu cho thấy một dự đoán rare được sửa và một dự đoán rare đúng trở thành sai, tổng đúng vẫn 5/17. Số lỗi rare → frequent tăng từ 4 lên 5. Mức tăng accuracy +0,15 điểm % chủ yếu phản ánh một dự đoán frequent đúng thêm. Macro-F1 giảm 0,51 điểm % trên validation gốc và 0,54 sau lọc trùng.

Cả hai head có 29.024.245 tham số. v7 ghi 298,31 giây train + validation so với 280,69 giây của v6. Số đo inference đơn lẻ dao động dưới một giây. Gate phân cấp làm thay đổi cách sử dụng các nguồn trên nhánh pooled, với tín hiệu chất lượng thấp hơn ngưỡng đã đặt. Loại A sau pilot seed 42. [Raw metrics A](audit/kaggle_downstream_v7/study/study_metrics.csv) · [Paired metrics](audit/r52_pilot/study_paired_summary.csv)

## Pilot B trên cùng kiến trúc AG-CMF

v8 thay objective CE bằng training-time logit adjustment τ=1. Các trường còn lại trong experiment config, training config, thứ tự mẫu và feature provenance đều khớp v6. Checkpoint theo validation accuracy là epoch 64. Các số dưới đây dùng logits gốc của checkpoint B.

| Seed 42, validation | Accuracy gốc | Macro-F1 có support | Accuracy lọc trùng | Macro-F1 có support lọc trùng | Rare F1 có support |
|---|---:|---:|---:|---:|---:|
| AG-CMF + CE | 96,07% | 68,52% | 96,11% | 68,55% | 27,45% |
| AG-CMF + LA τ=1 | 95,77% | 81,82% | 95,96% | 81,91% | 62,75% |
| Δ điểm % | −0,30 | +13,30 | −0,16 | +13,36 | +35,29 |

B dự đoán đúng 12/17 rare và 54/56 medium, so với 5/17 và 52/56 ở baseline. Nhóm frequent giảm từ 579/589 xuống 568/589. So với phép dịch logits hậu nghiệm trên cùng baseline, B đạt accuracy cao hơn 1,36 điểm % và Macro-F1 có support cao hơn 5,62 điểm %. Phép so sánh này gồm cả quá trình học và decision rule. Gate mass LLaMA giảm từ 90,61% xuống 86,49%, dù kiến trúc giữ nguyên. Đây là tín hiệu objective ảnh hưởng biểu diễn học được, chưa tách được tác động của nó khỏi classifier. Thời gian train + validation của B là 282,40 giây, so với 280,69 giây của baseline trong một lượt mỗi phương án.

Đã khóa lựa chọn B bằng validation gốc và validation lọc trước khi chạy test. [Raw metrics B](audit/kaggle_downstream_v8/study/study_metrics.csv) · [Quyết định khóa](audit/r52_pilot/decision.json)

## Xác nhận trên test R52

Kaggle Downstream v9 chạy đủ chín lượt, mỗi phương pháp có seeds 42/43/44, 100 epochs và checkpoint chọn bằng validation accuracy. Ba phương pháp dùng cùng cache FP16 Extraction v2. Đã xác minh quyết định đã khóa, cache receipt, lịch sử checkpoint, thứ tự mẫu và các chỉ số tính lại từ predictions và logits. Cả 52 nhãn đều hiện diện trong test. Các số là mean ± sample SD qua ba seeds, theo %.

| Tập test | Phương pháp | Accuracy | Macro-F1 | Rare F1 |
|---|---|---:|---:|---:|
| Gốc · 2.568 mẫu | AG-CMF + CE | 95,78 ± 0,15 | 76,92 ± 0,72 | 50,27 ± 2,08 |
| Gốc · 2.568 mẫu | MLP cân bằng tham số + CE | 95,53 ± 0,08 | 72,50 ± 0,94 | 38,37 ± 2,63 |
| Gốc · 2.568 mẫu | **AG-CMF + LA τ=1** | **96,04 ± 0,13** | **80,24 ± 1,31** | **61,09 ± 4,04** |
| Lọc trùng · 2.562 mẫu | AG-CMF + CE | 95,77 ± 0,15 | 76,87 ± 0,72 | 50,27 ± 2,08 |
| Lọc trùng · 2.562 mẫu | MLP cân bằng tham số + CE | 95,52 ± 0,08 | 72,45 ± 0,94 | 38,37 ± 2,63 |
| Lọc trùng · 2.562 mẫu | **AG-CMF + LA τ=1** | **96,03 ± 0,13** | **80,21 ± 1,31** | **61,09 ± 4,04** |

| Chênh lệch ghép theo seed | Tập test | Δ Accuracy | Δ Macro-F1 | Δ Rare F1 |
|---|---|---:|---:|---:|
| AG-CMF + LA − AG-CMF + CE | Gốc | +0,26 ± 0,04 | +3,32 ± 0,85 | +10,82 ± 2,32 |
| AG-CMF + LA − AG-CMF + CE | Lọc trùng | +0,26 ± 0,05 | +3,33 ± 0,86 | +10,82 ± 2,32 |
| AG-CMF + LA − MLP + CE | Gốc | +0,51 ± 0,12 | +7,74 ± 1,00 | +22,71 ± 3,94 |
| AG-CMF + LA − MLP + CE | Lọc trùng | +0,51 ± 0,12 | +7,76 ± 1,01 | +22,71 ± 3,94 |

Chênh lệch dùng điểm phần trăm. So với AG-CMF + CE, Macro-F1 của LA tăng lần lượt 4,17, 3,31 và 2,47 điểm % ở seeds 42/43/44. Rare F1 tăng 13,43, 10,01 và 9,02 điểm %. Accuracy tăng ở cả ba seed. Ba chỉ số trung bình cùng vượt các mốc tham khảo 95,5%, 80% và 60%. Lợi ích Macro-F1 nhỏ hơn mức tăng 13,30 điểm % quan sát trên validation seed 42.

### Sai số theo nhóm và chi phí

Trên 63 mẫu test rare, số dự đoán đúng theo seed tăng từ 34/28/33 lên 43/40/38. Tỷ lệ mẫu rare bị gán sang frequent giảm từ **29,10% xuống 17,46%**. F1 trung bình theo lớp tăng ở 10 trong 18 lớp rare, giảm ở ba lớp và giữ nguyên ở năm lớp. Medium Macro-F1 giảm 1,48 điểm % dù số dự đoán medium đúng tăng ở cả ba seed. Frequent Macro-F1 tăng 0,86 điểm %, trong khi số mẫu frequent đúng giảm ở cả ba seed. Vì Accuracy, recall và Macro-F1 đo các đại lượng khác nhau, các thay đổi này cần đọc cùng bảng precision, recall và confusion.

Trong test v9, tỉ trọng gate trung bình của LLaMA trên nhánh pooled giảm từ 89,01% xuống 84,82%, tương ứng 88,06% xuống 81,98% trên riêng mẫu rare. Kiến trúc giữ nguyên giữa hai lượt. Số đo mô tả sự thay đổi của routing học được dưới objective mới, chưa xác định được phần chênh lệch do routing hay classifier.

AG-CMF với CE và LA đều có 29.024.245 tham số. LA tăng train + validation từ 284,76 lên 287,60 giây trung bình mỗi seed. Inference trên source test lần lượt là 0,759 và 0,777 giây, với chênh lệch ghép +0,017 ± 0,051 giây. MLP nhanh hơn khi training, trung bình 260,84 giây mỗi seed, và có 29.018.751 tham số. Tổng downstream ước lượng từ executor logs là khoảng **2,30 GPU-hours** cho ba pilot và confirmation, tính cả hai T4 được cấp. v5 dừng sớm trước training và extraction được tính riêng.

Trong biểu đồ, mỗi điểm thể hiện một seed và số ghi cạnh nhóm là chênh lệch trung bình. [Xác minh v9](audit/r52_confirmation/verification.json) · [Raw metrics](audit/r52_confirmation/study_metrics.csv) · [Tổng hợp](audit/r52_confirmation/study_summary.csv) · [Theo lớp](audit/r52_confirmation/per_class_summary.csv) · [Paired differences](audit/r52_confirmation/study_paired_summary.csv) · [Biểu đồ](audit/r52_confirmation/paired_test_comparison.png) · [Kaggle confirmation](https://www.kaggle.com/code/longvucs/llmembed-fusion-fp16-downstream)

## Trả lời các câu hỏi nghiên cứu

1. **Hạn chế thực tế của AG-CMF.** Lợi ích so với MLP khác nhau giữa hai miền. Trên pilot baseline R52, gate tập trung vào LLaMA và điều chỉnh decision rule cải thiện F1 của lớp hiếm với trade-off accuracy. Đóng góp nhân quả của gate, độ dư thừa, mất đặc trưng sau pooling và ảnh hưởng của class-agnostic routing còn cần đối chứng.
2. **Fusion được chọn.** Hierarchical Adaptive Fusion phân bổ trọng số giữa năm tầng rồi giữa ba backbone, dùng lại gate MLP. Vẫn có 29.024.245 tham số trên R52, bằng AG-CMF, và giữ nguyên classifier cùng nhánh Hadamard. Khi gate logits đồng nhất, ba backbone nhận 1/3 mass mỗi backbone trên nhánh pooled. Chi phí cộng thêm của gate đã được xác định từ code, runtime inference còn phải đo.
3. **Objective được chọn.** Training-time logit adjustment với τ=1, prior từ train. CE cộng `log π` ở train, inference dùng logits gốc. Logit dịch hậu nghiệm trên checkpoint CE là một phép chẩn đoán decision rule ở validation. [Menon et al., ICLR 2021](https://arxiv.org/abs/2007.07314)
4. **Mức cải thiện khi chỉ đổi Fusion.** Trên pilot seed 42, Accuracy +0,15 điểm %, Macro-F1 có support −0,51 điểm % và Rare F1 giữ nguyên. Loại Hierarchical + CE ở vòng này.
5. **Đóng góp của objective.** Trên test R52 với cùng kiến trúc và cache, B tăng Accuracy 0,26, Macro-F1 3,32 và Rare F1 10,82 điểm %. Cả ba tăng ở từng seed và giữ chênh lệch tương tự sau lọc trùng.
6. **Hiệu ứng kết hợp.** A+B nằm ngoài vòng tiếp theo vì A không qua sàng lọc validation.
7. **Tính nhất quán giữa miền.** Test mới chỉ xác nhận R52. Lượt kiểm tra sau v9 vẫn nhận HTTP 403 với notebook FP16 Ohsumed. Kết quả Ohsumed trong phần đầu là đối chứng lịch sử.
8. **So với strong control.** Trên cùng cache R52, AG-CMF + LA hơn MLP + CE 7,74 điểm % Macro-F1 và 22,71 điểm % Rare F1. So sánh này thay cả head lẫn objective. So sánh chỉ thay objective là B với AG-CMF + CE. Chưa đo MLP + LA trong nghiên cứu này.
9. **Mức độ kiểm chứng.** Hiệu ứng của objective được xác nhận trên R52, với ba seeds và phép loại mẫu trùng. Bằng chứng về fusion phân cấp chỉ gồm pilot một seed. Gate mass và cosine là thống kê mô tả. Chưa xác nhận chuyển miền hay nguyên nhân nhân quả của routing.
10. **Quyết định.** Giữ AG-CMF và áp dụng LA τ=1 cho R52. Dừng tuning trên R52. Nếu mở xác nhận Ohsumed, giữ nguyên cấu hình đã khóa và dùng cache Ohsumed được xác thực.

## Bằng chứng thực thi và tái lập

Kaggle preflight v3 xác minh fingerprint và sáu test dữ liệu tổng hợp. Extraction v2 hoàn tất cache FP16 mới với ba model revisions đã pin trên T4 ×2. Cache mới có provenance riêng và là input chung của các đối chứng mới. Downstream v5 tìm thấy đúng receipt rồi dừng ở kiểm tra số GPU trước training. Pilot v6–v8 và test v9 hoàn tất trên T4 ×2. Phạm vi công bố và hướng dẫn chạy được ghi trong tài liệu tái lập.

[Preflight Kaggle](https://www.kaggle.com/code/longvucs/llmembed-fusion-r52-preflight) · [Extraction Kaggle](https://www.kaggle.com/code/longvucs/llmembed-fusion-r52-fp16-extraction) · [Downstream Kaggle](https://www.kaggle.com/code/longvucs/llmembed-fusion-fp16-downstream) · [Tái lập](REPRODUCIBILITY.md)
