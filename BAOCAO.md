# Báo cáo LAB 16 — Cloud AI Environment Setup (Oracle Cloud Infrastructure)

**Học viên:** Nguyễn Văn Đại
**Cloud:** Oracle Cloud Infrastructure (OCI) — gói Always Free
**Instance:** VM.Standard.E2.1.Micro (1 OCPU / 1 GB RAM, x86), region Singapore
**Bài toán:** Train + inference mô hình LightGBM phát hiện gian lận thẻ tín dụng (dataset Credit Card Fraud, 284,807 giao dịch).

## Kết quả benchmark

| Metric | Kết quả |
|---|---|
| Thời gian load data | 3.063 s |
| Thời gian training | 7.469 s |
| Best iteration | 1 |
| AUC-ROC | 0.9323 |
| Accuracy | 0.9679 |
| F1-Score | 0.0888 |
| Precision | 0.0467 |
| Recall | 0.9082 |
| Inference latency (1 row) | 4.945 ms |
| Inference throughput (1000 rows) | ~707,422 rows/s |

## Nhận xét

Môi trường CPU Always Free của OCI (E2.1.Micro, 1 GB RAM + 4 GB swap) đủ để chạy toàn bộ pipeline ML thực tế: load 284,807 dòng dữ liệu trong ~3 giây và huấn luyện LightGBM trong ~7.5 giây — rất nhanh nhờ thuật toán gradient boosting tối ưu cho CPU. Mô hình đạt **AUC-ROC 0.93** và **Recall 0.91**, tức bắt được phần lớn giao dịch gian lận. Tuy nhiên **Precision chỉ 4.7%** và **F1 thấp** do đã dùng cơ chế cân bằng lớp (scale_pos_weight / is_unbalance) trên tập cực kỳ mất cân bằng (chỉ 394/227,845 mẫu gian lận ≈ 0.17%): mô hình nghiêng về "báo gian lận" nên gây nhiều false positive, và early stopping chốt ngay ở cây thứ nhất. Về hiệu năng phục vụ, inference latency ~5 ms/dòng và throughput hàng trăm nghìn dòng/giây là thừa sức cho ứng dụng thực. Về chi phí, do dùng đúng shape trong hạn mức Always Free nên tổng chi phí hiển thị **$0.00** trên Cost Analysis — đây là kết quả đúng, không phải lỗi. Nếu muốn cải thiện chất lượng mô hình, có thể bỏ cân bằng lớp và điều chỉnh ngưỡng phân loại để cân đối lại Precision/Recall.

## Ghi chú về cài đặt môi trường (cloud-init)

Môi trường ML (Python, pip, LightGBM, scikit-learn, pandas, numpy, kaggle) được cài **thủ công qua SSH** sau khi instance khởi động, thay vì qua cloud-init lúc tạo instance. Nội dung cloud-init tương đương (nếu tự động hóa) sẽ là:

```yaml
#cloud-config
package_update: true
packages:
  - python3
  - python3-pip
runcmd:
  - pip3 install --upgrade pip
  - pip3 install lightgbm scikit-learn pandas numpy kaggle
```

Ngoài ra, do instance chỉ có 1 GB RAM (VM.Standard.E2.1.Micro), đã thêm swap để tránh hết bộ nhớ khi load dataset và train:

```bash
sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
```
