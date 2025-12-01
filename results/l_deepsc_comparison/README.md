# README: So Sánh L-DeepSC vs FuzSemCom

## Tổng Quan

Thư mục này chứa kết quả so sánh toàn diện giữa hai phương pháp Semantic Communication:
- **L-DeepSC**: Lite Distributed Semantic Communication System (baseline)
- **FuzSemCom**: Fuzzy Semantic Communication (proposed method)

## Mục Đích Thí Nghiệm

So sánh công bằng giữa L-DeepSC và FuzSemCom trên cùng một dataset (Agriculture Sensor Data) để đánh giá:
1. Độ chính xác phân loại semantic (Classification Accuracy)
2. Khả năng tái tạo dữ liệu gốc (Reconstruction Quality)
3. Hiệu quả băng thông (Bandwidth Efficiency)
4. Độ phù hợp với IoT (IoT Suitability)

## Cấu Trúc Files

```
l_deepsc_comparison/
├── README.md                          # File này
├── DETAILED_RESULTS.md                # Kết quả chi tiết từng metric
├── l_deepsc_results.json              # Kết quả đầy đủ của L-DeepSC
├── comparison_with_fuzsemcom.json     # So sánh trực tiếp 2 phương pháp
├── l_deepsc_training_results.png      # Biểu đồ training history
└── l_deepsc_model.pth                 # Model weights đã train
```

## Cách Chạy Thí Nghiệm

### Yêu Cầu

- Python 3.8+
- PyTorch
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn

### Chạy Script

```bash
cd /home/nhannv02/Hello/fuzsemcom_project_readme/experiments
python l_deepsc_compare.py
```

### Đầu Vào

- Dataset: `data/processed/semantic_dataset_preprocessed.csv`
- Phải chạy notebook trước để tạo preprocessed data

### Đầu Ra

- Kết quả JSON: `results/l_deepsc_comparison/l_deepsc_results.json`
- So sánh JSON: `results/l_deepsc_comparison/comparison_with_fuzsemcom.json`
- Biểu đồ: `results/l_deepsc_comparison/l_deepsc_training_results.png`
- Model: `results/l_deepsc_comparison/l_deepsc_model.pth`

## Kết Quả Tổng Quan

### Classification Metrics

| Metric | L-DeepSC | FuzSemCom | Winner |
|--------|----------|-----------|--------|
| Accuracy | 77.63% | 94.99% | FuzSemCom |
| F1 (macro) | 0.6329 | 0.9148 | FuzSemCom |
| Precision (macro) | 0.6342 | 0.9246 | FuzSemCom |
| Recall (macro) | 0.7093 | 0.9152 | FuzSemCom |

### Reconstruction Metrics

| Metric | L-DeepSC | FuzSemCom | Winner |
|--------|----------|-----------|--------|
| Overall RMSE | 4.02 | 10.0 | L-DeepSC |
| Overall MAE | 3.07 | ~8.0 | L-DeepSC |
| Overall R² | 0.8634 | ~0.6 | L-DeepSC |

### Efficiency Metrics

| Metric | L-DeepSC | FuzSemCom | Winner |
|--------|----------|-----------|--------|
| Bandwidth | 64 bytes | 2 bytes | FuzSemCom |
| Compression Ratio | 0.31x | 10.0x | FuzSemCom |
| Latency | 1.44 ms | < 1 ms | FuzSemCom |
| Hardware | GPU | CPU | FuzSemCom |

## Giải Thích Kết Quả

### Tại Sao FuzSemCom Thắng Về Classification?

1. **Thiết kế chuyên biệt**: FuzSemCom được thiết kế đặc biệt cho semantic classification với Fuzzy Semantic Encoder
2. **Rule-based classification**: Sử dụng logic fuzzy để phân loại trực tiếp từ semantic symbols
3. **Tối ưu hóa mục tiêu**: Mục tiêu chính là classification accuracy, không phải reconstruction

### Tại Sao L-DeepSC Thắng Về Reconstruction?

1. **Autoencoder end-to-end**: L-DeepSC là mô hình autoencoder được tối ưu hóa cho reconstruction
2. **Loss function**: Sử dụng MSE loss trực tiếp cho reconstruction task
3. **Kiến trúc**: Semantic decoder được thiết kế để khôi phục dữ liệu gốc

### Tại Sao FuzSemCom Tiết Kiệm Băng Thông Hơn?

1. **Compression mạnh**: FuzSemCom nén 5 sensors (20 bytes) xuống 2 bytes (10x compression)
2. **Semantic symbols**: Chỉ truyền ID (1 byte) + confidence (1 byte) thay vì toàn bộ dữ liệu
3. **L-DeepSC**: Truyền 16 float32 (64 bytes) qua kênh, không nén hiệu quả

## Phân Tích Chi Tiết

Xem file `DETAILED_RESULTS.md` để biết:
- Kết quả chi tiết theo từng sensor
- Phân tích từng metric
- So sánh kỹ thuật
- Kết luận và khuyến nghị

## Cấu Hình Thí Nghiệm

### L-DeepSC Configuration

- Input dimension: 5 (sensors)
- Latent dimension: 16
- Channel dimension: 16
- SNR: 10.0 dB
- Optimizer: Adam (lr=0.001)
- Batch size: 64
- Epochs: 100
- Loss: MSE (reconstruction) + CrossEntropy (classification)

### Dataset Configuration

- Total samples: 45,351
- Train: 36,280 (80%)
- Test: 9,071 (20%)
- Sensors: Moisture, pH, N, Temperature, Humidity
- Semantic classes: 8 classes

## Hạn Chế và Lưu Ý

1. **L-DeepSC sau kênh**: Classification accuracy giảm mạnh sau khi qua kênh (77.63% -> 18.38%) do nhiễu kênh
2. **FuzSemCom reconstruction**: RMSE cao (10.0) do trade-off với compression ratio
3. **SNR**: Thí nghiệm chỉ chạy ở SNR 10.0 dB, có thể thử các giá trị khác
4. **Dataset**: Kết quả chỉ áp dụng cho Agriculture Sensor Data, có thể khác với dataset khác

## Cách Sử Dụng Kết Quả

### Để So Sánh Trong Paper

1. Sử dụng bảng so sánh trong `DETAILED_RESULTS.md`
2. Trích dẫn số liệu từ `comparison_with_fuzsemcom.json`
3. Sử dụng biểu đồ `l_deepsc_training_results.png` để minh họa

### Để Cải Thiện Model

1. Xem training history trong biểu đồ để điều chỉnh hyperparameters
2. Phân tích confusion matrix để cải thiện classification
3. Thử các giá trị SNR khác để đánh giá robustness

### Để Reproduce

1. Chạy `l_deepsc_compare.py` với cùng cấu hình
2. Đảm bảo dataset giống nhau
3. Sử dụng cùng random seed nếu cần reproducibility

## Tham Khảo

- L-DeepSC Paper: "Lite Distributed Semantic Communication System for Internet of Things" (arXiv:2007.11095)
- FuzSemCom Implementation: `jupyter/notebook.ipynb`
- L-DeepSC Code: `experiments/l_deepSC.py`
- Comparison Script: `experiments/l_deepsc_compare.py`

## Liên Hệ và Đóng Góp

Nếu có câu hỏi hoặc muốn cải thiện thí nghiệm:
1. Kiểm tra code trong `experiments/l_deepsc_compare.py`
2. Xem chi tiết kết quả trong `DETAILED_RESULTS.md`
3. Điều chỉnh hyperparameters trong script nếu cần

## Lịch Sử Cập Nhật

- 2025-11-30: Tạo thí nghiệm so sánh ban đầu
- 2025-11-30: Thêm classification metrics
- 2025-11-30: Thêm efficiency metrics và so sánh toàn diện

