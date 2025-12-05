# L-DeepSC System-Level Optimization

## Tổng Quan

Phiên bản tối ưu **TOÀN HỆ THỐNG** L-DeepSC, không chỉ tập trung vào classification.

### Vấn đề của các phiên bản trước

| Version | Accuracy Before | Accuracy After | Vấn đề |
|---------|-----------------|----------------|--------|
| l_deepsc_compare | 77.63% | 18.38% | Giảm 59% sau kênh |
| l_deepsc_optimized | 71.62% | 17.61% | Không cải thiện |
| l_deepsc_classification | ~72% | ~18% | Chỉ focus classification |

**Nguyên nhân chính**: Channel robustness kém - model không được train để chịu được nhiễu kênh.

## Cải Tiến Trong Phiên Bản Này

### 1. Channel-Aware Training
- Train với **nhiều mức SNR** (0, 5, 10, 15, 20, 25, 30 dB)
- Model học cách xử lý nhiễu ở nhiều mức khác nhau
- Robust hơn với điều kiện kênh thay đổi

### 2. Repetition Coding
- Mỗi symbol được truyền **3 lần** (configurable)
- Receiver kết hợp (soft combining) để giảm lỗi
- Trade-off: tăng bandwidth nhưng tăng reliability

### 3. Improved ADNet với Self-Attention
- Thêm **Multi-head Self-Attention** cho CSI estimation
- Residual blocks với BatchNorm
- Cải thiện khả năng denoise

### 4. Joint Loss Function
```
Loss = λ_recon * L_recon + λ_before * L_class_before + λ_after * L_class_after + λ_consist * L_consistency
```
- **L_recon**: Reconstruction loss (MSE)
- **L_class_before**: Classification loss trước kênh
- **L_class_after**: Classification loss SAU kênh (quan trọng!)
- **L_consistency**: z vs z_hat consistency

### 5. SNR-Adaptive Processing
- SNR embedding trong channel decoder
- Điều chỉnh equalization theo mức nhiễu

## Cách Chạy

```bash
cd experiments/l_deepsc_system_opt
python l_deepsc_system_optimization.py
```

**Output:**
- `results/l_deepsc_system_opt_results.json`: Kết quả chi tiết theo SNR
- `results/l_deepsc_system_opt_results.png`: Biểu đồ
- `results/l_deepsc_system_opt_model.pth`: Model weights

## Kết Quả Mong Đợi

So với các phiên bản trước:
- **Accuracy sau kênh tăng đáng kể** (mục tiêu: >50% ở SNR=10dB)
- **Khoảng cách giữa "before" và "after" giảm** (model robust hơn)
- **Reconstruction quality duy trì** (RMSE không tăng quá nhiều)

## Cấu Hình

```python
# Training
NUM_EPOCHS = 150
BATCH_SIZE = 64
LR = 1e-3

# Architecture
LATENT_DIM = 32
CHAN_DIM = 32
HIDDEN_DIM = 128
NUM_REPETITIONS = 3

# Channel-aware
SNR_RANGE = [0, 5, 10, 15, 20, 25, 30]

# Loss weights
LAMBDA_RECON = 0.5
LAMBDA_CLASS_BEFORE = 0.3
LAMBDA_CLASS_AFTER = 0.5
LAMBDA_CONSISTENCY = 0.2
```

## So Sánh Với FuzSemCom

| Metric | L-DeepSC System Opt | FuzSemCom |
|--------|---------------------|-----------|
| Accuracy (SNR=10dB) | TBD | 99.4% |
| Bandwidth | 384 bytes | 2 bytes |
| Channel Robustness | Improved | Excellent |

## Tham Khảo

- Paper gốc: "Lite Distributed Semantic Communication System for Internet of Things" (arXiv:2007.11095)
- FuzSemCom: ICC 2026

