# Experiments - Baseline Models & L-DeepSC

## Tong Quan

Thu muc nay chua cac thi nghiem so sanh:
1. **Baseline Models**: Cac model ML truyen thong (RF, SVM, XGBoost, ...)
2. **L-DeepSC**: Lite Deep Semantic Communication (baseline deep learning)
3. **L-DeepSC Optimized**: Cac phien ban toi uu cua L-DeepSC

## Cau Truc

```
experiments/
├── README.md                       # File nay
├── baseline_models.py              # 13 baseline ML models
├── baseline_results/               # Ket qua baseline
│   ├── baseline_comparison_*.json
│   ├── baseline_comparison_*.csv
│   ├── baseline_comparison.png
│   └── baseline_radar.png
│
├── l_deepSC.py                     # L-DeepSC original (tu paper)
├── l_deepsc_compare.py             # L-DeepSC so sanh voi FuzSemCom
│
├── l_deepsc_optimized/             # Cac phien ban toi uu
│   ├── l_deepsc_optimized.py       # V1 - Basic optimization
│   ├── l_deepsc_v4_optimized.py    # V4 - Focal Loss + Class weights
│   ├── l_deepsc_v5_ensemble.py     # V5 - Ensemble + Oversampling
│   ├── l_deepsc_v6_extended_features.py  # V6 - Extended features
│   ├── l_deepsc_v7_final.py        # V7 - Final version
│   └── results_*/                  # Ket qua tung version
│
├── l_deepsc_system_opt/            # System-level optimization (channel-aware)
│   ├── l_deepsc_system_optimization.py   # Train channel-aware, repetition coding
│   └── results/                    # Ket qua system-opt
│
└── 2007.11095v3.pdf                # L-DeepSC paper
```

## Cach Chay

### 1. Baseline Models
```bash
cd experiments
pip install xgboost lightgbm -q  # Optional
python baseline_models.py
```

**Output:**
- `baseline_results/baseline_comparison_*.json`: Ket qua JSON
- `baseline_results/baseline_comparison_*.csv`: Ket qua CSV
- `baseline_results/baseline_comparison.png`: Bieu do so sanh
- `baseline_results/baseline_radar.png`: Radar chart

### 2. L-DeepSC Comparison
```bash
python l_deepsc_compare.py
```

**Output:**
- `results/l_deepsc_comparison/l_deepsc_results.json`
- `results/l_deepsc_comparison/comparison_with_fuzsemcom.json`
- `results/l_deepsc_comparison/l_deepsc_training_results.png`

### 2b. L-DeepSC System-Level Optimization (moi, channel-aware)
```bash
cd l_deepsc_system_opt
python l_deepsc_system_optimization.py
```

**Output:**
- `experiments/l_deepsc_system_opt/results/l_deepsc_system_opt_results.json`
- `experiments/l_deepsc_system_opt/results/l_deepsc_system_opt_results.png`
- `experiments/l_deepsc_system_opt/results/l_deepsc_system_opt_model.pth`

**Ghi chu nhanh:**
- Train channel-aware (SNR sweep 0–30 dB) + repetition coding (3x)
- Accuracy sau kenh @10 dB ~74% (tang ro rang so voi ~18% ban dau), nhung bandwidth cao (~384 bytes)

### 3. L-DeepSC Optimized
```bash
cd l_deepsc_optimized

# Chay phien ban final (khuyen nghi)
python l_deepsc_v7_final.py

# Hoac cac phien ban khac
python l_deepsc_v4_optimized.py
python l_deepsc_v5_ensemble.py
```

## Baseline Models

| # | Model | Mo Ta |
|---|-------|-------|
| 1 | Decision Tree | Cay quyet dinh voi class weights |
| 2 | Random Forest | 200 trees, balanced |
| 3 | Extra Trees | Extremely Randomized Trees |
| 4 | Gradient Boosting | Sklearn GradientBoosting |
| 5 | AdaBoost | Adaptive Boosting |
| 6 | XGBoost | Extreme Gradient Boosting |
| 7 | LightGBM | Light Gradient Boosting |
| 8 | SVM (RBF) | Support Vector Machine - RBF kernel |
| 9 | SVM (Linear) | Support Vector Machine - Linear kernel |
| 10 | KNN | K-Nearest Neighbors (k=5) |
| 11 | Naive Bayes | Gaussian Naive Bayes |
| 12 | Logistic Regression | Multinomial |
| 13 | MLP | Multi-Layer Perceptron (128-64-32) |

## Ket Qua

### So Sanh Accuracy

| Model | Accuracy | F1 (macro) | Notes |
|-------|----------|------------|-------|
| **FuzSemCom** | **94.99%** | **0.9148** | Proposed method |
| L-DeepSC | 77.63% | 0.6329 | Original paper |
| L-DeepSC V7 | ~71% | ~0.62 | Optimized |
| Random Forest | ~85% | ~0.82 | Best baseline |
| XGBoost | ~86% | ~0.83 | Best baseline |
| Decision Tree | ~77% | ~0.76 | |
| MLP | ~77% | ~0.66 | |

### Phan Tich

1. **FuzSemCom thang** ve:
   - Classification accuracy (+17% so voi L-DeepSC)
   - Bandwidth efficiency (2 bytes vs 64 bytes)
   - Computational efficiency (CPU vs GPU)

2. **L-DeepSC thang** ve:
   - Reconstruction quality (RMSE 4.02 vs 10.0)

3. **Baseline ML** (RF, XGBoost):
   - Accuracy cao hon L-DeepSC (~85% vs ~77%)
   - Nhung khong phai semantic communication

## Luu Y

1. **Data split**: Tat ca experiments su dung cung train/test split:
   - Test size: 20%
   - Random state: 42
   - Stratified

2. **Class imbalance**: Dataset co class imbalance nghiem trong:
   - `acidic_soil`: 27 samples
   - `water_deficit_alkaline`: 9317 samples
   - Cac experiments su dung class weights va oversampling

3. **L-DeepSC limitations**:
   - Accuracy thap do class imbalance
   - Reconstruction tot nhung classification kem
   - Bandwidth cao (64 bytes vs 2 bytes cua FuzSemCom)

## Tham Khao

- L-DeepSC Paper: `2007.11095v3.pdf`
- FuzSemCom: `../jupyter/notebook.ipynb`

