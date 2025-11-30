# FuzSemCom Project - Fuzzy Semantic Communication for IoT

## Tong Quan Du An

He thong Semantic Communication cho IoT su dung Fuzzy Logic, bao gom:
- **FuzSemCom**: Fuzzy Semantic Encoder (phuong phap de xuat)
- **L-DeepSC**: Lite Deep Semantic Communication (baseline)
- **Channel Simulation**: Mo phong kenh truyen (AWGN, Rayleigh, Rician, LoRa)
- **Baseline Models**: So sanh voi cac model ML truyen thong

## Cau Truc Du An

```
fuzsemcom_project/
├── README.md                           # File nay
├── requirements.txt                    # Dependencies
│
├── jupyter/                            # Jupyter notebook
│   ├── notebook.ipynb                  # Pipeline chinh
│   └── README.md                       # Huong dan notebook
│
├── src/                                # Core implementation
│   ├── fuzzy_engine.py                 # Fuzzy Semantic Encoder
│   └── ground_truth_generator.py       # Ground truth generation
│
├── scripts/                            # Pipeline scripts
│   ├── 01_data_exploration.py
│   ├── 02_data_preprocessing.py
│   ├── 03_generate_ground_truth.py
│   ├── 04_evaluate_fse.py
│   └── ...
│
├── experiments/                        # Experiments & Baselines
│   ├── baseline_models.py              # ML baselines (RF, SVM, XGBoost, ...)
│   ├── baseline_results/               # Ket qua baseline
│   ├── l_deepSC.py                     # L-DeepSC original
│   ├── l_deepsc_compare.py             # L-DeepSC comparison
│   ├── l_deepsc_optimized/             # L-DeepSC optimized versions
│   │   ├── l_deepsc_v4_optimized.py
│   │   ├── l_deepsc_v5_ensemble.py
│   │   ├── l_deepsc_v6_extended_features.py
│   │   ├── l_deepsc_v7_final.py
│   │   └── results_*/                  # Ket qua tung version
│   └── README.md                       # Huong dan experiments
│
├── channel_simulation/                 # [MOI] Mo phong kenh truyen
│   ├── README.md                       # Huong dan chi tiet
│   ├── models/                         # Channel models
│   │   ├── awgn_channel.py             # AWGN channel
│   │   ├── rayleigh_channel.py         # Rayleigh fading
│   │   ├── rician_channel.py           # Rician fading
│   │   ├── lora_channel.py             # LoRa/LoRaWAN channel
│   │   └── composite_channel.py        # Composite channel
│   ├── utils/                          # Utilities
│   │   ├── modulation.py               # BPSK, QPSK, QAM
│   │   └── metrics.py                  # BER, SER, Accuracy
│   ├── semantic_comm_system.py         # He thong SemCom hoan chinh
│   ├── monte_carlo_analysis.py         # Monte Carlo simulation
│   └── results/                        # Ket qua mo phong
│
├── data/
│   ├── raw/                            # Dataset goc
│   └── processed/                      # Dataset da xu ly
│
├── results/                            # Ket qua tong hop
│   ├── figures/
│   ├── reports/
│   ├── l_deepsc_comparison/
│   └── notebook_outputs/
│
├── models/                             # Trained models
│   └── neural_decoder.h5
│
└── docs/                               # Tai lieu
    ├── ICC_ENGLAND_2026.pdf            # FuzSemCom Paper (ICC 2026)
    └── student_guide_2026.pdf
```

## Cac Module Chinh

### 1. FuzSemCom (Phuong phap de xuat)
- **Vi tri**: `jupyter/notebook.ipynb`, `src/fuzzy_engine.py`
- **Mo ta**: Fuzzy Semantic Encoder - ma hoa du lieu cam bien thanh semantic symbols (2 bytes)
- **Ket qua**: Accuracy ~94.99%, Bandwidth 2 bytes/sample

### 2. L-DeepSC (Baseline)
- **Vi tri**: `experiments/l_deepsc_optimized/`
- **Mo ta**: Lite Deep Semantic Communication - deep learning based
- **Ket qua**: Accuracy ~70-77%, Bandwidth 64-128 bytes/sample

### 3. Channel Simulation [MOI]
- **Vi tri**: `channel_simulation/`
- **Mo ta**: Mo phong kenh truyen cho Semantic Communication
- **Kenh ho tro**:
  - AWGN (Additive White Gaussian Noise)
  - Rayleigh Fading (NLOS)
  - Rician Fading (LOS + NLOS)
  - LoRa/LoRaWAN (IoT-specific)
- **Tinh nang**:
  - Monte Carlo simulation
  - BER/SER/Semantic Accuracy analysis
  - SNR sweep analysis

### 4. Baseline Models
- **Vi tri**: `experiments/baseline_models.py`
- **Mo ta**: So sanh voi cac model ML truyen thong
- **Models**: Decision Tree, Random Forest, XGBoost, LightGBM, SVM, KNN, MLP, ...

## Quick Start

### 1. Cai dat dependencies
```bash
pip install -r requirements.txt
```

### 2. Chay FuzSemCom (Notebook)
```bash
cd jupyter
jupyter notebook notebook.ipynb
```

### 3. Chay Channel Simulation
```bash
cd channel_simulation

# Demo he thong SemCom
python semantic_comm_system.py

# Monte Carlo analysis
python monte_carlo_analysis.py --num_trials 100 --channel all
```

### 4. Chay Baseline Models
```bash
cd experiments
python baseline_models.py
```

### 5. Chay L-DeepSC
```bash
cd experiments/l_deepsc_optimized
python l_deepsc_v7_final.py
```

## Ket Qua So Sanh

### Classification Accuracy

| Model | Accuracy | F1 (macro) | Bandwidth |
|-------|----------|------------|-----------|
| **FuzSemCom (Ours)** | **94.99%** | **0.9148** | **2 bytes** |
| L-DeepSC | 77.63% | 0.6329 | 64 bytes |
| Random Forest | ~85% | ~0.82 | N/A |
| XGBoost | ~86% | ~0.83 | N/A |
| Decision Tree | ~77% | ~0.76 | N/A |
| MLP | ~77% | ~0.66 | N/A |

### Channel Performance (SNR = 10 dB)

| Channel | Semantic Accuracy | BER |
|---------|-------------------|-----|
| AWGN | ~95% | ~0.001 |
| Rayleigh | ~85% | ~0.05 |
| Rician (K=3dB) | ~90% | ~0.02 |
| LoRa (1km) | ~80% | ~0.08 |

## Huong Dan Chi Tiet

### Channel Simulation
Xem `channel_simulation/README.md` de biet:
- Cach su dung tung loai kenh
- Cach chay Monte Carlo simulation
- Cach tich hop voi FuzSemCom

### Experiments
Xem `experiments/README.md` de biet:
- Cach chay baseline models
- Cach optimize L-DeepSC
- Cach so sanh ket qua

### Notebook
Xem `jupyter/README.md` de biet:
- Cach chay tung step trong pipeline
- Cach interpret ket qua
- Cach generate reports

## Tham Khao

- FuzSemCom Paper: `docs/ICC_ENGLAND_2026.pdf`
- L-DeepSC Paper: arXiv:2007.11095
- Dataset: IEEE DataPort - Agriculture IoT Dataset

## Lien He

- Author: [Your Name]
- Email: [Your Email]
