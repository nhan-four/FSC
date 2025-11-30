# Channel Simulation for FuzSemCom

## Tong Quan

Module mo phong kenh truyen cho he thong **FuzSemCom** (Fuzzy Semantic Communication), bao gom:

### Kenh Truyen (4 loai)
1. **AWGN Channel**: Kenh nhieu Gaussian trang cong
2. **Rayleigh Fading Channel**: Kenh fading Rayleigh (NLOS - Non Line of Sight)
3. **Rician Fading Channel**: Kenh fading Rician (LOS + NLOS)
4. **LoRa/LoRaWAN Channel**: Kenh mo phong cho IoT (dac thu cho FuzSemCom)

### Encoder (2 loai)
1. **FuzSemCom Encoder**: Fuzzy Semantic Encoder - phuong phap de xuat (94.99% accuracy)
2. **Simple Semantic Encoder**: Baseline encoder don gian

## Cau Truc Thu Muc

```
channel_simulation/
├── README.md                       # File nay
├── models/
│   ├── __init__.py
│   ├── awgn_channel.py             # AWGN channel
│   ├── rayleigh_channel.py         # Rayleigh fading
│   ├── rician_channel.py           # Rician fading
│   ├── lora_channel.py             # LoRa/LoRaWAN channel
│   └── composite_channel.py        # Ket hop nhieu kenh
├── utils/
│   ├── __init__.py
│   ├── modulation.py               # BPSK, QPSK, QAM
│   └── metrics.py                  # BER, SER, Accuracy
├── semantic_comm_system.py         # He thong SemCom co ban
├── fuzsemcom_channel_system.py     # He thong FuzSemCom hoan chinh
├── monte_carlo_analysis.py         # Monte Carlo simulation
└── results/                        # Ket qua mo phong
    ├── fuzsemcom_channel_results_*.json
    ├── fuzsemcom_channel_comparison_*.png
    └── monte_carlo_*.json
```

## Cach Chay

### 1. Chay FuzSemCom Channel System (Khuyen nghi)
```bash
cd channel_simulation
python fuzsemcom_channel_system.py
```

**Output:**
- So sanh 4 kenh (AWGN, Rayleigh, Rician, LoRa)
- Su dung FuzSemCom encoder thuc
- Luu ket qua JSON va bieu do PNG

### 2. Chay Monte Carlo Analysis
```bash
python monte_carlo_analysis.py --num_trials 100 --num_samples 500
```

### 3. Chay Semantic Comm System (Baseline)
```bash
python semantic_comm_system.py
```

## Ket Qua

### FuzSemCom Semantic Accuracy (%)

| SNR (dB) | AWGN | Rayleigh | Rician | LoRa |
|----------|------|----------|--------|------|
| 0 | 29.60 | 45.75 | 40.85 | 12.90 |
| 5 | 73.95 | 72.90 | 71.60 | 18.00 |
| 10 | 99.40 | 89.55 | 92.25 | 37.50 |
| 15 | 100.00 | 96.80 | 97.50 | 71.75 |
| 20 | 100.00 | 99.05 | 99.30 | 94.20 |
| 25 | 100.00 | 99.70 | 99.60 | 98.90 |
| 30 | 100.00 | 99.95 | 99.85 | 99.90 |

### Phan Tich

1. **AWGN**: Kenh ly tuong, dat 100% accuracy tu SNR >= 15dB
2. **Rayleigh/Rician**: Kenh fading, can SNR cao hon (~20-25dB) de dat 99%+
3. **LoRa**: Kenh IoT thuc te, can SNR >= 20dB de dat 94%+

### So Sanh Voi Paper

| Metric | Paper | Simulation |
|--------|-------|------------|
| FuzSemCom Accuracy | 94.99% | 100% (SNR >= 15dB, AWGN) |
| Bandwidth | 2 bytes | 2 bytes |
| Latency | Low | Low |

## Mo Hinh Kenh

### 1. AWGN Channel
```
y = x + n
n ~ N(0, sigma^2)
SNR = 10 * log10(P_signal / P_noise)
```

### 2. Rayleigh Fading
```
y = h * x + n
h ~ CN(0, 1)
|h| ~ Rayleigh distribution
```

### 3. Rician Fading
```
y = h * x + n
h = sqrt(K/(K+1)) * h_LOS + sqrt(1/(K+1)) * h_NLOS
K: Rician K-factor (LOS/NLOS power ratio)
```

### 4. LoRa Channel
```
- Path loss: Log-distance model
- Shadowing: Log-normal
- Small-scale fading: Rayleigh
- Parameters: SF, BW, Distance, Tx Power
```

## Tham So

### AWGN
- `snr_db`: SNR (dB)

### Rayleigh
- `snr_db`: SNR (dB)
- `num_taps`: So tap multipath (1 = flat fading)
- `doppler_freq`: Tan so Doppler (Hz)

### Rician
- `snr_db`: SNR (dB)
- `k_factor_db`: Rician K-factor (dB)

### LoRa
- `distance`: Khoang cach (m)
- `sf`: Spreading Factor (7-12)
- `bw`: Bandwidth (125kHz, 250kHz, 500kHz)
- `tx_power`: Cong suat phat (dBm)

## Tham Khao

- FuzSemCom Paper: ICC 2026
- L-DeepSC Paper: arXiv:2007.11095
- LoRa Channel Model: IEEE papers on LoRaWANA
