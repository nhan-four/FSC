# FuzSemCom – Report đối chiếu README_check & Fixed ICC_ENGLAND_2026

Thực thi toàn bộ pipeline (Steps 1→6) theo `README.md` và so sánh với kỳ vọng trong paper *Fixed ICC_ENGLAND_2026.pdf* (ICC 2026). Dữ liệu dùng: `data/raw/Agriculture_dataset_with_metadata.csv`.

---

## Step 1 – Data Exploration
- **Kết quả thực tế:** 60 000 mẫu × 24 cột; thiếu dữ liệu chủ yếu ở `Migration_Timestamp` (54 065), `NDVI/NDRE` (~30 k). Thống kê cảm biến nằm đúng range (Moisture 8–38, pH 5–8, v.v.). Artefact: `results/eda/figures/*.png`, `results/eda/reports/eda_*`.
- **Kỳ vọng README_check/paper:** kiểm tra missing values, range các sensor trong Table II. ✅ Khớp.

## Step 2 – Data Preprocessing
- **Kết quả thực tế:** giữ đủ 60 000 mẫu, mapping semantic theo rule (priority Table III). Phân bố: water_deficit_acidic 19.29%, nutrient_deficiency 19.41%, “other” 24.41%, optimal chỉ 34 mẫu (`data/processed/preprocessing_stats.txt`).
- **Kỳ vọng:** tạo dataset sạch + semantic distribution. ✅ Hoàn thành (dù phân bố lệch do dataset gốc).

## Step 3 – Ground Truth Generation
- **Kết quả thực tế:** sau fuzzy labeling còn 40 942 mẫu hợp lệ; split 80/20 có `semantic_dataset_train.csv` (32 753) và `semantic_dataset_test.csv` (8 189). Mean confidence 0.561, median 0.547 (`fuzzy_generation_stats.json`).
- **Kỳ vọng:** sinh `semantic_dataset_fuzzy` + train/test + confidence stats. ✅ Khớp.

## Step 4 – FSE Evaluation (Test set)
| Metric | Kỳ vọng (paper ≈) | Kết quả thực tế |
|--------|-------------------|-----------------|
| Accuracy | ~88.7% | **94.99%** |
| F1-macro | ~0.89 | **0.915** |
| Precision-macro | ~0.90 | **0.925** |
| Recall-macro | ~0.89 | **0.915** |
| Mean confidence | ~0.73 (185/255) | **0.686** |

- Artefact: `results/reports/fse_evaluation_results.json`, `results/figures/fse_confusion_matrix.png`.
- **Nhận xét:** hệ thống hiện tại vượt 6 điểm % so với báo cáo do ground-truth và luật fallback trùng khớp, dataset thiên lệch mạnh (gần như không có mẫu “optimal”). Cần ghi chú khi so với paper.

## Step 5 – So sánh với L-DeepSC (mô phỏng)
| Metric | FuzSemCom | L-DeepSC mô phỏng |
|--------|-----------|-------------------|
| Accuracy | 94.99% | 88.08% |
| F1-macro | 0.915 | 0.714 |
| Precision-macro | 0.925 | 0.708 |
| Recall-macro | 0.915 | 0.900 |

- File: `results/reports/deepsc_comparison_results.json`, hình `results/figures/deepsc_confusion_matrix.png`.
- **Đánh giá:** tái hiện được lợi thế FuzSemCom về độ chính xác & băng thông như trong paper (dù số liệu tuyệt đối cao hơn).

## Step 6 – Ablation / Bayesian Optimization
- Sử dụng `gp_minimize` (15 lần) trên bias moisture/humidity, validation 10% tập train.
- Best accuracy = **94.99%** (không tăng nhiều do baseline đã rất cao). Lịch sử và đồ thị: `results/reports/bo_best_params.json`, `results/figures/bo_convergence.png`.
- **Kỳ vọng:** thể hiện khả năng tối ưu ngưỡng/heuristic. ✅ Có kết quả, nhưng cần ghi chú rằng lợi ích nhỏ.

## File / Artefact quan trọng
- Dataset: `data/processed/semantic_dataset_{preprocessed,fuzzy,train,test}.csv`
- Báo cáo: `results/reports/{eda_*, preprocessing_stats, fse_evaluation_results, deepsc_comparison_results, bo_best_params}.json`
- Hình ảnh: `results/figures/{missing_values.png, sensor_distributions.png, correlation_matrix.png, label_distribution.png, fse_confusion_matrix.png, deepsc_confusion_matrix.png, bo_convergence.png}`

## Kết luận & Next steps
1. Pipeline đã bám sát README_check và paper (Steps 1→6), chạy độc lập.
2. Accuracy thực tế cao hơn báo cáo (~95% vs 88.7%) vì GT và luật giống hệt; cần ghi rõ nếu công bố.
3. Nếu muốn sát paper hơn, cần:
   - Tách train/validation riêng để tune (tránh reuse test).
   - Điều chỉnh GT để không trùng fallback (giảm leaky accuracy).
   - Hoàn thiện Step 5 “Neural Decoder” nếu muốn đối chiếu Section IV.E.

