# Jupyter Notebook - FuzSemCom Pipeline

Folder này chứa Jupyter notebook để chạy toàn bộ pipeline với visualization trực quan.

## Files

- `notebook.ipynb` - Notebook chính chứa toàn bộ pipeline
- `test_notebook.py` - Script test để verify logic notebook

## Cách sử dụng

### Chạy Notebook

```bash
# Từ thư mục gốc project
cd jupyter
jupyter notebook notebook.ipynb

# Hoặc với JupyterLab
jupyter lab notebook.ipynb
```

### Chạy Notebook Logic (không cần Jupyter)

```bash
# Chạy toàn bộ pipeline và lưu tất cả kết quả
python run_notebook.py
```

Script này sẽ:
- ✅ Chạy toàn bộ pipeline (Step 1-6)
- ✅ Tạo tất cả visualizations (PNG files)
- ✅ Lưu tất cả metrics và reports (JSON files)
- ✅ Tự động lưu vào `results/notebook_outputs/`

### Test Logic (verify kết quả)

```bash
python test_notebook.py
```

Script này sẽ:
- Chạy FSE evaluation
- So sánh kết quả với `results/reports/fse_evaluation_results.json`
- Test encode/decode 2-byte
- Verify metrics khớp 100%

## Kết quả

Tất cả kết quả từ notebook được lưu vào:
```
../results/notebook_outputs/
```

Bao gồm:
- `fse_metrics.json` - Metrics chính (accuracy, F1, precision, recall)
- `fse_payload_stats.json` - Thống kê 2-byte payload
- `fse_classification_report.json` - Per-class metrics
- `*.png` - Tất cả visualizations (confusion matrix, distributions, etc.)
- `deepsc_comparison.json` - So sánh với L-DeepSC
- `bo_results.json` - Bayesian Optimization results

## Verification

Kết quả từ notebook đã được verify và khớp 100% với các scripts:
- ✅ Accuracy: 94.99%
- ✅ F1-macro: 0.9148
- ✅ Precision-macro: 0.9246
- ✅ Recall-macro: 0.9152
- ✅ Payload: 16,378 bytes (15.99 KB)

