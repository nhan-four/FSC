"""Test script để chạy các phần chính của notebook và so sánh kết quả."""
import sys
from pathlib import Path
import json

# Thiết lập đường dẫn
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from fuzzy_engine import FuzzyEngine, SEMANTIC_CLASSES

NOTEBOOK_OUTPUT_DIR = PROJECT_ROOT / "results" / "notebook_outputs"
NOTEBOOK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("TEST NOTEBOOK LOGIC - So sánh với kết quả hiện có")
print("=" * 60)

# Test 1: Load và kiểm tra dữ liệu
print("\n[TEST 1] Kiểm tra dữ liệu test set...")
TEST_DATA = PROJECT_ROOT / "data" / "processed" / "semantic_dataset_test.csv"
if TEST_DATA.exists():
    df_test = pd.read_csv(TEST_DATA)
    print(f"✓ Load test data: {len(df_test)} samples")
else:
    print(f"❌ Không tìm thấy test data: {TEST_DATA}")
    sys.exit(1)

# Test 2: FSE Evaluation
print("\n[TEST 2] Chạy FSE Evaluation...")
engine = FuzzyEngine()
predictions = []
confidences = []

for idx, row in df_test.iterrows():
    pred = engine.predict(
        moisture=row["Moisture"],
        ph=row["pH"],
        nitrogen=row["N"],
        temperature=row["Temperature"],
        humidity=row["Humidity"],
        ndi_label=row.get("NDI_Label"),
        pdi_label=row.get("PDI_Label"),
    )
    predictions.append(pred.class_id)
    confidences.append(pred.confidence)

y_true = df_test["ground_truth_id"].values
y_pred = np.array(predictions)

accuracy = accuracy_score(y_true, y_pred)
f1_macro = f1_score(y_true, y_pred, average="macro")
precision_macro = precision_score(y_true, y_pred, average="macro")
recall_macro = recall_score(y_true, y_pred, average="macro")
avg_confidence = np.mean(confidences)

notebook_metrics = {
    "accuracy": float(accuracy),
    "f1_macro": float(f1_macro),
    "precision_macro": float(precision_macro),
    "recall_macro": float(recall_macro),
    "average_confidence": float(avg_confidence),
}

print(f"  Accuracy: {accuracy*100:.2f}%")
print(f"  F1-macro: {f1_macro:.4f}")
print(f"  Precision-macro: {precision_macro:.4f}")
print(f"  Recall-macro: {recall_macro:.4f}")
print(f"  Avg Confidence: {avg_confidence:.4f}")

# So sánh với kết quả đã có
EXISTING_RESULTS = PROJECT_ROOT / "results" / "reports" / "fse_evaluation_results.json"
if EXISTING_RESULTS.exists():
    print("\n[TEST 3] So sánh với kết quả đã có...")
    with open(EXISTING_RESULTS, 'r') as f:
        existing_metrics = json.load(f)
    
    print("\nSo sánh Metrics:")
    print("-" * 60)
    for key in ["accuracy", "f1_macro", "precision_macro", "recall_macro", "average_confidence"]:
        if key in existing_metrics:
            existing_val = existing_metrics[key]
            notebook_val = notebook_metrics[key]
            diff = abs(existing_val - notebook_val)
            match = "✓" if diff < 0.001 else "✗"
            if key == "accuracy":
                print(f"{key:20s}: Existing={existing_val*100:6.2f}% | Notebook={notebook_val*100:6.2f}% | Diff={diff*100:.4f}% {match}")
            else:
                print(f"{key:20s}: Existing={existing_val:6.4f} | Notebook={notebook_val:6.4f} | Diff={diff:.6f} {match}")
    
    # Kiểm tra payload
    if "payload" in existing_metrics:
        n_samples = len(df_test)
        payload_bytes = n_samples * 2
        existing_payload = existing_metrics["payload"]["total_payload_bytes"]
        print(f"\nPayload comparison:")
        print(f"  Existing: {existing_payload} bytes")
        print(f"  Notebook: {payload_bytes} bytes")
        print(f"  Match: {'✓' if existing_payload == payload_bytes else '✗'}")
else:
    print(f"\n⚠️ Không tìm thấy kết quả đã có: {EXISTING_RESULTS}")

# Lưu kết quả notebook
print("\n[TEST 4] Lưu kết quả notebook...")
metrics_file = NOTEBOOK_OUTPUT_DIR / "fse_metrics.json"
with open(metrics_file, 'w', encoding='utf-8') as f:
    json.dump(notebook_metrics, f, indent=2)
print(f"✓ Đã lưu: {metrics_file}")

# Test encode/decode
print("\n[TEST 5] Test Encode/Decode 2-byte...")
test_cases = [
    (0, 1.0),
    (1, 0.85),
    (2, 0.5),
    (3, 0.25),
    (7, 0.0),
]

all_pass = True
for class_id, confidence in test_cases:
    symbol_bytes = FuzzyEngine.encode_to_bytes(class_id, confidence)
    decoded_class_id, decoded_confidence = FuzzyEngine.decode_from_bytes(symbol_bytes)
    is_correct = (class_id == decoded_class_id) and (abs(confidence - decoded_confidence) < 0.01)
    if not is_correct:
        all_pass = False
        print(f"  ✗ Class {class_id}, confidence {confidence:.3f}")

if all_pass:
    print("  ✓ Tất cả test cases đều PASS")

# Payload stats
n_samples = len(df_test)
payload_stats = {
    "bytes_per_sample": 2,
    "total_samples": n_samples,
    "total_payload_bytes": n_samples * 2,
    "total_payload_kb": round(n_samples * 2 / 1024, 2),
}

payload_file = NOTEBOOK_OUTPUT_DIR / "fse_payload_stats.json"
with open(payload_file, 'w', encoding='utf-8') as f:
    json.dump(payload_stats, f, indent=2)
print(f"✓ Đã lưu payload stats: {payload_file}")

print("\n" + "=" * 60)
print("TEST HOÀN TẤT")
print("=" * 60)
print(f"\nTất cả kết quả đã được lưu vào: {NOTEBOOK_OUTPUT_DIR}")

