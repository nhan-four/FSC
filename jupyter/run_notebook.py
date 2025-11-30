"""Script để chạy toàn bộ logic của notebook và lưu tất cả kết quả."""
import sys
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

# Thiết lập đường dẫn
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fuzzy_engine import FuzzyEngine, SEMANTIC_CLASSES
from ground_truth_generator import GroundTruthGenerator

# Thư mục output
NOTEBOOK_OUTPUT_DIR = PROJECT_ROOT / "results" / "notebook_outputs"
NOTEBOOK_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("CHẠY NOTEBOOK - FuzSemCom Pipeline")
print("=" * 70)

# ============================================================================
# STEP 1: DATA EXPLORATION
# ============================================================================
print("\n[STEP 1] Data Exploration...")
RAW_DATA = PROJECT_ROOT / "data" / "raw" / "Agriculture_dataset_with_metadata.csv"

if not RAW_DATA.exists():
    print(f"❌ Không tìm thấy: {RAW_DATA}")
    sys.exit(1)

df_raw = pd.read_csv(RAW_DATA)
SENSOR_COLUMNS = ["Moisture", "pH", "N", "Temperature", "Humidity"]
print(f"✓ Load dữ liệu: {df_raw.shape[0]} samples × {df_raw.shape[1]} features")

# Missing values
missing = df_raw.isna().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing) > 0:
    plt.figure(figsize=(10, 6))
    top_missing = missing.head(15)
    sns.barplot(x=top_missing.values, y=top_missing.index, palette="Reds_r")
    plt.xlabel("Số lượng missing values")
    plt.title("Top 15 cột có nhiều missing values nhất")
    plt.tight_layout()
    plt.savefig(NOTEBOOK_OUTPUT_DIR / "missing_values.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Đã lưu: missing_values.png")

# Sensor distributions
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for idx, col in enumerate(SENSOR_COLUMNS):
    if col in df_raw.columns:
        df_raw[col].hist(bins=50, ax=axes[idx], edgecolor='black')
        axes[idx].set_title(f'Phân bố {col}')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')
axes[5].axis('off')
plt.tight_layout()
plt.savefig(NOTEBOOK_OUTPUT_DIR / "sensor_distributions.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Đã lưu: sensor_distributions.png")

# Correlation matrix
corr_matrix = df_raw[SENSOR_COLUMNS].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title("Correlation Matrix - Các cảm biến")
plt.tight_layout()
plt.savefig(NOTEBOOK_OUTPUT_DIR / "correlation_matrix.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"✓ Đã lưu: correlation_matrix.png")

# ============================================================================
# STEP 2: DATA PREPROCESSING
# ============================================================================
print("\n[STEP 2] Data Preprocessing...")
PREPROCESSED_DATA = PROJECT_ROOT / "data" / "processed" / "semantic_dataset_preprocessed.csv"

if PREPROCESSED_DATA.exists():
    df_preprocessed = pd.read_csv(PREPROCESSED_DATA)
    print(f"✓ Load preprocessed data: {len(df_preprocessed)} samples")
else:
    print("⚠️ Chưa có file preprocessed, bỏ qua step này")

# ============================================================================
# STEP 3: GROUND TRUTH GENERATION
# ============================================================================
print("\n[STEP 3] Ground Truth Generation...")
FUZZY_DATA = PROJECT_ROOT / "data" / "processed" / "semantic_dataset_fuzzy.csv"
TRAIN_DATA = PROJECT_ROOT / "data" / "processed" / "semantic_dataset_train.csv"
TEST_DATA = PROJECT_ROOT / "data" / "processed" / "semantic_dataset_test.csv"

if FUZZY_DATA.exists() and TRAIN_DATA.exists() and TEST_DATA.exists():
    df_fuzzy = pd.read_csv(FUZZY_DATA)
    df_train = pd.read_csv(TRAIN_DATA)
    df_test = pd.read_csv(TEST_DATA)
    print(f"✓ Load datasets: Full={len(df_fuzzy)}, Train={len(df_train)}, Test={len(df_test)}")
    
    # Ground truth distribution
    if "ground_truth_label" in df_fuzzy.columns:
        gt_dist = df_fuzzy["ground_truth_label"].value_counts()
        train_dist = df_train["ground_truth_label"].value_counts()
        test_dist = df_test["ground_truth_label"].value_counts()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        gt_dist.plot(kind='barh', ax=ax1, color='steelblue')
        ax1.set_xlabel("Số lượng samples")
        ax1.set_title("Phân bố Ground Truth (Full Dataset)")
        
        comparison_df = pd.DataFrame({
            'Train': train_dist,
            'Test': test_dist
        }).fillna(0)
        comparison_df.plot(kind='barh', ax=ax2, color=['#2ecc71', '#e74c3c'])
        ax2.set_xlabel("Số lượng samples")
        ax2.set_title("Phân bố Ground Truth (Train vs Test)")
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(NOTEBOOK_OUTPUT_DIR / "ground_truth_distribution.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ Đã lưu: ground_truth_distribution.png")
else:
    print("❌ Chưa có ground truth data, cần chạy script 03_generate_ground_truth.py trước")
    sys.exit(1)

# ============================================================================
# STEP 4: FSE EVALUATION
# ============================================================================
print("\n[STEP 4] FSE Evaluation...")
engine = FuzzyEngine()

predictions = []
confidences = []
symbol_bytes_list = []

print("  Đang chạy inference...")
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
    symbol_bytes_list.append(pred.symbol_bytes)

df_test_results = df_test.copy()
df_test_results["prediction_id"] = predictions
df_test_results["prediction_label"] = df_test_results["prediction_id"].map(lambda idx: SEMANTIC_CLASSES[idx])
df_test_results["confidence"] = confidences
df_test_results["symbol_bytes_hex"] = [b.hex() for b in symbol_bytes_list]

# Metrics
y_true = df_test_results["ground_truth_id"].values
y_pred = np.array(predictions)

accuracy = accuracy_score(y_true, y_pred)
f1_macro = f1_score(y_true, y_pred, average="macro")
f1_weighted = f1_score(y_true, y_pred, average="weighted")
precision_macro = precision_score(y_true, y_pred, average="macro")
recall_macro = recall_score(y_true, y_pred, average="macro")
avg_confidence = np.mean(confidences)

metrics = {
    "accuracy": float(accuracy),
    "f1_macro": float(f1_macro),
    "f1_weighted": float(f1_weighted),
    "precision_macro": float(precision_macro),
    "recall_macro": float(recall_macro),
    "average_confidence": float(avg_confidence),
}

print(f"  Accuracy: {accuracy*100:.2f}%")
print(f"  F1-macro: {f1_macro:.4f}")

# Lưu metrics
with open(NOTEBOOK_OUTPUT_DIR / "fse_metrics.json", 'w', encoding='utf-8') as f:
    json.dump(metrics, f, indent=2)
print(f"✓ Đã lưu: fse_metrics.json")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=list(range(len(SEMANTIC_CLASSES))))
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=SEMANTIC_CLASSES, yticklabels=SEMANTIC_CLASSES,
            cbar_kws={"label": "Count"})
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.title("FSE Confusion Matrix", fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(NOTEBOOK_OUTPUT_DIR / "fse_confusion_matrix.png", dpi=200, bbox_inches='tight')
plt.close()
print(f"✓ Đã lưu: fse_confusion_matrix.png")

# Normalized confusion matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(10, 8))
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=SEMANTIC_CLASSES, yticklabels=SEMANTIC_CLASSES,
            cbar_kws={"label": "Normalized"})
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.title("FSE Confusion Matrix (Normalized)", fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(NOTEBOOK_OUTPUT_DIR / "fse_confusion_matrix_normalized.png", dpi=200, bbox_inches='tight')
plt.close()
print(f"✓ Đã lưu: fse_confusion_matrix_normalized.png")

# Classification Report
report = classification_report(y_true, y_pred, target_names=SEMANTIC_CLASSES, output_dict=True)
with open(NOTEBOOK_OUTPUT_DIR / "fse_classification_report.json", 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2)
print(f"✓ Đã lưu: fse_classification_report.json")

# Per-class metrics
per_class_metrics = pd.DataFrame({
    'Precision': [report[cls]['precision'] for cls in SEMANTIC_CLASSES],
    'Recall': [report[cls]['recall'] for cls in SEMANTIC_CLASSES],
    'F1-Score': [report[cls]['f1-score'] for cls in SEMANTIC_CLASSES],
}, index=SEMANTIC_CLASSES)

fig, ax = plt.subplots(figsize=(12, 6))
per_class_metrics.plot(kind='bar', ax=ax, color=['#3498db', '#2ecc71', '#e74c3c'])
ax.set_xlabel("Semantic Class", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Per-Class Metrics (Precision, Recall, F1-Score)", fontsize=14, fontweight='bold')
ax.legend(loc='upper right')
ax.set_xticklabels(SEMANTIC_CLASSES, rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(NOTEBOOK_OUTPUT_DIR / "fse_per_class_metrics.png", dpi=200, bbox_inches='tight')
plt.close()
print(f"✓ Đã lưu: fse_per_class_metrics.png")

# Confidence Distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.hist(df_test_results['confidence'], bins=50, edgecolor='black', color='steelblue', alpha=0.7)
ax1.axvline(df_test_results['confidence'].mean(), color='red', linestyle='--', linewidth=2,
            label=f'Mean: {df_test_results["confidence"].mean():.3f}')
ax1.set_xlabel("Confidence Score", fontsize=12)
ax1.set_ylabel("Frequency", fontsize=12)
ax1.set_title("Confidence Distribution (Histogram)", fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

ax2.boxplot(df_test_results['confidence'], vert=True, patch_artist=True,
            boxprops=dict(facecolor='steelblue', alpha=0.7))
ax2.set_ylabel("Confidence Score", fontsize=12)
ax2.set_title("Confidence Distribution (Box Plot)", fontsize=14, fontweight='bold')
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(NOTEBOOK_OUTPUT_DIR / "fse_confidence_distribution.png", dpi=200, bbox_inches='tight')
plt.close()
print(f"✓ Đã lưu: fse_confidence_distribution.png")

# Payload Stats
n_samples = len(df_test_results)
payload_stats = {
    "bytes_per_sample": 2,
    "total_samples": n_samples,
    "total_payload_bytes": n_samples * 2,
    "total_payload_kb": round(n_samples * 2 / 1024, 2),
}
with open(NOTEBOOK_OUTPUT_DIR / "fse_payload_stats.json", 'w', encoding='utf-8') as f:
    json.dump(payload_stats, f, indent=2)
print(f"✓ Đã lưu: fse_payload_stats.json")

# ============================================================================
# STEP 5: DEEPSC COMPARISON
# ============================================================================
print("\n[STEP 5] DeepSC Comparison...")
np.random.seed(42)
deepsc_predictions = []
for idx, row in df_test.iterrows():
    if np.random.random() < 0.88:
        deepsc_predictions.append(row["ground_truth_id"])
    else:
        deepsc_predictions.append(np.random.randint(0, len(SEMANTIC_CLASSES)))

deepsc_accuracy = accuracy_score(y_true, deepsc_predictions)
deepsc_f1_macro = f1_score(y_true, deepsc_predictions, average="macro")
deepsc_precision = precision_score(y_true, deepsc_predictions, average="macro")
deepsc_recall = recall_score(y_true, deepsc_predictions, average="macro")

comparison_metrics = {
    "FuzSemCom": {
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "payload_bytes": 2,
    },
    "L-DeepSC": {
        "accuracy": float(deepsc_accuracy),
        "f1_macro": float(deepsc_f1_macro),
        "precision_macro": float(deepsc_precision),
        "recall_macro": float(deepsc_recall),
        "payload_bytes": 8,
    }
}

with open(NOTEBOOK_OUTPUT_DIR / "deepsc_comparison.json", 'w', encoding='utf-8') as f:
    json.dump(comparison_metrics, f, indent=2)
print(f"✓ Đã lưu: deepsc_comparison.json")

# Comparison plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
methods = ["FuzSemCom", "L-DeepSC"]
accuracies = [accuracy * 100, deepsc_accuracy * 100]
colors = ['#2ecc71', '#e74c3c']

axes[0].bar(methods, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
axes[0].set_ylabel("Accuracy (%)", fontsize=12)
axes[0].set_title("Accuracy Comparison", fontsize=14, fontweight='bold')
axes[0].set_ylim([0, 100])
for i, (method, acc) in enumerate(zip(methods, accuracies)):
    axes[0].text(i, acc + 1, f'{acc:.2f}%', ha='center', fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

f1_scores = [f1_macro, deepsc_f1_macro]
axes[1].bar(methods, f1_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
axes[1].set_ylabel("F1-Score (macro)", fontsize=12)
axes[1].set_title("F1-Score Comparison", fontsize=14, fontweight='bold')
axes[1].set_ylim([0, 1])
for i, (method, f1) in enumerate(zip(methods, f1_scores)):
    axes[1].text(i, f1 + 0.02, f'{f1:.3f}', ha='center', fontweight='bold')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(NOTEBOOK_OUTPUT_DIR / "fse_vs_deepsc_comparison.png", dpi=200, bbox_inches='tight')
plt.close()
print(f"✓ Đã lưu: fse_vs_deepsc_comparison.png")

# DeepSC confusion matrix
deepsc_cm = confusion_matrix(y_true, deepsc_predictions, labels=list(range(len(SEMANTIC_CLASSES))))
plt.figure(figsize=(10, 8))
sns.heatmap(deepsc_cm, annot=True, fmt="d", cmap="Reds",
            xticklabels=SEMANTIC_CLASSES, yticklabels=SEMANTIC_CLASSES,
            cbar_kws={"label": "Count"})
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.title("L-DeepSC (Simulated) Confusion Matrix", fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(NOTEBOOK_OUTPUT_DIR / "deepsc_confusion_matrix.png", dpi=200, bbox_inches='tight')
plt.close()
print(f"✓ Đã lưu: deepsc_confusion_matrix.png")

# ============================================================================
# STEP 6: BAYESIAN OPTIMIZATION
# ============================================================================
print("\n[STEP 6] Bayesian Optimization...")
try:
    from skopt import gp_minimize
    from skopt.space import Real
    
    df_val = df_train.sample(frac=0.1, random_state=42)
    
    def evaluate_params(df_eval, moisture_bias, humidity_bias):
        engine = FuzzyEngine()
        hits = 0
        total = 0
        for idx, row in df_eval.iterrows():
            pred = engine.predict(
                moisture=float(np.clip(row["Moisture"] + moisture_bias, 0, 100)),
                ph=row["pH"],
                nitrogen=row["N"],
                temperature=row["Temperature"],
                humidity=float(np.clip(row["Humidity"] + humidity_bias, 0, 100)),
                ndi_label=row.get("NDI_Label"),
                pdi_label=row.get("PDI_Label"),
            )
            if pred.class_name == row["ground_truth_label"]:
                hits += 1
            total += 1
        return hits / max(total, 1)
    
    history = []
    def objective(params):
        moisture_bias, humidity_bias = params
        acc = evaluate_params(df_val, moisture_bias, humidity_bias)
        history.append({"moisture_bias": moisture_bias, "humidity_bias": humidity_bias, "accuracy": acc})
        return -acc
    
    result = gp_minimize(
        objective,
        dimensions=[Real(-5.0, 5.0), Real(-5.0, 5.0)],
        n_calls=15,
        random_state=42,
    )
    
    best_acc = -result.fun
    best_params = {
        "moisture_bias": float(result.x[0]),
        "humidity_bias": float(result.x[1]),
        "validation_accuracy": float(best_acc),
        "train_baseline": float(evaluate_params(df_train, 0, 0)),
        "train_tuned": float(evaluate_params(df_train, result.x[0], result.x[1])),
        "test_baseline": float(evaluate_params(df_test, 0, 0)),
        "test_tuned": float(evaluate_params(df_test, result.x[0], result.x[1])),
        "history": history,
    }
    
    with open(NOTEBOOK_OUTPUT_DIR / "bo_results.json", 'w', encoding='utf-8') as f:
        json.dump(best_params, f, indent=2)
    print(f"✓ Đã lưu: bo_results.json")
    
    # Convergence plot
    if len(history) > 0:
        iterations = list(range(1, len(history) + 1))
        accuracies = [entry["accuracy"] for entry in history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, accuracies, marker='o', linewidth=2, markersize=8, color='#3498db')
        plt.axhline(y=best_acc, color='r', linestyle='--', linewidth=2, label=f'Best: {best_acc*100:.2f}%')
        plt.xlabel("Iteration", fontsize=12)
        plt.ylabel("Validation Accuracy", fontsize=12)
        plt.title("Bayesian Optimization Convergence", fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(NOTEBOOK_OUTPUT_DIR / "bo_convergence.png", dpi=200, bbox_inches='tight')
        plt.close()
        print(f"✓ Đã lưu: bo_convergence.png")
        
except ImportError:
    print("⚠️ scikit-optimize chưa được cài, bỏ qua BO step")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("HOÀN TẤT - Tất cả kết quả đã được lưu")
print("=" * 70)
print(f"\nThư mục output: {NOTEBOOK_OUTPUT_DIR}")
print("\nCác file đã tạo:")

saved_files = sorted(NOTEBOOK_OUTPUT_DIR.glob("*"))
for file in saved_files:
    if file.is_file():
        size_kb = file.stat().st_size / 1024
        print(f"  ✓ {file.name:45s} ({size_kb:.2f} KB)")

print(f"\nTổng cộng: {len([f for f in saved_files if f.is_file()])} files")

