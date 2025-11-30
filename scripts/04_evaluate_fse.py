"""Step 4 – Evaluate the Fuzzy Semantic Encoder."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from fuzzy_engine import FuzzyEngine, SEMANTIC_CLASSES

TEST_DATA = PROJECT_ROOT / "data" / "processed" / "semantic_dataset_test.csv"
REPORTS_DIR = PROJECT_ROOT / "results" / "reports"
FIG_DIR = PROJECT_ROOT / "results" / "figures"


def ensure_dirs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    if not TEST_DATA.exists():
        raise FileNotFoundError(
            f"Cannot find {TEST_DATA}. Please run scripts/03_generate_ground_truth.py first."
        )
    return pd.read_csv(TEST_DATA)


def run_inference(df: pd.DataFrame) -> pd.DataFrame:
    engine = FuzzyEngine()
    predictions: list[int] = []
    confidences: list[float] = []
    symbol_bytes_list: list[bytes] = []
    for row in df.itertuples(index=False):
        pred = engine.predict(
            moisture=getattr(row, "Moisture"),
            ph=getattr(row, "pH"),
            nitrogen=getattr(row, "N"),
            temperature=getattr(row, "Temperature"),
            humidity=getattr(row, "Humidity"),
            ndi_label=getattr(row, "NDI_Label", None),
            pdi_label=getattr(row, "PDI_Label", None),
        )
        predictions.append(pred.class_id)
        confidences.append(pred.confidence)
        symbol_bytes_list.append(pred.symbol_bytes)
    df = df.copy()
    df["prediction_id"] = predictions
    df["prediction_label"] = df["prediction_id"].map(lambda idx: SEMANTIC_CLASSES[idx])
    df["confidence"] = confidences
    df["symbol_bytes"] = symbol_bytes_list
    df["symbol_bytes_hex"] = df["symbol_bytes"].apply(lambda b: b.hex())
    return df


def compute_metrics(df: pd.DataFrame) -> dict:
    y_true = df["ground_truth_id"].to_numpy()
    y_pred = df["prediction_id"].to_numpy()
    n_samples = len(df)
    payload_bytes_per_sample = 2  # FuzSemCom: 2 bytes per symbol
    total_payload_bytes = n_samples * payload_bytes_per_sample
    
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "average_confidence": float(df["confidence"].mean()),
        "payload": {
            "bytes_per_sample": payload_bytes_per_sample,
            "total_samples": n_samples,
            "total_payload_bytes": total_payload_bytes,
            "total_payload_kb": round(total_payload_bytes / 1024, 2),
        },
    }
    metrics["classification_report"] = classification_report(
        y_true,
        y_pred,
        target_names=SEMANTIC_CLASSES,
        output_dict=True,
    )
    return metrics


def save_confusion_matrix(df: pd.DataFrame) -> None:
    y_true = df["ground_truth_id"].to_numpy()
    y_pred = df["prediction_id"].to_numpy()
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(SEMANTIC_CLASSES))))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=SEMANTIC_CLASSES, yticklabels=SEMANTIC_CLASSES)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.title("FSE Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fse_confusion_matrix.png", dpi=300)
    plt.close()


def save_reports(metrics: dict) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "fse_evaluation_results.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    payload = metrics["payload"]
    lines = [
        "FUZZY SEMANTIC ENCODER EVALUATION",
        "=================================",
        f"Accuracy:          {metrics['accuracy']*100:.2f}%",
        f"F1-macro:          {metrics['f1_macro']:.4f}",
        f"Precision (macro): {metrics['precision_macro']:.4f}",
        f"Recall (macro):     {metrics['recall_macro']:.4f}",
        f"Avg Confidence:     {metrics['average_confidence']:.4f}",
        "",
        "SEMANTIC PAYLOAD (2-byte encoding)",
        "----------------------------------",
        f"Bytes per sample:   {payload['bytes_per_sample']} bytes",
        f"Total samples:      {payload['total_samples']}",
        f"Total payload:      {payload['total_payload_bytes']} bytes ({payload['total_payload_kb']} KB)",
    ]
    (REPORTS_DIR / "fse_evaluation_report.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    df = load_dataset()
    df = run_inference(df)
    metrics = compute_metrics(df)
    save_confusion_matrix(df)
    save_reports(metrics)
    
    # Save predictions with symbol_bytes to CSV
    output_cols = ["prediction_id", "prediction_label", "confidence", "symbol_bytes_hex", "ground_truth_id"]
    if "ground_truth_label" in df.columns:
        output_cols.append("ground_truth_label")
    output_df = df[output_cols].copy()
    output_df.to_csv(REPORTS_DIR / "fse_predictions_with_symbols.csv", index=False)
    
    print("FSE evaluation complete. Results stored in results/reports.")
    print(f"  - Predictions with 2-byte symbols: results/reports/fse_predictions_with_symbols.csv")


if __name__ == "__main__":
    main()
