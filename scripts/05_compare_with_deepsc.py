"""Step 5 – Compare FuzSemCom with simulated L-DeepSC results."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "results" / "reports"
FIG_DIR = PROJECT_ROOT / "results" / "figures"
TEST_DATA = PROJECT_ROOT / "data" / "processed" / "semantic_dataset_test.csv"
FSE_RESULTS = REPORTS_DIR / "fse_evaluation_results.json"


class LDeepSCSimulator:
    def __init__(self) -> None:
        self.class_names = [
            "optimal",
            "nutrient_deficiency",
            "fungal_risk",
            "water_deficit_acidic",
            "water_deficit_alkaline",
            "acidic_soil",
            "alkaline_soil",
            "heat_stress",
        ]
        self.class_accuracy = {
            0: 0.95,
            1: 0.88,
            2: 0.90,
            3: 0.85,
            4: 0.87,
            5: 0.92,
            6: 0.91,
            7: 0.89,
        }
        self.confusion_map = {
            0: [1, 5, 6],
            1: [0, 2],
            2: [7, 1],
            3: [5, 4],
            4: [6, 3],
            5: [3, 0],
            6: [4, 0],
            7: [2, 0],
        }

    def simulate(self, y_true: np.ndarray) -> np.ndarray:
        rng = np.random.default_rng(42)
        preds: list[int] = []
        for true_label in y_true:
            acc = self.class_accuracy.get(int(true_label), 0.88)
            if rng.random() < acc:
                preds.append(int(true_label))
            else:
                errors = self.confusion_map.get(int(true_label), list(range(len(self.class_names))))
                errors = [e for e in errors if e != int(true_label)]
                if not errors:
                    errors = list(range(len(self.class_names)))
                preds.append(int(rng.choice(errors)))
        return np.array(preds)


def ensure_dirs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple[pd.DataFrame, dict]:
    if not TEST_DATA.exists():
        raise FileNotFoundError("Test dataset missing. Run Step 3 first.")
    df = pd.read_csv(TEST_DATA)
    if not FSE_RESULTS.exists():
        raise FileNotFoundError("FSE results missing. Run Step 4 first.")
    with open(FSE_RESULTS, "r", encoding="utf-8") as f:
        fse_metrics = json.load(f)
    return df, fse_metrics


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "classification_report": classification_report(
            y_true, y_pred, labels=list(range(8)), output_dict=True
        ),
    }


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(8)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds")
    plt.title("Simulated L-DeepSC Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "deepsc_confusion_matrix.png", dpi=300)
    plt.close()


def save_results(fse_metrics: dict, deepsc_metrics: dict) -> None:
    comparison = {
        "fuzsemcom": {
            "accuracy": fse_metrics["accuracy"],
            "f1_macro": fse_metrics["f1_macro"],
            "precision_macro": fse_metrics["precision_macro"],
            "recall_macro": fse_metrics["recall_macro"],
        },
        "l_deepsc": {
            "accuracy": deepsc_metrics["accuracy"],
            "f1_macro": deepsc_metrics["f1_macro"],
            "precision_macro": deepsc_metrics["precision_macro"],
            "recall_macro": deepsc_metrics["recall_macro"],
        },
        "diff": {
            "accuracy": deepsc_metrics["accuracy"] - fse_metrics["accuracy"],
            "f1_macro": deepsc_metrics["f1_macro"] - fse_metrics["f1_macro"],
        },
    }
    (REPORTS_DIR / "deepsc_comparison_results.json").write_text(
        json.dumps(comparison, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    ensure_dirs()
    df, fse_metrics = load_data()
    y_true = df["ground_truth_id"].to_numpy()
    simulator = LDeepSCSimulator()
    y_pred = simulator.simulate(y_true)
    deepsc_metrics = compute_metrics(y_true, y_pred)
    save_confusion_matrix(y_true, y_pred)
    save_results(fse_metrics, deepsc_metrics)
    print("Comparison complete. Results stored in results/reports/.")


if __name__ == "__main__":
    main()
