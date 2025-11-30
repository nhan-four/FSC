"""Step 6 – Simple Bayesian optimization over heuristic biases."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from fuzzy_engine import FuzzyEngine

TRAIN_DATA = PROJECT_ROOT / "data" / "processed" / "semantic_dataset_train.csv"
TEST_DATA = PROJECT_ROOT / "data" / "processed" / "semantic_dataset_test.csv"
FULL_DATA = PROJECT_ROOT / "data" / "processed" / "semantic_dataset_fuzzy.csv"
REPORTS_DIR = PROJECT_ROOT / "results" / "reports"
FIG_DIR = PROJECT_ROOT / "results" / "figures"


def load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run Step 3 first.")
    return pd.read_csv(path)


def evaluate_params(df: pd.DataFrame, moisture_bias: float, humidity_bias: float) -> float:
    engine = FuzzyEngine()
    hits = 0
    total = 0
    for row in df.itertuples(index=False):
        pred = engine.predict(
            moisture=float(np.clip(getattr(row, "Moisture") + moisture_bias, 0, 100)),
            ph=getattr(row, "pH"),
            nitrogen=getattr(row, "N"),
            temperature=getattr(row, "Temperature"),
            humidity=float(np.clip(getattr(row, "Humidity") + humidity_bias, 0, 100)),
            ndi_label=getattr(row, "NDI_Label", None),
            pdi_label=getattr(row, "PDI_Label", None),
        )
        if pred.class_name == getattr(row, "ground_truth_label"):
            hits += 1
        total += 1
    return hits / max(total, 1)


def run_optimization(df: pd.DataFrame) -> dict:
    history: List[dict] = []

    def objective(params: list[float]) -> float:
        moisture_bias, humidity_bias = params
        acc = evaluate_params(df, moisture_bias, humidity_bias)
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
        "moisture_bias": result.x[0],
        "humidity_bias": result.x[1],
        "accuracy": best_acc,
        "history": history,
    }
    return best_params


def plot_convergence(history: List[dict]) -> None:
    xs = list(range(1, len(history) + 1))
    accs = [entry["accuracy"] for entry in history]
    plt.figure(figsize=(6, 4))
    plt.plot(xs, accs, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Validation Accuracy")
    plt.title("Bayesian Optimization Convergence")
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIG_DIR / "bo_convergence.png", dpi=200)
    plt.close()


def save_results(results: dict) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "bo_best_params.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    plot_convergence(results["history"])


def evaluate_on_datasets(bias: tuple[float, float], df_train: pd.DataFrame) -> dict:
    moisture_bias, humidity_bias = bias
    df_test = load_dataframe(TEST_DATA)
    df_full = load_dataframe(FULL_DATA)
    metrics = {
        "train": {
            "baseline": evaluate_params(df_train, 0.0, 0.0),
            "tuned": evaluate_params(df_train, moisture_bias, humidity_bias),
        },
        "test": {
            "baseline": evaluate_params(df_test, 0.0, 0.0),
            "tuned": evaluate_params(df_test, moisture_bias, humidity_bias),
        },
        "full_dataset": {
            "baseline": evaluate_params(df_full, 0.0, 0.0),
            "tuned": evaluate_params(df_full, moisture_bias, humidity_bias),
        },
    }
    return metrics


def main() -> None:
    df_train = load_dataframe(TRAIN_DATA)
    # Use 10% subset as validation for speed
    df_val = df_train.sample(frac=0.1, random_state=42).reset_index(drop=True)
    baseline_val_acc = evaluate_params(df_val, 0.0, 0.0)
    best = run_optimization(df_val)
    tuned_bias = (best["moisture_bias"], best["humidity_bias"])
    metrics = evaluate_on_datasets(tuned_bias, df_train)
    results = {
        "validation": {
            "baseline_accuracy": baseline_val_acc,
            "tuned_accuracy": best["accuracy"],
        },
        "train": metrics["train"],
        "test": metrics["test"],
        "full_dataset": metrics["full_dataset"],
        "best_params": {
            "moisture_bias": best["moisture_bias"],
            "humidity_bias": best["humidity_bias"],
        },
        "history": best["history"],
    }
    save_results(results)
    print("Ablation/BO study complete. Results stored in results/reports.")


if __name__ == "__main__":
    main()
