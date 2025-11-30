"""Step 3 – Generate ground truth dataset using fuzzy inference."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from fuzzy_engine import FuzzyEngine, SEMANTIC_CLASSES
from ground_truth_generator import GroundTruthGenerator

INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "semantic_dataset_preprocessed.csv"
FUZZY_DATASET = PROJECT_ROOT / "data" / "processed" / "semantic_dataset_fuzzy.csv"
TRAIN_DATASET = PROJECT_ROOT / "data" / "processed" / "semantic_dataset_train.csv"
TEST_DATASET = PROJECT_ROOT / "data" / "processed" / "semantic_dataset_test.csv"
STATS_FILE = PROJECT_ROOT / "data" / "processed" / "fuzzy_generation_stats.json"

SENSOR_COLUMNS = ["Moisture", "pH", "N", "Temperature", "Humidity"]


def load_dataset() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Cannot find {INPUT_CSV}. Please run scripts/02_data_preprocessing.py first."
        )
    return pd.read_csv(INPUT_CSV)


def generate_ground_truth(df: pd.DataFrame) -> pd.DataFrame:
    generator = GroundTruthGenerator()
    labels = generator.generate(df)
    df = df.copy()
    df["ground_truth_id"] = labels
    df = df[df["ground_truth_id"] >= 0].reset_index(drop=True)
    df["ground_truth_label"] = df["ground_truth_id"].map(lambda idx: SEMANTIC_CLASSES[idx])
    return df


def enrich_with_confidence(df: pd.DataFrame) -> pd.DataFrame:
    engine = FuzzyEngine()
    confidences: list[float] = []
    predictions: list[str] = []
    for row in df.itertuples(index=False):
        pred = engine.predict(
            moisture=getattr(row, "Moisture"),
            ph=getattr(row, "pH"),
            nitrogen=getattr(row, "N"),
            temperature=getattr(row, "Temperature"),
            humidity=getattr(row, "Humidity"),
            ndi_label=getattr(row, "NDI_Label", None),
            pdi_label=getattr(row, "PDI_Label", None),
            enable_thresholds=False,
            enable_fallback=True,
        )
        confidences.append(pred.confidence)
        predictions.append(pred.class_name)
    df = df.copy()
    df["fse_prediction"] = predictions
    df["confidence"] = confidences
    return df


def save_splits(df: pd.DataFrame) -> tuple[int, int]:
    stratify_col = df["ground_truth_label"] if df["ground_truth_label"].nunique() > 1 else None
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=stratify_col,
    )
    train_df.to_csv(TRAIN_DATASET, index=False)
    test_df.to_csv(TEST_DATASET, index=False)
    return len(train_df), len(test_df)


def write_stats(df: pd.DataFrame, train_len: int, test_len: int) -> None:
    stats = {
        "total_samples": len(df),
        "train_samples": train_len,
        "test_samples": test_len,
        "label_distribution": df["ground_truth_label"].value_counts().to_dict(),
        "mean_confidence": df["confidence"].mean(),
        "median_confidence": df["confidence"].median(),
    }
    STATS_FILE.write_text(json.dumps(stats, indent=2), encoding="utf-8")


def main() -> None:
    df = load_dataset()
    df = generate_ground_truth(df)
    df = enrich_with_confidence(df)
    FUZZY_DATASET.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FUZZY_DATASET, index=False)
    train_len, test_len = save_splits(df)
    write_stats(df, train_len, test_len)
    print("Ground truth dataset generated.")
    print(f"- Full dataset: {FUZZY_DATASET}")
    print(f"- Train split:  {TRAIN_DATASET}")
    print(f"- Test split:   {TEST_DATASET}")
    print(f"- Stats:        {STATS_FILE}")


if __name__ == "__main__":
    main()
