"""Utility script to inspect a single sample through the FuzzyEngine."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))

from fuzzy_engine import FuzzyEngine, SEMANTIC_CLASSES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug a prediction by index.")
    parser.add_argument(
        "--dataset",
        default=PROJECT_ROOT / "data" / "processed" / "semantic_dataset_fuzzy.csv",
        type=Path,
        help="CSV file to inspect",
    )
    parser.add_argument("--index", type=int, default=0, help="Row index to inspect")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.dataset.exists():
        raise FileNotFoundError(args.dataset)
    df = pd.read_csv(args.dataset)
    if args.index < 0 or args.index >= len(df):
        raise IndexError(f"Index {args.index} outside dataset size {len(df)}")
    row = df.iloc[args.index]
    engine = FuzzyEngine()
    pred = engine.predict(
        moisture=row["Moisture"],
        ph=row["pH"],
        nitrogen=row["N"],
        temperature=row["Temperature"],
        humidity=row["Humidity"],
        ndi_label=row.get("NDI_Label"),
        pdi_label=row.get("PDI_Label"),
    )
    print("Sample features:")
    print(row[SENSOR_COLUMNS := ["Moisture", "pH", "N", "Temperature", "Humidity"]])
    print("\nFuzzy prediction:")
    print(f"class: {pred.class_name} (id={pred.class_id}) confidence={pred.confidence:.3f}")
    print("Rule strengths:")
    for cls in SEMANTIC_CLASSES:
        print(f"  {cls:<24}: {pred.rule_strengths.get(cls, 0):.3f}")


if __name__ == "__main__":
    main()
