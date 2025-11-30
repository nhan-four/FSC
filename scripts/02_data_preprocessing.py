"""Step 2 – Clean and preprocess the FuzSemCom dataset."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = PROJECT_ROOT / "data" / "raw" / "Agriculture_dataset_with_metadata.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "semantic_dataset_preprocessed.csv"
STATS_FILE = PROJECT_ROOT / "data" / "processed" / "preprocessing_stats.txt"

SENSOR_COLUMNS = ["Moisture", "pH", "N", "Temperature", "Humidity"]
LABEL_COLUMNS = ["NDI_Label", "PDI_Label"]

VALID_RANGES = {
    "Moisture": (0, 100),
    "pH": (4.0, 9.0),
    "N": (0, 300),
    "Temperature": (10, 45),
    "Humidity": (20, 100),
}

SEMANTIC_CLASSES = [
    "optimal",
    "water_deficit_acidic",
    "water_deficit_alkaline",
    "nutrient_deficiency",
    "fungal_risk",
    "acidic_soil",
    "alkaline_soil",
    "heat_stress",
    "other",
]


def load_dataset() -> pd.DataFrame:
    if not RAW_DATA.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_DATA}")
    return pd.read_csv(RAW_DATA)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=SENSOR_COLUMNS)
    after_dropna = len(df)
    for col, (low, high) in VALID_RANGES.items():
        df = df[(df[col] >= low) & (df[col] <= high)]
    after_range = len(df)
    df = df.reset_index(drop=True)
    df.attrs["stats"] = {
        "original_samples": before,
        "after_dropna": after_dropna,
        "after_range": after_range,
    }
    return df


def classify_semantic(row: pd.Series) -> str:
    moisture = row["Moisture"]
    ph = row["pH"]
    nitrogen = row["N"]
    temperature = row["Temperature"]
    humidity = row["Humidity"]
    ndi = row.get("NDI_Label")
    pdi = row.get("PDI_Label")

    if (
        30 <= moisture <= 60
        and 6.0 <= ph <= 6.8
        and 50 <= nitrogen <= 100
        and 22 <= temperature <= 26
        and 60 <= humidity <= 70
    ):
        return "optimal"
    if moisture < 30 and ph < 5.8:
        return "water_deficit_acidic"
    if moisture < 30 and ph > 7.5:
        return "water_deficit_alkaline"
    if ndi == "High" or (nitrogen < 40 and ph < 5.8):
        return "nutrient_deficiency"
    if pdi == "High" or (humidity > 80 and temperature < 22):
        return "fungal_risk"
    if ph < 5.8 and moisture >= 30:
        return "acidic_soil"
    if ph > 7.5 and moisture >= 30:
        return "alkaline_soil"
    if temperature > 30 and humidity < 60:
        return "heat_stress"
    return "other"


def write_stats(df: pd.DataFrame, cleaned: pd.DataFrame) -> None:
    stats = cleaned.attrs.get("stats", {})
    lines = [
        "DATA PREPROCESSING SUMMARY",
        "==========================",
        f"Original samples: {stats.get('original_samples')}",
        f"After dropna:     {stats.get('after_dropna')}",
        f"After range check:{stats.get('after_range')}",
        "",
        "Semantic distribution:",
    ]
    distribution = (
        cleaned["semantic_label"]
        .value_counts()
        .reindex(SEMANTIC_CLASSES, fill_value=0)
    )
    total = len(cleaned)
    for cls, count in distribution.items():
        percent = count / total * 100 if total else 0
        lines.append(f"  - {cls:<24}: {count:>6} ({percent:5.2f}%)")
    STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATS_FILE.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = load_dataset()
    cleaned = clean_dataset(df)
    cleaned["semantic_label"] = cleaned.apply(classify_semantic, axis=1)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(OUTPUT_CSV, index=False)
    write_stats(df, cleaned)
    print("Preprocessing complete.")
    print(f"Saved cleaned dataset to {OUTPUT_CSV}")
    print(f"Stats written to {STATS_FILE}")


if __name__ == "__main__":
    main()
