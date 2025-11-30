"""Step 1 – Exploratory Data Analysis for FuzSemCom dataset."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = PROJECT_ROOT / "data" / "raw" / "Agriculture_dataset_with_metadata.csv"
EDA_DIR = PROJECT_ROOT / "results" / "eda"
FIG_DIR = EDA_DIR / "figures"
REPORT_DIR = EDA_DIR / "reports"

SENSOR_COLUMNS = ["Moisture", "pH", "N", "Temperature", "Humidity"]
LABEL_COLUMNS = ["NDI_Label", "PDI_Label", "Semantic_Tag"]


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    if not RAW_DATA.exists():
        raise FileNotFoundError(f"Raw dataset not found at {RAW_DATA}")
    df = pd.read_csv(RAW_DATA)
    return df


def summarize_dataset(df: pd.DataFrame) -> dict:
    summary = {
        "shape": df.shape,
        "columns": df.columns.tolist(),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
    }
    missing = df.isna().sum()
    summary["missing"] = missing[missing > 0].sort_values(ascending=False).to_dict()
    stats = df[SENSOR_COLUMNS].describe().to_dict()
    summary["stats"] = stats
    return summary


def plot_missing(df: pd.DataFrame) -> None:
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    plt.figure(figsize=(10, 4))
    sns.barplot(x=missing.values, y=missing.index, palette="Reds_r")
    plt.title("Missing Value Analysis")
    plt.xlabel("Count")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "missing_values.png", dpi=200)
    plt.close()


def plot_sensor_distributions(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(len(SENSOR_COLUMNS), 2, figsize=(10, 15))
    for idx, col in enumerate(SENSOR_COLUMNS):
        sns.histplot(df[col].dropna(), ax=axes[idx, 0], kde=True, color="steelblue")
        axes[idx, 0].set_title(f"Histogram – {col}")
        sns.boxplot(x=df[col].dropna(), ax=axes[idx, 1], color="orange")
        axes[idx, 1].set_title(f"Boxplot – {col}")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "sensor_distributions.png", dpi=200)
    plt.close()


def plot_correlation(df: pd.DataFrame) -> None:
    corr = df[SENSOR_COLUMNS].corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Sensor Correlation Matrix")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "correlation_matrix.png", dpi=200)
    plt.close()


def plot_label_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 4))
    melted = (
        df[LABEL_COLUMNS]
        .melt(var_name="label_type", value_name="label")
        .dropna()
    )
    sns.countplot(data=melted, x="label", hue="label_type")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "label_distribution.png", dpi=200)
    plt.close()


def write_report(summary: dict) -> None:
    lines = [
        "EXPLORATORY DATA ANALYSIS REPORT",
        "============================================================",
        f"Dataset path: {RAW_DATA}",
        f"Shape: {summary['shape'][0]} rows × {summary['shape'][1]} columns",
        f"Memory usage: {summary['memory_mb']} MB",
        "",
        "Missing values:",
    ]
    if summary["missing"]:
        for col, cnt in summary["missing"].items():
            percent = cnt / summary["shape"][0] * 100
            lines.append(f"  - {col}: {cnt} ({percent:.2f}%)")
    else:
        lines.append("  None")
    lines.append("")
    lines.append("Sensor statistics:")
    for col in SENSOR_COLUMNS:
        stats = summary["stats"].get(col, {})
        if stats:
            lines.append(
                f"  {col}: min={stats.get('min', 'NA'):.2f}, max={stats.get('max', 'NA'):.2f}, "
                f"mean={stats.get('mean', 'NA'):.2f}, std={stats.get('std', 'NA'):.2f}"
            )
    (REPORT_DIR / "eda_report.txt").write_text("\n".join(lines), encoding="utf-8")
    (REPORT_DIR / "eda_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    ensure_dirs()
    df = load_dataset()
    summary = summarize_dataset(df)
    plot_missing(df)
    plot_sensor_distributions(df)
    plot_correlation(df)
    plot_label_distribution(df)
    write_report(summary)
    print("EDA complete. Outputs saved to results/eda/")


if __name__ == "__main__":
    main()
