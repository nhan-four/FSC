from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import pandas as pd


@dataclass
class GroundTruthGenerator:
    def generate(self, df: pd.DataFrame) -> List[int]:
        return self._create_ground_truth_labels(df)

    def _create_ground_truth_labels(self, df: pd.DataFrame) -> List[int]:
        labels: List[int] = []
        for _, row in df.iterrows():
            labels.append(self._classify_row(row))
        return labels

    def _classify_row(self, row: pd.Series) -> int:
        moisture = row.get("Moisture")
        ph = row.get("pH")
        nitrogen = row.get("N")
        temperature = row.get("Temperature")
        humidity = row.get("Humidity")
        ndi = row.get("NDI_Label")
        pdi = row.get("PDI_Label")

        if (
            30 <= moisture <= 60
            and 6.0 <= ph <= 6.8
            and 50 <= nitrogen <= 100
            and 22 <= temperature <= 26
            and 60 <= humidity <= 70
        ):
            return 0
        if ndi == "High":
            return 1
        if pdi == "High" and humidity > 80 and temperature < 22:
            return 2
        if moisture < 30 and ph < 5.8:
            return 3
        if moisture < 30 and ph > 7.5:
            return 4
        if ph < 5.8 and moisture >= 30:
            return 5
        if ph > 7.5 and moisture >= 30:
            return 6
        if temperature > 30 and humidity < 60:
            return 7
        return -1


def save_ground_truth(df: pd.DataFrame, labels: Iterable[int], output_path: str) -> pd.DataFrame:
    df_out = df.copy()
    df_out["ground_truth"] = list(labels)
    df_labeled = df_out[df_out["ground_truth"] != -1]
    df_labeled.to_csv(output_path, index=False)
    return df_labeled
