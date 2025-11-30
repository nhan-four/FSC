from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, NamedTuple


class FSEPrediction(NamedTuple):
    class_id: int
    class_name: str
    confidence: float
    rule_strengths: Dict[str, float]
    raw_class_name: str
    raw_confidence: float
    crisp_class: str
    symbol_bytes: bytes  # 2 bytes: [class_id (0-7), confidence_quantized (0-255)]


def _trimf(x: float, a: float, b: float, c: float) -> float:
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a)
    return (c - x) / (c - b)


CONFIDENCE_OVERRIDE_THRESHOLD = 0.29

CLASS_CONFIDENCE_THRESHOLDS = {
    "nutrient_deficiency": 0.90,
    "fungal_risk": 0.80,
}


SEMANTIC_CLASSES = [
    "optimal",
    "nutrient_deficiency",
    "fungal_risk",
    "water_deficit_acidic",
    "water_deficit_alkaline",
    "acidic_soil",
    "alkaline_soil",
    "heat_stress",
]

CLASS_TO_ID = {name: idx for idx, name in enumerate(SEMANTIC_CLASSES)}


@dataclass
class FuzzyEngine:
    """Tomato Fuzzy Semantic Encoder theo Table II của paper ICC 2026."""

    def predict(
        self,
        *,
        moisture: float,
        ph: float,
        nitrogen: float,
        temperature: float,
        humidity: float,
        ndi_label: str | None = None,
        pdi_label: str | None = None,
        enable_thresholds: bool = True,
        enable_fallback: bool = True,
    ) -> FSEPrediction:
        m_moisture = {
            "dry": _trimf(moisture, 15, 20, 30),
            "ideal": _trimf(moisture, 30, 45, 60),
            "wet": _trimf(moisture, 60, 70, 85),
        }
        m_ph = {
            "acidic": _trimf(ph, 4.5, 5.0, 5.8),
            "ideal": _trimf(ph, 6.0, 6.3, 6.8),
            "alkaline": _trimf(ph, 6.8, 7.5, 8.5),
        }
        m_n = {
            "low": _trimf(nitrogen, 0, 20, 40),
            "adequate": _trimf(nitrogen, 50, 80, 100),
            "high": _trimf(nitrogen, 150, 200, 250),
        }
        m_temp = {
            "cool": _trimf(temperature, 15, 18, 22),
            "ideal": _trimf(temperature, 22, 24, 26),
            "hot": _trimf(temperature, 26, 30, 38),
        }
        m_hum = {
            "dry": _trimf(humidity, 40, 50, 60),
            "ideal": _trimf(humidity, 60, 65, 70),
            "humid": _trimf(humidity, 70, 80, 90),
        }

        strengths: Dict[str, float] = {name: 0.0 for name in SEMANTIC_CLASSES}

        strengths["water_deficit_acidic"] = max(
            strengths["water_deficit_acidic"],
            min(m_moisture["dry"], m_ph["acidic"]),
        )
        strengths["water_deficit_alkaline"] = max(
            strengths["water_deficit_alkaline"],
            min(m_moisture["dry"], m_ph["alkaline"]),
        )
        strengths["acidic_soil"] = max(
            strengths["acidic_soil"],
            min(m_ph["acidic"], max(m_moisture["ideal"], m_moisture["wet"])),
        )
        strengths["alkaline_soil"] = max(
            strengths["alkaline_soil"],
            min(m_ph["alkaline"], max(m_moisture["ideal"], m_moisture["wet"])),
        )
        strengths["optimal"] = max(
            strengths["optimal"],
            min(
                m_moisture["ideal"],
                m_ph["ideal"],
                m_n["adequate"],
                m_temp["ideal"],
                m_hum["ideal"],
            ),
        )
        strengths["heat_stress"] = max(
            strengths["heat_stress"],
            min(m_temp["hot"], m_hum["dry"]),
        )
        strengths["nutrient_deficiency"] = max(
            strengths["nutrient_deficiency"],
            min(m_n["low"], m_ph["acidic"]),
        )
        strengths["fungal_risk"] = max(
            strengths["fungal_risk"],
            min(m_hum["humid"], m_temp["cool"]),
        )

        if ndi_label == "High":
            strengths["nutrient_deficiency"] = max(strengths["nutrient_deficiency"], 1.0)
        if pdi_label == "High" and humidity > 70 and temperature < 22:
            strengths["fungal_risk"] = max(strengths["fungal_risk"], 1.0)

        best_class = max(
            SEMANTIC_CLASSES,
            key=lambda name: (strengths[name], -CLASS_TO_ID[name]),
        )
        raw_confidence = strengths[best_class]

        crisp_class = self._crisp_fallback(
            moisture, ph, nitrogen, temperature, humidity, ndi_label, pdi_label
        )
        threshold = CLASS_CONFIDENCE_THRESHOLDS.get(best_class, CONFIDENCE_OVERRIDE_THRESHOLD)
        confidence = raw_confidence
        if enable_fallback and raw_confidence == 0.0:
            best_class = crisp_class
            confidence = 0.0
        elif enable_thresholds and raw_confidence < threshold:
            best_class = crisp_class
            confidence = max(raw_confidence, 1.0 if crisp_class == "optimal" else 0.7)

        class_id = CLASS_TO_ID[best_class]
        confidence_quantized = int(confidence * 255) & 0xFF  # 0-255
        symbol_bytes = bytes([class_id, confidence_quantized])

        return FSEPrediction(
            class_id=class_id,
            class_name=best_class,
            confidence=float(confidence),
            rule_strengths=strengths,
            raw_class_name=best_class,
            raw_confidence=float(raw_confidence),
            crisp_class=crisp_class,
            symbol_bytes=symbol_bytes,
        )

    @staticmethod
    def encode_to_bytes(class_id: int, confidence: float) -> bytes:
        """
        Encode FSE prediction to 2 bytes semantic symbol.
        
        Args:
            class_id: Semantic class ID (0-7)
            confidence: Confidence value (0.0-1.0)
        
        Returns:
            2 bytes: [class_id (1 byte), confidence_quantized (1 byte, 0-255)]
        """
        if not (0 <= class_id <= 7):
            raise ValueError(f"class_id must be 0-7, got {class_id}")
        confidence_quantized = int(confidence * 255) & 0xFF
        return bytes([class_id, confidence_quantized])

    @staticmethod
    def decode_from_bytes(symbol_bytes: bytes) -> tuple[int, float]:
        """
        Decode 2-byte semantic symbol back to class_id and confidence.
        
        Args:
            symbol_bytes: 2 bytes [class_id, confidence_quantized]
        
        Returns:
            (class_id, confidence) tuple
        """
        if len(symbol_bytes) != 2:
            raise ValueError(f"symbol_bytes must be 2 bytes, got {len(symbol_bytes)}")
        class_id = symbol_bytes[0]
        confidence_quantized = symbol_bytes[1]
        confidence = confidence_quantized / 255.0
        return class_id, confidence

    @staticmethod
    def _crisp_fallback(
        moisture: float,
        ph: float,
        nitrogen: float,
        temperature: float,
        humidity: float,
        ndi_label: str | None,
        pdi_label: str | None,
    ) -> str:
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
        if ph < 5.8 and moisture >= 30:
            return "acidic_soil"
        if ph > 7.5 and moisture >= 30:
            return "alkaline_soil"
        if temperature > 30 and humidity < 60:
            return "heat_stress"
        if humidity > 80 and temperature < 22:
            return "fungal_risk"
        if nitrogen < 40 and ph < 5.8:
            return "nutrient_deficiency"
        if ndi_label == "High":
            return "nutrient_deficiency"
        if pdi_label == "High":
            return "fungal_risk"
        return "optimal"
