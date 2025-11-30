"""Demo script: Encode/Decode 2-byte semantic symbols."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.fuzzy_engine import FuzzyEngine, SEMANTIC_CLASSES


def demo_encode_decode() -> None:
    """Demo encode/decode 2-byte semantic symbols."""
    engine = FuzzyEngine()
    
    print("=" * 60)
    print("DEMO: 2-BYTE SEMANTIC SYMBOL ENCODE/DECODE")
    print("=" * 60)
    print()
    
    # Test cases: different classes and confidences
    test_cases = [
        (0, 1.0),   # optimal, full confidence
        (1, 0.85),  # nutrient_deficiency, high confidence
        (2, 0.5),   # fungal_risk, medium confidence
        (3, 0.25),  # water_deficit_acidic, low confidence
        (7, 0.0),   # heat_stress, zero confidence
    ]
    
    print("Test Cases:")
    print("-" * 60)
    for class_id, confidence in test_cases:
        class_name = SEMANTIC_CLASSES[class_id]
        
        # Encode
        symbol_bytes = engine.encode_to_bytes(class_id, confidence)
        
        # Decode
        decoded_class_id, decoded_confidence = engine.decode_from_bytes(symbol_bytes)
        
        # Verify
        is_correct = (class_id == decoded_class_id) and (
            abs(confidence - decoded_confidence) < 0.01
        )
        
        print(f"Class: {class_name:25s} (ID={class_id})")
        print(f"  Original:    confidence={confidence:.3f}")
        print(f"  Encoded:     {symbol_bytes.hex()} (hex) = {list(symbol_bytes)} (bytes)")
        print(f"  Decoded:     class_id={decoded_class_id}, confidence={decoded_confidence:.3f}")
        print(f"  Verification: {'✓ PASS' if is_correct else '✗ FAIL'}")
        print()
    
    # Test with actual predictions
    print("=" * 60)
    print("Test with Real FSE Predictions:")
    print("-" * 60)
    
    test_inputs = [
        {
            "moisture": 45.0,
            "ph": 6.3,
            "nitrogen": 75.0,
            "temperature": 24.0,
            "humidity": 65.0,
            "description": "Optimal conditions",
        },
        {
            "moisture": 20.0,
            "ph": 5.0,
            "nitrogen": 30.0,
            "temperature": 24.0,
            "humidity": 65.0,
            "description": "Nutrient deficiency + acidic",
        },
        {
            "moisture": 70.0,
            "ph": 7.5,
            "nitrogen": 80.0,
            "temperature": 18.0,
            "humidity": 85.0,
            "description": "Fungal risk conditions",
        },
    ]
    
    for i, inputs in enumerate(test_inputs, 1):
        pred = engine.predict(**{k: v for k, v in inputs.items() if k != "description"})
        symbol_bytes = pred.symbol_bytes
        
        # Decode to verify
        decoded_class_id, decoded_confidence = engine.decode_from_bytes(symbol_bytes)
        
        print(f"Test {i}: {inputs['description']}")
        print(f"  Prediction:  {pred.class_name} (ID={pred.class_id})")
        print(f"  Confidence:  {pred.confidence:.3f}")
        print(f"  Symbol:      {symbol_bytes.hex()} (hex) = {list(symbol_bytes)} (bytes)")
        print(f"  Decoded:     class_id={decoded_class_id}, confidence={decoded_confidence:.3f}")
        print(f"  Match:       {'✓' if pred.class_id == decoded_class_id and abs(pred.confidence - decoded_confidence) < 0.01 else '✗'}")
        print()
    
    print("=" * 60)
    print("Summary:")
    print("  - Each semantic symbol is exactly 2 bytes")
    print("  - Byte 0: class_id (0-7)")
    print("  - Byte 1: quantized confidence (0-255, representing 0.0-1.0)")
    print("  - Total payload: 2 bytes per sample")
    print("=" * 60)


if __name__ == "__main__":
    demo_encode_decode()

