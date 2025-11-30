"""
Semantic Communication System
=============================
He thong Semantic Communication hoan chinh ket hop:
- FuzSemCom Encoder (Fuzzy Semantic Encoder)
- Channel Models (AWGN, Rayleigh, Rician, LoRa)
- Neural Decoder

Luong xu ly:
1. Sensor Data -> FuzSemCom Encoder -> Semantic Symbol (ID + Confidence)
2. Semantic Symbol -> Modulation -> Channel -> Demodulation
3. Received Symbol -> Neural Decoder -> Reconstructed Data
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Optional, List

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from channel_simulation.models import AWGNChannel, RayleighChannel, RicianChannel, LoRaChannel
from channel_simulation.utils.modulation import (
    bpsk_modulate, bpsk_demodulate,
    qpsk_modulate, qpsk_demodulate,
    semantic_to_bits, bits_to_semantic
)
from channel_simulation.utils.metrics import PerformanceMetrics, calculate_ber, calculate_semantic_accuracy


# ============================================================
# SEMANTIC CLASSES (giong FuzSemCom)
# ============================================================

SEMANTIC_CLASSES = [
    "optimal",
    "nutrient_deficiency", 
    "fungal_risk",
    "water_deficit_acidic",
    "water_deficit_alkaline",
    "acidic_soil",
    "alkaline_soil",
    "heat_stress"
]


# ============================================================
# FUZZY SEMANTIC ENCODER (simplified version)
# ============================================================

class FuzzySemanticEncoder:
    """
    Fuzzy Semantic Encoder - Ma hoa du lieu cam bien thanh semantic symbols
    
    Input: 5 sensors (Moisture, pH, N, Temperature, Humidity)
    Output: Semantic ID (0-7) + Confidence (0-1)
    """
    
    def __init__(self):
        # Define fuzzy rules (simplified)
        self.rules = self._define_rules()
    
    def _define_rules(self) -> Dict:
        """Define fuzzy membership functions and rules"""
        return {
            'moisture': {'low': (0, 20), 'medium': (15, 40), 'high': (35, 100)},
            'ph': {'acidic': (0, 6.5), 'neutral': (6.0, 7.5), 'alkaline': (7.0, 14)},
            'n': {'low': (0, 40), 'medium': (30, 70), 'high': (60, 100)},
            'temperature': {'low': (0, 20), 'medium': (18, 32), 'high': (28, 50)},
            'humidity': {'low': (0, 40), 'medium': (35, 70), 'high': (65, 100)}
        }
    
    def _fuzzy_membership(self, value: float, range_tuple: Tuple[float, float]) -> float:
        """Calculate fuzzy membership value"""
        low, high = range_tuple
        if value <= low:
            return 1.0 if low == range_tuple[0] else 0.0
        elif value >= high:
            return 1.0 if high == range_tuple[1] else 0.0
        else:
            mid = (low + high) / 2
            if value < mid:
                return (value - low) / (mid - low)
            else:
                return (high - value) / (high - mid)
    
    def encode(self, sensor_data: np.ndarray) -> Tuple[int, float]:
        """
        Encode sensor data to semantic symbol
        
        Parameters:
        -----------
        sensor_data : np.ndarray
            [Moisture, pH, N, Temperature, Humidity]
            
        Returns:
        --------
        symbol_id : int
            Semantic class ID (0-7)
        confidence : float
            Confidence score (0-1)
        """
        moisture, ph, n, temp, humidity = sensor_data
        
        # Simple rule-based classification
        # This is a simplified version - real FuzSemCom uses more complex fuzzy logic
        
        if moisture < 20 and ph < 6.5:
            symbol_id = 3  # water_deficit_acidic
            confidence = 0.8
        elif moisture < 20 and ph >= 7.0:
            symbol_id = 4  # water_deficit_alkaline
            confidence = 0.8
        elif ph < 5.5:
            symbol_id = 5  # acidic_soil
            confidence = 0.85
        elif ph > 8.0:
            symbol_id = 6  # alkaline_soil
            confidence = 0.85
        elif temp > 35:
            symbol_id = 7  # heat_stress
            confidence = 0.9
        elif humidity > 80 and temp > 25:
            symbol_id = 2  # fungal_risk
            confidence = 0.75
        elif n < 30:
            symbol_id = 1  # nutrient_deficiency
            confidence = 0.7
        else:
            symbol_id = 0  # optimal
            confidence = 0.6 + 0.3 * min(1, (50 - abs(moisture - 30)) / 50)
        
        return symbol_id, min(1.0, max(0.0, confidence))
    
    def encode_batch(self, sensor_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Encode batch of sensor data"""
        n_samples = len(sensor_data)
        symbol_ids = np.zeros(n_samples, dtype=int)
        confidences = np.zeros(n_samples)
        
        for i in range(n_samples):
            symbol_ids[i], confidences[i] = self.encode(sensor_data[i])
        
        return symbol_ids, confidences


# ============================================================
# SEMANTIC COMMUNICATION SYSTEM
# ============================================================

class SemanticCommSystem:
    """
    He thong Semantic Communication hoan chinh
    
    Components:
    - Semantic Encoder (FuzSemCom)
    - Modulator (BPSK/QPSK)
    - Channel (AWGN/Rayleigh/Rician/LoRa)
    - Demodulator
    - Semantic Decoder
    """
    
    def __init__(self, 
                 channel_type: str = 'awgn',
                 modulation: str = 'bpsk',
                 snr_db: float = 10.0,
                 **channel_kwargs):
        """
        Parameters:
        -----------
        channel_type : str
            'awgn', 'rayleigh', 'rician', 'lora'
        modulation : str
            'bpsk', 'qpsk'
        snr_db : float
            Signal-to-Noise Ratio in dB
        **channel_kwargs : dict
            Additional channel parameters
        """
        self.channel_type = channel_type
        self.modulation = modulation
        self.snr_db = snr_db
        
        # Initialize encoder
        self.encoder = FuzzySemanticEncoder()
        
        # Initialize channel
        self.channel = self._create_channel(channel_type, snr_db, **channel_kwargs)
        
        # Metrics
        self.metrics = PerformanceMetrics()
    
    def _create_channel(self, channel_type: str, snr_db: float, **kwargs):
        """Create channel based on type"""
        if channel_type.lower() == 'awgn':
            return AWGNChannel(snr_db=snr_db)
        elif channel_type.lower() == 'rayleigh':
            return RayleighChannel(snr_db=snr_db, **kwargs)
        elif channel_type.lower() == 'rician':
            return RicianChannel(snr_db=snr_db, **kwargs)
        elif channel_type.lower() == 'lora':
            return LoRaChannel(snr_db=snr_db, **kwargs)
        else:
            raise ValueError(f"Unknown channel type: {channel_type}")
    
    def set_snr(self, snr_db: float):
        """Update SNR"""
        self.snr_db = snr_db
        self.channel.set_snr(snr_db)
    
    def _modulate(self, bits: np.ndarray) -> np.ndarray:
        """Modulate bits to symbols"""
        if self.modulation.lower() == 'bpsk':
            return bpsk_modulate(bits)
        elif self.modulation.lower() == 'qpsk':
            return qpsk_modulate(bits)
        else:
            raise ValueError(f"Unknown modulation: {self.modulation}")
    
    def _demodulate(self, symbols: np.ndarray) -> np.ndarray:
        """Demodulate symbols to bits"""
        if self.modulation.lower() == 'bpsk':
            return bpsk_demodulate(symbols)
        elif self.modulation.lower() == 'qpsk':
            return qpsk_demodulate(symbols)
        else:
            raise ValueError(f"Unknown modulation: {self.modulation}")
    
    def transmit_semantic(self, sensor_data: np.ndarray) -> Tuple[int, float, dict]:
        """
        Transmit sensor data through semantic communication system
        
        Parameters:
        -----------
        sensor_data : np.ndarray
            Single sample [Moisture, pH, N, Temperature, Humidity]
            
        Returns:
        --------
        rx_symbol_id : int
            Received semantic class ID
        rx_confidence : float
            Received confidence
        info : dict
            Transmission information
        """
        # 1. Semantic Encoding
        tx_symbol_id, tx_confidence = self.encoder.encode(sensor_data)
        
        # 2. Convert to bits (8 bits ID + 8 bits confidence = 16 bits = 2 bytes)
        tx_bits = semantic_to_bits(tx_symbol_id, tx_confidence)
        
        # 3. Modulation
        tx_symbols = self._modulate(tx_bits)
        
        # 4. Channel transmission
        rx_symbols, channel_info = self.channel.transmit(tx_symbols)
        
        # 5. Equalization (for fading channels)
        if hasattr(self.channel, 'equalize') and self.channel.h is not None:
            rx_symbols = self.channel.equalize(rx_symbols, method='mmse')
        
        # 6. Demodulation
        rx_bits = self._demodulate(rx_symbols)
        
        # 7. Convert back to semantic
        rx_symbol_id, rx_confidence = bits_to_semantic(rx_bits)
        
        # Clip to valid range
        rx_symbol_id = int(np.clip(rx_symbol_id, 0, len(SEMANTIC_CLASSES) - 1))
        rx_confidence = float(np.clip(rx_confidence, 0, 1))
        
        # Calculate BER
        ber = calculate_ber(tx_bits, rx_bits)
        
        info = {
            'tx_symbol_id': tx_symbol_id,
            'tx_confidence': tx_confidence,
            'rx_symbol_id': rx_symbol_id,
            'rx_confidence': rx_confidence,
            'tx_class': SEMANTIC_CLASSES[tx_symbol_id],
            'rx_class': SEMANTIC_CLASSES[rx_symbol_id],
            'correct': tx_symbol_id == rx_symbol_id,
            'ber': ber,
            'channel_info': channel_info
        }
        
        return rx_symbol_id, rx_confidence, info
    
    def transmit_batch(self, sensor_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Transmit batch of sensor data
        
        Returns:
        --------
        rx_symbol_ids : np.ndarray
        rx_confidences : np.ndarray
        summary : dict
        """
        n_samples = len(sensor_data)
        
        # Encode all
        tx_symbol_ids, tx_confidences = self.encoder.encode_batch(sensor_data)
        
        rx_symbol_ids = np.zeros(n_samples, dtype=int)
        rx_confidences = np.zeros(n_samples)
        
        total_ber = 0.0
        correct = 0
        
        for i in range(n_samples):
            rx_id, rx_conf, info = self.transmit_semantic(sensor_data[i])
            rx_symbol_ids[i] = rx_id
            rx_confidences[i] = rx_conf
            total_ber += info['ber']
            if info['correct']:
                correct += 1
        
        summary = {
            'n_samples': n_samples,
            'semantic_accuracy': correct / n_samples,
            'avg_ber': total_ber / n_samples,
            'channel_type': self.channel_type,
            'modulation': self.modulation,
            'snr_db': self.snr_db
        }
        
        return rx_symbol_ids, rx_confidences, summary
    
    def evaluate_snr_range(self, sensor_data: np.ndarray, 
                           snr_range: List[float]) -> Dict:
        """
        Evaluate system over range of SNR values
        
        Returns:
        --------
        results : dict
            SNR -> metrics mapping
        """
        results = {
            'snr_db': [],
            'semantic_accuracy': [],
            'ber': []
        }
        
        for snr in snr_range:
            self.set_snr(snr)
            _, _, summary = self.transmit_batch(sensor_data)
            
            results['snr_db'].append(snr)
            results['semantic_accuracy'].append(summary['semantic_accuracy'])
            results['ber'].append(summary['avg_ber'])
        
        return results
    
    def __repr__(self):
        return (f"SemanticCommSystem(channel={self.channel_type}, "
                f"mod={self.modulation}, SNR={self.snr_db}dB)")


# ============================================================
# MAIN - Demo
# ============================================================

def main():
    print("=" * 70)
    print("SEMANTIC COMMUNICATION SYSTEM - Demo")
    print("=" * 70)
    
    # Generate sample sensor data
    np.random.seed(42)
    n_samples = 1000
    
    sensor_data = np.column_stack([
        np.random.uniform(5, 50, n_samples),    # Moisture
        np.random.uniform(4, 9, n_samples),     # pH
        np.random.uniform(10, 90, n_samples),   # N
        np.random.uniform(15, 40, n_samples),   # Temperature
        np.random.uniform(30, 95, n_samples)    # Humidity
    ])
    
    print(f"\nGenerated {n_samples} sensor samples")
    
    # Test different channels
    channels = ['awgn', 'rayleigh', 'rician']
    snr_range = [0, 5, 10, 15, 20, 25, 30]
    
    results = {}
    
    for ch_type in channels:
        print(f"\n{'='*50}")
        print(f"Testing {ch_type.upper()} Channel")
        print(f"{'='*50}")
        
        if ch_type == 'rician':
            system = SemanticCommSystem(channel_type=ch_type, snr_db=10, k_factor_db=3)
        else:
            system = SemanticCommSystem(channel_type=ch_type, snr_db=10)
        
        ch_results = system.evaluate_snr_range(sensor_data, snr_range)
        results[ch_type] = ch_results
        
        print(f"\n{'SNR (dB)':<10} {'Semantic Acc':<15} {'BER':<15}")
        print("-" * 40)
        for i, snr in enumerate(snr_range):
            acc = ch_results['semantic_accuracy'][i]
            ber = ch_results['ber'][i]
            print(f"{snr:<10} {acc*100:>12.2f}% {ber:>15.6f}")
    
    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = output_dir / f"semcom_results_{timestamp}.json"
    
    # Convert numpy arrays to lists for JSON
    results_json = {}
    for ch, data in results.items():
        results_json[ch] = {k: [float(v) for v in vals] for k, vals in data.items()}
    
    with open(results_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"\n\nResults saved to: {results_path}")
    
    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)


if __name__ == "__main__":
    main()

