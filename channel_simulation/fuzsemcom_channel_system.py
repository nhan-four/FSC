"""
FuzSemCom Channel System
========================
He thong Semantic Communication hoan chinh su dung FuzSemCom Encoder thuc
(tu src/fuzzy_engine.py) va so sanh voi Semantic Encoder don gian.

So sanh 4 kenh:
1. AWGN
2. Rayleigh Fading
3. Rician Fading
4. LoRa/LoRaWAN

So sanh 2 encoder:
1. FuzSemCom (Fuzzy Semantic Encoder - proposed method)
2. Simple Semantic Encoder (baseline)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from channel_simulation.models import AWGNChannel, RayleighChannel, RicianChannel, LoRaChannel
from channel_simulation.utils.modulation import (
    bpsk_modulate, bpsk_demodulate,
    semantic_to_bits, bits_to_semantic
)
from channel_simulation.utils.metrics import calculate_ber, calculate_semantic_accuracy

# Import FuzSemCom encoder
try:
    from fuzzy_engine import FuzzyEngine, SEMANTIC_CLASSES
    FUZSEMCOM_AVAILABLE = True
    print("FuzSemCom encoder loaded successfully!")
except ImportError as e:
    print(f"Warning: Could not import FuzzyEngine: {e}")
    FUZSEMCOM_AVAILABLE = False
    SEMANTIC_CLASSES = [
        "optimal", "nutrient_deficiency", "fungal_risk",
        "water_deficit_acidic", "water_deficit_alkaline",
        "acidic_soil", "alkaline_soil", "heat_stress"
    ]


# ============================================================
# FUZSEMCOM ENCODER (Real Implementation)
# ============================================================

class FuzSemComEncoder:
    """
    FuzSemCom Encoder - Fuzzy Semantic Encoder tu paper ICC 2026
    Su dung FuzzyEngine thuc tu src/fuzzy_engine.py
    """
    
    def __init__(self):
        if FUZSEMCOM_AVAILABLE:
            self.engine = FuzzyEngine()
        else:
            self.engine = None
            print("Warning: FuzzyEngine not available, using fallback")
    
    def encode(self, sensor_data: np.ndarray, 
               ndi_label: str = None, pdi_label: str = None) -> Tuple[int, float]:
        """
        Encode sensor data using FuzSemCom
        
        Parameters:
        -----------
        sensor_data : np.ndarray
            [Moisture, pH, N, Temperature, Humidity]
        ndi_label : str
            NDI label (optional)
        pdi_label : str
            PDI label (optional)
            
        Returns:
        --------
        symbol_id : int
            Semantic class ID (0-7)
        confidence : float
            Confidence score (0-1)
        """
        moisture, ph, n, temp, humidity = sensor_data
        
        if self.engine is not None:
            prediction = self.engine.predict(
                moisture=moisture,
                ph=ph,
                nitrogen=n,
                temperature=temp,
                humidity=humidity,
                ndi_label=ndi_label,
                pdi_label=pdi_label
            )
            return prediction.class_id, prediction.confidence
        else:
            # Fallback simple encoding
            return self._simple_encode(sensor_data)
    
    def _simple_encode(self, sensor_data: np.ndarray) -> Tuple[int, float]:
        """Simple fallback encoding"""
        moisture, ph, n, temp, humidity = sensor_data
        
        if moisture < 20 and ph < 6.5:
            return 3, 0.8  # water_deficit_acidic
        elif moisture < 20 and ph >= 7.0:
            return 4, 0.8  # water_deficit_alkaline
        elif ph < 5.5:
            return 5, 0.85  # acidic_soil
        elif ph > 8.0:
            return 6, 0.85  # alkaline_soil
        elif temp > 35:
            return 7, 0.9  # heat_stress
        elif humidity > 80 and temp > 25:
            return 2, 0.75  # fungal_risk
        elif n < 30:
            return 1, 0.7  # nutrient_deficiency
        else:
            return 0, 0.6  # optimal
    
    def encode_batch(self, sensor_data: np.ndarray,
                     ndi_labels: List[str] = None,
                     pdi_labels: List[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Encode batch of sensor data"""
        n_samples = len(sensor_data)
        symbol_ids = np.zeros(n_samples, dtype=int)
        confidences = np.zeros(n_samples)
        
        for i in range(n_samples):
            ndi = ndi_labels[i] if ndi_labels is not None else None
            pdi = pdi_labels[i] if pdi_labels is not None else None
            symbol_ids[i], confidences[i] = self.encode(sensor_data[i], ndi, pdi)
        
        return symbol_ids, confidences


# ============================================================
# CHANNEL SYSTEM
# ============================================================

class FuzSemComChannelSystem:
    """
    He thong truyen tin Semantic Communication voi FuzSemCom
    """
    
    def __init__(self, channel_type: str = 'awgn', snr_db: float = 10.0, **kwargs):
        self.channel_type = channel_type
        self.snr_db = snr_db
        self.encoder = FuzSemComEncoder()
        self.channel = self._create_channel(channel_type, snr_db, **kwargs)
    
    def _create_channel(self, channel_type: str, snr_db: float, **kwargs):
        if channel_type.lower() == 'awgn':
            return AWGNChannel(snr_db=snr_db)
        elif channel_type.lower() == 'rayleigh':
            return RayleighChannel(snr_db=snr_db, **kwargs)
        elif channel_type.lower() == 'rician':
            return RicianChannel(snr_db=snr_db, **kwargs)
        elif channel_type.lower() == 'lora':
            return LoRaChannel(**kwargs)
        else:
            raise ValueError(f"Unknown channel: {channel_type}")
    
    def set_snr(self, snr_db: float):
        self.snr_db = snr_db
        if hasattr(self.channel, 'set_snr'):
            self.channel.set_snr(snr_db)
    
    def transmit(self, sensor_data: np.ndarray, 
                 ndi_label: str = None, pdi_label: str = None) -> Tuple[int, float, dict]:
        """Transmit single sample through channel"""
        # 1. FuzSemCom Encoding
        tx_id, tx_conf = self.encoder.encode(sensor_data, ndi_label, pdi_label)
        
        # 2. Convert to bits (2 bytes = 16 bits)
        tx_bits = semantic_to_bits(tx_id, tx_conf)
        
        # 3. Modulation (BPSK)
        tx_symbols = bpsk_modulate(tx_bits)
        
        # 4. Channel
        rx_symbols, ch_info = self.channel.transmit(tx_symbols)
        
        # 5. Equalization (for fading channels)
        if hasattr(self.channel, 'equalize') and self.channel.h is not None:
            rx_symbols = self.channel.equalize(rx_symbols, method='mmse')
        
        # 6. Demodulation
        rx_bits = bpsk_demodulate(rx_symbols)
        
        # 7. Decode semantic
        rx_id, rx_conf = bits_to_semantic(rx_bits)
        rx_id = int(np.clip(rx_id, 0, 7))
        rx_conf = float(np.clip(rx_conf, 0, 1))
        
        # BER
        ber = calculate_ber(tx_bits, rx_bits)
        
        info = {
            'tx_id': tx_id, 'tx_conf': tx_conf,
            'rx_id': rx_id, 'rx_conf': rx_conf,
            'tx_class': SEMANTIC_CLASSES[tx_id],
            'rx_class': SEMANTIC_CLASSES[rx_id],
            'correct': tx_id == rx_id,
            'ber': ber
        }
        
        return rx_id, rx_conf, info
    
    def transmit_batch(self, sensor_data: np.ndarray,
                       ndi_labels: List[str] = None,
                       pdi_labels: List[str] = None) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Transmit batch"""
        n = len(sensor_data)
        rx_ids = np.zeros(n, dtype=int)
        rx_confs = np.zeros(n)
        total_ber = 0.0
        correct = 0
        
        for i in range(n):
            ndi = ndi_labels[i] if ndi_labels else None
            pdi = pdi_labels[i] if pdi_labels else None
            rx_id, rx_conf, info = self.transmit(sensor_data[i], ndi, pdi)
            rx_ids[i] = rx_id
            rx_confs[i] = rx_conf
            total_ber += info['ber']
            if info['correct']:
                correct += 1
        
        summary = {
            'n_samples': n,
            'semantic_accuracy': correct / n,
            'avg_ber': total_ber / n,
            'channel': self.channel_type,
            'snr_db': self.snr_db,
            'encoder': 'FuzSemCom'
        }
        
        return rx_ids, rx_confs, summary


# ============================================================
# COMPARISON FUNCTION
# ============================================================

def compare_channels_and_encoders(
    sensor_data: np.ndarray,
    snr_range: List[float] = None,
    ndi_labels: List[str] = None,
    pdi_labels: List[str] = None
) -> Dict:
    """
    So sanh 4 kenh voi FuzSemCom encoder
    """
    if snr_range is None:
        snr_range = [0, 5, 10, 15, 20, 25, 30]
    
    channels = ['awgn', 'rayleigh', 'rician', 'lora']
    results = {}
    
    for ch_type in channels:
        print(f"\n{'='*50}")
        print(f"Channel: {ch_type.upper()}")
        print(f"{'='*50}")
        
        ch_results = {
            'snr_db': [],
            'semantic_accuracy': [],
            'ber': []
        }
        
        for snr in snr_range:
            # Create system
            kwargs = {}
            if ch_type == 'rician':
                kwargs['k_factor_db'] = 3.0
            elif ch_type == 'lora':
                # For LoRa, adjust distance to achieve target SNR
                # Short distance = high SNR, long distance = low SNR
                distance_map = {0: 3000, 5: 1500, 10: 800, 15: 400, 20: 200, 25: 100, 30: 50}
                kwargs['distance'] = distance_map.get(snr, 500)
                kwargs['sf'] = 7
                kwargs['tx_power'] = 14.0
            
            system = FuzSemComChannelSystem(
                channel_type=ch_type, 
                snr_db=snr,
                **kwargs
            )
            
            # Transmit
            _, _, summary = system.transmit_batch(sensor_data, ndi_labels, pdi_labels)
            
            ch_results['snr_db'].append(snr)
            ch_results['semantic_accuracy'].append(summary['semantic_accuracy'])
            ch_results['ber'].append(summary['avg_ber'])
            
            print(f"  SNR={snr:3d}dB: Acc={summary['semantic_accuracy']*100:.2f}%, BER={summary['avg_ber']:.6f}")
        
        results[ch_type] = ch_results
    
    return results


def plot_comparison(results: Dict, output_path: Path):
    """Plot comparison results"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {'awgn': 'blue', 'rayleigh': 'red', 'rician': 'green', 'lora': 'orange'}
    markers = {'awgn': 'o', 'rayleigh': 's', 'rician': '^', 'lora': 'D'}
    
    # Semantic Accuracy
    ax = axes[0]
    for ch_type, data in results.items():
        snr = data['snr_db']
        acc = [x * 100 for x in data['semantic_accuracy']]
        ax.plot(snr, acc, f'-{markers[ch_type]}', color=colors[ch_type],
                linewidth=2, markersize=8, label=ch_type.upper())
    
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('Semantic Accuracy (%)', fontsize=12)
    ax.set_title('FuzSemCom: Semantic Accuracy vs SNR', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)
    
    # BER
    ax = axes[1]
    for ch_type, data in results.items():
        snr = data['snr_db']
        ber = data['ber']
        # Replace 0 with small value for log scale
        ber = [max(b, 1e-7) for b in ber]
        ax.semilogy(snr, ber, f'-{markers[ch_type]}', color=colors[ch_type],
                    linewidth=2, markersize=8, label=ch_type.upper())
    
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('Bit Error Rate (BER)', fontsize=12)
    ax.set_title('FuzSemCom: BER vs SNR', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved plot: {output_path}")


def load_real_dataset() -> Tuple[np.ndarray, List[str], List[str]]:
    """Load real sensor data from preprocessed dataset"""
    data_path = PROJECT_ROOT / "data" / "processed" / "semantic_dataset_preprocessed.csv"
    
    if data_path.exists():
        df = pd.read_csv(data_path)
        print(f"Loaded real dataset: {len(df)} samples")
        
        sensor_cols = ['Moisture', 'pH', 'N', 'Temperature', 'Humidity']
        sensor_data = df[sensor_cols].values
        
        ndi_labels = df['NDI_Label'].tolist() if 'NDI_Label' in df.columns else None
        pdi_labels = df['PDI_Label'].tolist() if 'PDI_Label' in df.columns else None
        
        return sensor_data, ndi_labels, pdi_labels
    else:
        print("Dataset not found, generating synthetic data...")
        np.random.seed(42)
        n = 1000
        sensor_data = np.column_stack([
            np.random.uniform(5, 50, n),
            np.random.uniform(4, 9, n),
            np.random.uniform(10, 90, n),
            np.random.uniform(15, 40, n),
            np.random.uniform(30, 95, n)
        ])
        return sensor_data, None, None


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("FUZSEMCOM CHANNEL SIMULATION")
    print("Semantic Communication System with Fuzzy Encoder")
    print("=" * 70)
    
    # Load data
    sensor_data, ndi_labels, pdi_labels = load_real_dataset()
    
    # Use subset for faster testing
    n_samples = min(2000, len(sensor_data))
    sensor_data = sensor_data[:n_samples]
    ndi_labels = ndi_labels[:n_samples] if ndi_labels else None
    pdi_labels = pdi_labels[:n_samples] if pdi_labels else None
    
    print(f"\nUsing {n_samples} samples for simulation")
    
    # Run comparison
    snr_range = [0, 5, 10, 15, 20, 25, 30]
    results = compare_channels_and_encoders(
        sensor_data, snr_range, ndi_labels, pdi_labels
    )
    
    # Save results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON
    results_json = {}
    for ch, data in results.items():
        results_json[ch] = {k: [float(v) for v in vals] for k, vals in data.items()}
    
    json_path = output_dir / f"fuzsemcom_channel_results_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults saved: {json_path}")
    
    # Plot
    plot_path = output_dir / f"fuzsemcom_channel_comparison_{timestamp}.png"
    plot_comparison(results, plot_path)
    
    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE - FuzSemCom Semantic Accuracy (%)")
    print("=" * 70)
    print(f"{'SNR (dB)':<10}", end="")
    for ch in ['awgn', 'rayleigh', 'rician', 'lora']:
        print(f"{ch.upper():<12}", end="")
    print()
    print("-" * 58)
    
    for i, snr in enumerate(snr_range):
        print(f"{snr:<10}", end="")
        for ch in ['awgn', 'rayleigh', 'rician', 'lora']:
            acc = results[ch]['semantic_accuracy'][i] * 100
            print(f"{acc:<12.2f}", end="")
        print()
    
    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)


if __name__ == "__main__":
    main()

