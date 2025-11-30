"""
Performance Metrics
===================
BER, SER, Semantic Accuracy, Throughput
"""

import numpy as np
from typing import Tuple, List, Optional


def calculate_ber(tx_bits: np.ndarray, rx_bits: np.ndarray) -> float:
    """
    Calculate Bit Error Rate
    
    BER = Number of bit errors / Total bits
    """
    if len(tx_bits) != len(rx_bits):
        min_len = min(len(tx_bits), len(rx_bits))
        tx_bits = tx_bits[:min_len]
        rx_bits = rx_bits[:min_len]
    
    errors = np.sum(tx_bits != rx_bits)
    return errors / len(tx_bits) if len(tx_bits) > 0 else 0.0


def calculate_ser(tx_symbols: np.ndarray, rx_symbols: np.ndarray) -> float:
    """
    Calculate Symbol Error Rate
    
    SER = Number of symbol errors / Total symbols
    """
    if len(tx_symbols) != len(rx_symbols):
        min_len = min(len(tx_symbols), len(rx_symbols))
        tx_symbols = tx_symbols[:min_len]
        rx_symbols = rx_symbols[:min_len]
    
    errors = np.sum(tx_symbols != rx_symbols)
    return errors / len(tx_symbols) if len(tx_symbols) > 0 else 0.0


def calculate_semantic_accuracy(tx_ids: np.ndarray, rx_ids: np.ndarray) -> float:
    """
    Calculate Semantic Classification Accuracy
    
    Accuracy = Correct classifications / Total samples
    """
    if len(tx_ids) != len(rx_ids):
        min_len = min(len(tx_ids), len(rx_ids))
        tx_ids = tx_ids[:min_len]
        rx_ids = rx_ids[:min_len]
    
    correct = np.sum(tx_ids == rx_ids)
    return correct / len(tx_ids) if len(tx_ids) > 0 else 0.0


def calculate_mse(tx_signal: np.ndarray, rx_signal: np.ndarray) -> float:
    """
    Calculate Mean Squared Error
    """
    if len(tx_signal) != len(rx_signal):
        min_len = min(len(tx_signal), len(rx_signal))
        tx_signal = tx_signal[:min_len]
        rx_signal = rx_signal[:min_len]
    
    return np.mean(np.abs(tx_signal - rx_signal) ** 2)


def calculate_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    """
    Calculate Signal-to-Noise Ratio in dB
    """
    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = np.mean(np.abs(noise) ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    return 10 * np.log10(signal_power / noise_power)


def calculate_throughput(payload_bytes: int, toa_seconds: float) -> float:
    """
    Calculate Throughput in bits per second
    """
    if toa_seconds == 0:
        return float('inf')
    
    return (payload_bytes * 8) / toa_seconds


def calculate_spectral_efficiency(bit_rate: float, bandwidth: float) -> float:
    """
    Calculate Spectral Efficiency in bits/s/Hz
    """
    if bandwidth == 0:
        return 0.0
    
    return bit_rate / bandwidth


def calculate_energy_per_bit(tx_power_mw: float, bit_rate: float) -> float:
    """
    Calculate Energy per bit in Joules
    """
    if bit_rate == 0:
        return float('inf')
    
    return (tx_power_mw / 1000) / bit_rate


class PerformanceMetrics:
    """
    Class to track and aggregate performance metrics
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.total_bits = 0
        self.bit_errors = 0
        self.total_symbols = 0
        self.symbol_errors = 0
        self.total_semantic = 0
        self.semantic_correct = 0
        self.mse_sum = 0.0
        self.mse_count = 0
    
    def update_ber(self, tx_bits: np.ndarray, rx_bits: np.ndarray):
        """Update BER statistics"""
        min_len = min(len(tx_bits), len(rx_bits))
        self.total_bits += min_len
        self.bit_errors += np.sum(tx_bits[:min_len] != rx_bits[:min_len])
    
    def update_ser(self, tx_symbols: np.ndarray, rx_symbols: np.ndarray):
        """Update SER statistics"""
        min_len = min(len(tx_symbols), len(rx_symbols))
        self.total_symbols += min_len
        self.symbol_errors += np.sum(tx_symbols[:min_len] != rx_symbols[:min_len])
    
    def update_semantic(self, tx_ids: np.ndarray, rx_ids: np.ndarray):
        """Update semantic accuracy statistics"""
        min_len = min(len(tx_ids), len(rx_ids))
        self.total_semantic += min_len
        self.semantic_correct += np.sum(tx_ids[:min_len] == rx_ids[:min_len])
    
    def update_mse(self, tx_signal: np.ndarray, rx_signal: np.ndarray):
        """Update MSE statistics"""
        min_len = min(len(tx_signal), len(rx_signal))
        self.mse_sum += np.sum(np.abs(tx_signal[:min_len] - rx_signal[:min_len]) ** 2)
        self.mse_count += min_len
    
    def get_ber(self) -> float:
        return self.bit_errors / self.total_bits if self.total_bits > 0 else 0.0
    
    def get_ser(self) -> float:
        return self.symbol_errors / self.total_symbols if self.total_symbols > 0 else 0.0
    
    def get_semantic_accuracy(self) -> float:
        return self.semantic_correct / self.total_semantic if self.total_semantic > 0 else 0.0
    
    def get_mse(self) -> float:
        return self.mse_sum / self.mse_count if self.mse_count > 0 else 0.0
    
    def get_summary(self) -> dict:
        return {
            'ber': self.get_ber(),
            'ser': self.get_ser(),
            'semantic_accuracy': self.get_semantic_accuracy(),
            'mse': self.get_mse(),
            'total_bits': self.total_bits,
            'total_symbols': self.total_symbols,
            'total_semantic': self.total_semantic
        }
    
    def __repr__(self):
        return (f"PerformanceMetrics(BER={self.get_ber():.6f}, "
                f"SER={self.get_ser():.6f}, "
                f"Semantic Acc={self.get_semantic_accuracy():.4f})")


if __name__ == "__main__":
    print("Testing Metrics")
    print("=" * 50)
    
    # Test BER
    tx = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    rx = np.array([0, 1, 1, 1, 1, 0, 0, 0])  # 2 errors
    print(f"BER: {calculate_ber(tx, rx):.4f} (expected: 0.25)")
    
    # Test semantic accuracy
    tx_ids = np.array([0, 1, 2, 3, 4])
    rx_ids = np.array([0, 1, 2, 2, 4])  # 1 error
    print(f"Semantic Acc: {calculate_semantic_accuracy(tx_ids, rx_ids):.4f} (expected: 0.8)")
    
    # Test aggregation
    metrics = PerformanceMetrics()
    for _ in range(10):
        tx = np.random.randint(0, 2, 100)
        rx = tx.copy()
        rx[np.random.choice(100, 5, replace=False)] ^= 1  # 5% errors
        metrics.update_ber(tx, rx)
    
    print(f"Aggregated BER: {metrics.get_ber():.4f} (expected: ~0.05)")

