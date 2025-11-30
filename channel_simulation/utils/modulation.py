"""
Modulation Schemes
==================
BPSK, QPSK, QAM modulation/demodulation
"""

import numpy as np
from typing import Tuple


# ============================================================
# BPSK
# ============================================================

def bpsk_modulate(bits: np.ndarray) -> np.ndarray:
    """
    BPSK Modulation: 0 -> -1, 1 -> +1
    """
    return 2 * bits.astype(float) - 1


def bpsk_demodulate(symbols: np.ndarray) -> np.ndarray:
    """
    BPSK Demodulation: <0 -> 0, >=0 -> 1
    """
    return (np.real(symbols) >= 0).astype(int)


# ============================================================
# QPSK
# ============================================================

def qpsk_modulate(bits: np.ndarray) -> np.ndarray:
    """
    QPSK Modulation (Gray coding)
    00 -> -1-1j, 01 -> -1+1j, 10 -> +1-1j, 11 -> +1+1j
    """
    # Ensure even number of bits
    if len(bits) % 2 != 0:
        bits = np.append(bits, 0)
    
    bits = bits.reshape(-1, 2)
    
    # Gray coding
    I = 2 * bits[:, 0] - 1
    Q = 2 * bits[:, 1] - 1
    
    return (I + 1j * Q) / np.sqrt(2)


def qpsk_demodulate(symbols: np.ndarray) -> np.ndarray:
    """
    QPSK Demodulation
    """
    I_bits = (np.real(symbols) >= 0).astype(int)
    Q_bits = (np.imag(symbols) >= 0).astype(int)
    
    bits = np.zeros(2 * len(symbols), dtype=int)
    bits[0::2] = I_bits
    bits[1::2] = Q_bits
    
    return bits


# ============================================================
# QAM
# ============================================================

def qam_constellation(M: int) -> np.ndarray:
    """
    Generate M-QAM constellation (Gray coded)
    """
    k = int(np.sqrt(M))
    if k * k != M:
        raise ValueError(f"M must be a perfect square, got {M}")
    
    # Create constellation points
    I = np.arange(-(k-1), k, 2)
    Q = np.arange(-(k-1), k, 2)
    
    constellation = []
    for q in Q:
        for i in I:
            constellation.append(i + 1j * q)
    
    constellation = np.array(constellation)
    
    # Normalize average power to 1
    constellation = constellation / np.sqrt(np.mean(np.abs(constellation) ** 2))
    
    return constellation


def qam_modulate(bits: np.ndarray, M: int = 16) -> np.ndarray:
    """
    M-QAM Modulation
    """
    k = int(np.log2(M))
    
    # Pad bits if necessary
    if len(bits) % k != 0:
        bits = np.append(bits, np.zeros(k - len(bits) % k, dtype=int))
    
    # Group bits
    bits = bits.reshape(-1, k)
    
    # Convert to indices
    indices = np.sum(bits * (2 ** np.arange(k)[::-1]), axis=1)
    
    # Get constellation
    constellation = qam_constellation(M)
    
    return constellation[indices]


def qam_demodulate(symbols: np.ndarray, M: int = 16) -> np.ndarray:
    """
    M-QAM Demodulation (hard decision)
    """
    constellation = qam_constellation(M)
    k = int(np.log2(M))
    
    # Find closest constellation point
    distances = np.abs(symbols[:, np.newaxis] - constellation[np.newaxis, :])
    indices = np.argmin(distances, axis=1)
    
    # Convert indices to bits
    bits = np.zeros((len(symbols), k), dtype=int)
    for i in range(k):
        bits[:, k - 1 - i] = (indices >> i) & 1
    
    return bits.flatten()


# ============================================================
# Semantic Symbol Encoding (for FuzSemCom)
# ============================================================

def semantic_to_bits(symbol_id: int, confidence: float, id_bits: int = 8, conf_bits: int = 8) -> np.ndarray:
    """
    Convert semantic symbol to bits
    
    Parameters:
    -----------
    symbol_id : int
        Semantic class ID (0-255 for 8 bits)
    confidence : float
        Confidence value (0-1)
    id_bits : int
        Number of bits for ID
    conf_bits : int
        Number of bits for confidence
    """
    # Convert ID to bits
    id_binary = np.array([(symbol_id >> i) & 1 for i in range(id_bits - 1, -1, -1)])
    
    # Quantize confidence to bits
    conf_quantized = int(confidence * (2 ** conf_bits - 1))
    conf_binary = np.array([(conf_quantized >> i) & 1 for i in range(conf_bits - 1, -1, -1)])
    
    return np.concatenate([id_binary, conf_binary])


def bits_to_semantic(bits: np.ndarray, id_bits: int = 8, conf_bits: int = 8) -> Tuple[int, float]:
    """
    Convert bits back to semantic symbol
    """
    id_binary = bits[:id_bits]
    conf_binary = bits[id_bits:id_bits + conf_bits]
    
    # Convert to values
    symbol_id = int(np.sum(id_binary * (2 ** np.arange(id_bits - 1, -1, -1))))
    conf_quantized = int(np.sum(conf_binary * (2 ** np.arange(conf_bits - 1, -1, -1))))
    confidence = conf_quantized / (2 ** conf_bits - 1)
    
    return symbol_id, confidence


if __name__ == "__main__":
    print("Testing Modulation Schemes")
    print("=" * 50)
    
    # Test BPSK
    bits = np.array([0, 1, 0, 1, 1, 0])
    symbols = bpsk_modulate(bits)
    recovered = bpsk_demodulate(symbols)
    print(f"BPSK: {bits} -> {symbols} -> {recovered}")
    
    # Test QPSK
    bits = np.array([0, 0, 0, 1, 1, 0, 1, 1])
    symbols = qpsk_modulate(bits)
    recovered = qpsk_demodulate(symbols)
    print(f"QPSK: {bits} -> {np.round(symbols, 2)} -> {recovered}")
    
    # Test semantic encoding
    symbol_id = 5
    confidence = 0.85
    bits = semantic_to_bits(symbol_id, confidence)
    rec_id, rec_conf = bits_to_semantic(bits)
    print(f"Semantic: ID={symbol_id}, Conf={confidence:.2f} -> {bits} -> ID={rec_id}, Conf={rec_conf:.2f}")

