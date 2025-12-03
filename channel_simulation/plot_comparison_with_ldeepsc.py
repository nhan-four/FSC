"""
Plot comparison between FuzSemCom and L-DeepSC
==============================================
Tạo biểu đồ so sánh FuzSemCom (4 kênh) với L-DeepSC baseline.
L-DeepSC chỉ có kết quả tại SNR=10dB nên sẽ vẽ như điểm/đường ngang.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Paths
RESULTS_DIR = Path(__file__).parent / "results"
FUZSEMCOM_JSON = RESULTS_DIR / "full_comparison_20251201_225241.json"

# L-DeepSC results (from results/l_deepsc_comparison)
LDEEPSC_ACCURACY_BEFORE_CHANNEL = 0.7763  # 77.63% - before channel noise
LDEEPSC_ACCURACY_AFTER_CHANNEL = 0.1838   # 18.38% - after channel (SNR=10dB)
LDEEPSC_SNR = 10.0


def load_fuzsemcom_results():
    """Load FuzSemCom channel comparison results"""
    with open(FUZSEMCOM_JSON, 'r') as f:
        data = json.load(f)
    return data


def plot_accuracy_vs_snr_with_ldeepsc(fuzsemcom_data, output_path):
    """
    Plot Semantic Accuracy vs SNR for all channels + L-DeepSC baseline
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Colors and markers for FuzSemCom channels
    colors = {
        'awgn': '#2196F3',      # Blue
        'rayleigh': '#F44336',  # Red
        'rician': '#4CAF50',    # Green
        'lora': '#FF9800'       # Orange
    }
    markers = {'awgn': 'o', 'rayleigh': 's', 'rician': '^', 'lora': 'D'}
    
    # Plot FuzSemCom channels
    for ch_type in ['awgn', 'rayleigh', 'rician', 'lora']:
        data = fuzsemcom_data['results'][ch_type]
        snr = data['snr_db']
        acc = [x * 100 for x in data['semantic_accuracy']]
        ax.plot(snr, acc, f'-{markers[ch_type]}', color=colors[ch_type],
                linewidth=2.5, markersize=9, label=f'FuzSemCom ({ch_type.upper()})')
    
    # Plot L-DeepSC as horizontal dashed line + marker at SNR=10
    ax.axhline(y=LDEEPSC_ACCURACY_BEFORE_CHANNEL * 100, color='#9C27B0', 
               linestyle='--', linewidth=2, alpha=0.7,
               label=f'L-DeepSC (before channel): {LDEEPSC_ACCURACY_BEFORE_CHANNEL*100:.1f}%')
    
    ax.axhline(y=LDEEPSC_ACCURACY_AFTER_CHANNEL * 100, color='#795548', 
               linestyle=':', linewidth=2, alpha=0.7,
               label=f'L-DeepSC (after channel, SNR=10dB): {LDEEPSC_ACCURACY_AFTER_CHANNEL*100:.1f}%')
    
    # Mark L-DeepSC points at SNR=10
    ax.scatter([LDEEPSC_SNR], [LDEEPSC_ACCURACY_BEFORE_CHANNEL * 100], 
               color='#9C27B0', s=150, marker='*', zorder=5, edgecolors='white', linewidths=1.5)
    ax.scatter([LDEEPSC_SNR], [LDEEPSC_ACCURACY_AFTER_CHANNEL * 100], 
               color='#795548', s=150, marker='*', zorder=5, edgecolors='white', linewidths=1.5)
    
    # Styling
    ax.set_xlabel('SNR (dB)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Semantic Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('FuzSemCom vs L-DeepSC: Semantic Accuracy over Different Channels', 
                 fontsize=16, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='-')
    ax.legend(fontsize=10, loc='lower right', framealpha=0.95)
    ax.set_ylim(0, 105)
    ax.set_xlim(-1, 31)
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
    
    # Add annotation
    ax.annotate('L-DeepSC degrades\nsignificantly after\nchannel transmission', 
                xy=(10, LDEEPSC_ACCURACY_AFTER_CHANNEL * 100 + 3),
                xytext=(15, 35), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='#795548', lw=1.5),
                color='#795548', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_bar_comparison_at_snr10(fuzsemcom_data, output_path):
    """
    Bar chart comparing all methods at SNR=10dB
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Get FuzSemCom accuracies at SNR=10dB (index 2 in snr_range [0,5,10,15,20,25,30])
    snr_idx = 2  # SNR=10dB
    
    methods = []
    accuracies = []
    colors_list = []
    
    # FuzSemCom channels
    channel_colors = {
        'awgn': '#2196F3',
        'rayleigh': '#F44336', 
        'rician': '#4CAF50',
        'lora': '#FF9800'
    }
    
    for ch_type in ['awgn', 'rayleigh', 'rician', 'lora']:
        acc = fuzsemcom_data['results'][ch_type]['semantic_accuracy'][snr_idx] * 100
        methods.append(f'FuzSemCom\n({ch_type.upper()})')
        accuracies.append(acc)
        colors_list.append(channel_colors[ch_type])
    
    # L-DeepSC
    methods.append('L-DeepSC\n(before channel)')
    accuracies.append(LDEEPSC_ACCURACY_BEFORE_CHANNEL * 100)
    colors_list.append('#9C27B0')
    
    methods.append('L-DeepSC\n(after channel)')
    accuracies.append(LDEEPSC_ACCURACY_AFTER_CHANNEL * 100)
    colors_list.append('#795548')
    
    # Create bars
    x = np.arange(len(methods))
    bars = ax.bar(x, accuracies, color=colors_list, edgecolor='white', linewidth=1.5)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.annotate(f'{acc:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Styling
    ax.set_xlabel('Method', fontsize=14, fontweight='bold')
    ax.set_ylabel('Semantic Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Comparison at SNR = 10 dB: FuzSemCom vs L-DeepSC', 
                 fontsize=16, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylim(0, 115)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add horizontal line at 100%
    ax.axhline(y=100, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_efficiency_comparison(output_path):
    """
    Bar chart comparing bandwidth efficiency
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: Bandwidth
    ax = axes[0]
    methods = ['FuzSemCom', 'L-DeepSC']
    bandwidth = [2, 64]  # bytes per sample
    colors = ['#4CAF50', '#9C27B0']
    
    bars = ax.bar(methods, bandwidth, color=colors, edgecolor='white', linewidth=2)
    for bar, bw in zip(bars, bandwidth):
        ax.annotate(f'{bw} bytes',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_ylabel('Bandwidth (bytes/sample)', fontsize=13, fontweight='bold')
    ax.set_title('Bandwidth Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 80)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Subplot 2: Compression Ratio
    ax = axes[1]
    original_size = 20  # 5 sensors × 4 bytes
    compression = [original_size / 2, original_size / 64]  # compression ratio
    
    bars = ax.bar(methods, compression, color=colors, edgecolor='white', linewidth=2)
    for bar, cr in zip(bars, compression):
        ax.annotate(f'{cr:.1f}x',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_ylabel('Compression Ratio', fontsize=13, fontweight='bold')
    ax.set_title('Compression Efficiency', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('FuzSemCom vs L-DeepSC: Efficiency Comparison', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    print("=" * 60)
    print("Creating comparison plots: FuzSemCom vs L-DeepSC")
    print("=" * 60)
    
    # Load FuzSemCom results
    fuzsemcom_data = load_fuzsemcom_results()
    print(f"Loaded FuzSemCom results: {len(fuzsemcom_data['results'])} channels")
    
    # Timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot 1: Accuracy vs SNR with L-DeepSC baseline
    output1 = RESULTS_DIR / f"comparison_fuzsemcom_vs_ldeepsc_snr_{timestamp}.png"
    plot_accuracy_vs_snr_with_ldeepsc(fuzsemcom_data, output1)
    
    # Plot 2: Bar comparison at SNR=10dB
    output2 = RESULTS_DIR / f"comparison_bar_snr10_{timestamp}.png"
    plot_bar_comparison_at_snr10(fuzsemcom_data, output2)
    
    # Plot 3: Efficiency comparison
    output3 = RESULTS_DIR / f"comparison_efficiency_{timestamp}.png"
    plot_efficiency_comparison(output3)
    
    print("\n" + "=" * 60)
    print("DONE! Created 3 comparison plots:")
    print(f"  1. {output1.name}")
    print(f"  2. {output2.name}")
    print(f"  3. {output3.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()

