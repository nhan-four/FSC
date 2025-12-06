"""
Updated Comparison Plots: FuzSemCom vs L-DeepSC System Opt
==========================================================
Cập nhật biểu đồ so sánh với kết quả L-DeepSC System Optimization mới.

L-DeepSC System Opt đạt ~74% accuracy sau kênh (so với 18% trước đó).
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
LDEEPSC_JSON = Path(__file__).parent.parent / "experiments" / "l_deepsc_system_opt" / "results" / "l_deepsc_system_opt_results.json"

# L-DeepSC cũ (để so sánh improvement)
LDEEPSC_OLD = {
    "accuracy_before": 0.7763,
    "accuracy_after": 0.1838,
}


def load_data():
    """Load FuzSemCom và L-DeepSC results"""
    with open(FUZSEMCOM_JSON, 'r') as f:
        fuzsemcom = json.load(f)
    
    with open(LDEEPSC_JSON, 'r') as f:
        ldeepsc = json.load(f)
    
    return fuzsemcom, ldeepsc


def plot_accuracy_vs_snr_updated(fuzsemcom, ldeepsc, output_path):
    """
    Plot Accuracy vs SNR: FuzSemCom (4 kênh) + L-DeepSC System Opt
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Colors
    colors = {
        'awgn': '#2196F3',
        'rayleigh': '#F44336',
        'rician': '#4CAF50',
        'lora': '#FF9800',
        'ldeepsc_before': '#9C27B0',
        'ldeepsc_after': '#795548',
    }
    
    # FuzSemCom channels
    for ch in ['awgn', 'rayleigh', 'rician', 'lora']:
        data = fuzsemcom['results'][ch]
        snr = data['snr_db']
        acc = [x * 100 for x in data['semantic_accuracy']]
        ax.plot(snr, acc, '-o', color=colors[ch], linewidth=2.5, markersize=8,
                label=f'FuzSemCom ({ch.upper()})')
    
    # L-DeepSC System Opt
    snr_range = ldeepsc['config']['snr_range']
    acc_before = [ldeepsc['results_by_snr'][str(s)]['accuracy_before'] * 100 for s in snr_range]
    acc_after = [ldeepsc['results_by_snr'][str(s)]['accuracy_after'] * 100 for s in snr_range]
    
    ax.plot(snr_range, acc_before, '--^', color=colors['ldeepsc_before'], linewidth=2.5, markersize=9,
            label='L-DeepSC Sys.Opt (Before Channel)')
    ax.plot(snr_range, acc_after, '-s', color=colors['ldeepsc_after'], linewidth=2.5, markersize=9,
            label='L-DeepSC Sys.Opt (After Channel)')
    
    # Old L-DeepSC reference (horizontal line)
    ax.axhline(y=LDEEPSC_OLD['accuracy_after'] * 100, color='gray', linestyle=':', 
               linewidth=1.5, alpha=0.7, label=f'L-DeepSC OLD (After): {LDEEPSC_OLD["accuracy_after"]*100:.1f}%')
    
    ax.set_xlabel('SNR (dB)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Semantic Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('FuzSemCom vs L-DeepSC System Optimization: Accuracy over SNR', 
                 fontsize=16, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, loc='lower right', framealpha=0.95)
    ax.set_ylim(0, 105)
    ax.set_xlim(-1, 31)
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
    
    # Annotation
    ax.annotate('L-DeepSC improvement:\n18% → 74% after channel', 
                xy=(5, 50), fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_bar_comparison_snr10(fuzsemcom, ldeepsc, output_path):
    """
    Bar chart so sánh tại SNR=10dB
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
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
    
    for ch in ['awgn', 'rayleigh', 'rician', 'lora']:
        acc = fuzsemcom['results'][ch]['semantic_accuracy'][snr_idx] * 100
        methods.append(f'FuzSemCom\n({ch.upper()})')
        accuracies.append(acc)
        colors_list.append(channel_colors[ch])
    
    # L-DeepSC System Opt
    ldeepsc_10 = ldeepsc['results_by_snr']['10']
    methods.append('L-DeepSC Sys.Opt\n(Before)')
    accuracies.append(ldeepsc_10['accuracy_before'] * 100)
    colors_list.append('#9C27B0')
    
    methods.append('L-DeepSC Sys.Opt\n(After)')
    accuracies.append(ldeepsc_10['accuracy_after'] * 100)
    colors_list.append('#795548')
    
    # L-DeepSC OLD for comparison
    methods.append('L-DeepSC OLD\n(After)')
    accuracies.append(LDEEPSC_OLD['accuracy_after'] * 100)
    colors_list.append('#BDBDBD')
    
    # Create bars
    x = np.arange(len(methods))
    bars = ax.bar(x, accuracies, color=colors_list, edgecolor='white', linewidth=1.5)
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.annotate(f'{acc:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Method', fontsize=14, fontweight='bold')
    ax.set_ylabel('Semantic Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Comparison at SNR = 10 dB', fontsize=16, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9)
    ax.set_ylim(0, 115)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=100, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_efficiency_comparison(ldeepsc, output_path):
    """
    So sánh Efficiency: Bandwidth và Trade-offs
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Data
    methods = ['FuzSemCom', 'L-DeepSC\nSys.Opt', 'L-DeepSC\nOLD']
    bandwidth = [2, ldeepsc['bandwidth_bytes'], 64]
    accuracy_10db = [99.36, ldeepsc['results_by_snr']['10']['accuracy_after'] * 100, 18.38]
    colors = ['#4CAF50', '#795548', '#BDBDBD']
    
    # Bandwidth
    ax = axes[0]
    bars = ax.bar(methods, bandwidth, color=colors, edgecolor='white', linewidth=2)
    for bar, bw in zip(bars, bandwidth):
        ax.annotate(f'{bw} B', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points", ha='center', fontsize=12, fontweight='bold')
    ax.set_ylabel('Bandwidth (bytes/sample)', fontsize=12, fontweight='bold')
    ax.set_title('Bandwidth Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Accuracy at SNR=10dB
    ax = axes[1]
    bars = ax.bar(methods, accuracy_10db, color=colors, edgecolor='white', linewidth=2)
    for bar, acc in zip(bars, accuracy_10db):
        ax.annotate(f'{acc:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points", ha='center', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy @ SNR=10dB (%)', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy After Channel', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Efficiency ratio (accuracy / bandwidth)
    ax = axes[2]
    efficiency = [acc / bw for acc, bw in zip(accuracy_10db, bandwidth)]
    bars = ax.bar(methods, efficiency, color=colors, edgecolor='white', linewidth=2)
    for bar, eff in zip(bars, efficiency):
        ax.annotate(f'{eff:.2f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 5), textcoords="offset points", ha='center', fontsize=12, fontweight='bold')
    ax.set_ylabel('Efficiency (Acc%/Byte)', fontsize=12, fontweight='bold')
    ax.set_title('Efficiency Ratio', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('FuzSemCom vs L-DeepSC: Efficiency Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_ldeepsc_improvement(ldeepsc, output_path):
    """
    So sánh L-DeepSC OLD vs L-DeepSC System Opt
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Before/After comparison
    ax = axes[0]
    categories = ['Before Channel', 'After Channel']
    old_values = [LDEEPSC_OLD['accuracy_before'] * 100, LDEEPSC_OLD['accuracy_after'] * 100]
    new_values = [ldeepsc['results_by_snr']['10']['accuracy_before'] * 100, 
                  ldeepsc['results_by_snr']['10']['accuracy_after'] * 100]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, old_values, width, label='L-DeepSC OLD', color='#BDBDBD')
    bars2 = ax.bar(x + width/2, new_values, width, label='L-DeepSC Sys.Opt', color='#795548')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha='center', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('L-DeepSC Improvement @ SNR=10dB', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add improvement arrow
    ax.annotate('', xy=(1.175, new_values[1]), xytext=(1.175, old_values[1]),
                arrowprops=dict(arrowstyle='->', color='green', lw=3))
    ax.text(1.25, (old_values[1] + new_values[1]) / 2, f'+{new_values[1] - old_values[1]:.1f}%', 
            fontsize=12, fontweight='bold', color='green', va='center')
    
    # SNR sweep comparison
    ax = axes[1]
    snr_range = ldeepsc['config']['snr_range']
    acc_after = [ldeepsc['results_by_snr'][str(s)]['accuracy_after'] * 100 for s in snr_range]
    
    ax.plot(snr_range, acc_after, '-s', color='#795548', linewidth=2.5, markersize=9, 
            label='L-DeepSC Sys.Opt (After)')
    ax.axhline(y=LDEEPSC_OLD['accuracy_after'] * 100, color='#BDBDBD', linestyle='--', 
               linewidth=2, label=f'L-DeepSC OLD (After): {LDEEPSC_OLD["accuracy_after"]*100:.1f}%')
    
    ax.fill_between(snr_range, LDEEPSC_OLD['accuracy_after'] * 100, acc_after, 
                    alpha=0.3, color='green', label='Improvement')
    
    ax.set_xlabel('SNR (dB)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy After Channel (%)', fontsize=12, fontweight='bold')
    ax.set_title('L-DeepSC System Opt: Accuracy vs SNR', fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('L-DeepSC System Optimization: Before vs After', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_detailed_comparison(fuzsemcom, ldeepsc, output_path):
    """
    So sánh chi tiết với nhiều metrics
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    snr_range = [0, 5, 10, 15, 20, 25, 30]
    
    # 1. All methods accuracy vs SNR
    ax = axes[0, 0]
    for ch in ['awgn', 'rayleigh', 'rician', 'lora']:
        data = fuzsemcom['results'][ch]
        acc = [x * 100 for x in data['semantic_accuracy']]
        ax.plot(snr_range, acc, '-o', linewidth=2, markersize=6, label=f'FuzSemCom ({ch.upper()})')
    
    acc_after = [ldeepsc['results_by_snr'][str(s)]['accuracy_after'] * 100 for s in snr_range]
    ax.plot(snr_range, acc_after, '-s', color='#795548', linewidth=2.5, markersize=8, 
            label='L-DeepSC Sys.Opt')
    
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Semantic Accuracy vs SNR')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    
    # 2. FuzSemCom vs L-DeepSC at each SNR (bar chart)
    ax = axes[0, 1]
    x = np.arange(len(snr_range))
    width = 0.15
    
    # Only AWGN and LoRa for clarity
    awgn_acc = [fuzsemcom['results']['awgn']['semantic_accuracy'][i] * 100 for i in range(len(snr_range))]
    lora_acc = [fuzsemcom['results']['lora']['semantic_accuracy'][i] * 100 for i in range(len(snr_range))]
    ldeepsc_acc = [ldeepsc['results_by_snr'][str(s)]['accuracy_after'] * 100 for s in snr_range]
    
    ax.bar(x - width, awgn_acc, width, label='FuzSemCom (AWGN)', color='#2196F3')
    ax.bar(x, lora_acc, width, label='FuzSemCom (LoRa)', color='#FF9800')
    ax.bar(x + width, ldeepsc_acc, width, label='L-DeepSC Sys.Opt', color='#795548')
    
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Grouped Comparison by SNR')
    ax.set_xticks(x)
    ax.set_xticklabels(snr_range)
    ax.legend()
    ax.set_ylim(0, 110)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Reconstruction RMSE (L-DeepSC)
    ax = axes[1, 0]
    rmse = [ldeepsc['results_by_snr'][str(s)]['rmse'] for s in snr_range]
    ax.plot(snr_range, rmse, '-^', color='#E91E63', linewidth=2.5, markersize=9)
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('RMSE')
    ax.set_title('L-DeepSC Sys.Opt: Reconstruction RMSE')
    ax.grid(True, alpha=0.3)
    
    # 4. Summary table as text
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = """
    ╔══════════════════════════════════════════════════════════════╗
    ║         SUMMARY COMPARISON @ SNR = 10 dB                     ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  Method                 │ Accuracy │ Bandwidth │ Efficiency  ║
    ╠─────────────────────────┼──────────┼───────────┼─────────────╣
    ║  FuzSemCom (AWGN)       │  99.4%   │   2 B     │   49.7      ║
    ║  FuzSemCom (Rayleigh)   │  90.7%   │   2 B     │   45.3      ║
    ║  FuzSemCom (Rician)     │  91.0%   │   2 B     │   45.5      ║
    ║  FuzSemCom (LoRa)       │  75.4%   │   2 B     │   37.7      ║
    ╠─────────────────────────┼──────────┼───────────┼─────────────╣
    ║  L-DeepSC Sys.Opt       │  73.9%   │  384 B    │    0.19     ║
    ║  L-DeepSC OLD           │  18.4%   │   64 B    │    0.29     ║
    ╠═════════════════════════════════════════════════════════════╣
    ║  * Efficiency = Accuracy / Bandwidth                        ║
    ║  * FuzSemCom vượt trội về efficiency (2 bytes only!)        ║
    ║  * L-DeepSC Sys.Opt cải thiện 4x accuracy sau kênh          ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    
    ax.text(0.5, 0.5, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', horizontalalignment='center',
            fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Detailed Comparison: FuzSemCom vs L-DeepSC System Optimization', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    print("=" * 60)
    print("Creating Updated Comparison Plots")
    print("FuzSemCom vs L-DeepSC System Optimization")
    print("=" * 60)
    
    # Load data
    fuzsemcom, ldeepsc = load_data()
    print(f"Loaded FuzSemCom: {len(fuzsemcom['results'])} channels")
    print(f"Loaded L-DeepSC Sys.Opt: best acc = {ldeepsc['best_accuracy_after_channel']*100:.2f}%")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot 1: Accuracy vs SNR
    output1 = RESULTS_DIR / f"comparison_acc_vs_snr_updated_{timestamp}.png"
    plot_accuracy_vs_snr_updated(fuzsemcom, ldeepsc, output1)
    
    # Plot 2: Bar comparison at SNR=10dB
    output2 = RESULTS_DIR / f"comparison_bar_snr10_updated_{timestamp}.png"
    plot_bar_comparison_snr10(fuzsemcom, ldeepsc, output2)
    
    # Plot 3: Efficiency comparison
    output3 = RESULTS_DIR / f"comparison_efficiency_updated_{timestamp}.png"
    plot_efficiency_comparison(ldeepsc, output3)
    
    # Plot 4: L-DeepSC improvement
    output4 = RESULTS_DIR / f"ldeepsc_improvement_{timestamp}.png"
    plot_ldeepsc_improvement(ldeepsc, output4)
    
    # Plot 5: Detailed comparison
    output5 = RESULTS_DIR / f"detailed_comparison_{timestamp}.png"
    plot_detailed_comparison(fuzsemcom, ldeepsc, output5)
    
    print("\n" + "=" * 60)
    print("DONE! Created 5 comparison plots:")
    print(f"  1. {output1.name}")
    print(f"  2. {output2.name}")
    print(f"  3. {output3.name}")
    print(f"  4. {output4.name}")
    print(f"  5. {output5.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()

