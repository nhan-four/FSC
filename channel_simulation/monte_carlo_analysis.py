"""
Monte Carlo Analysis for Semantic Communication
================================================
Mo phong ngau nhien de danh gia hieu nang he thong
voi confidence intervals.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, List
import argparse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from channel_simulation.semantic_comm_system import SemanticCommSystem, SEMANTIC_CLASSES


def run_monte_carlo(
    sensor_data: np.ndarray,
    channel_type: str = 'awgn',
    snr_db: float = 10.0,
    num_trials: int = 1000,
    **channel_kwargs
) -> Dict:
    """
    Run Monte Carlo simulation
    
    Parameters:
    -----------
    sensor_data : np.ndarray
        Sensor data samples
    channel_type : str
        Channel type
    snr_db : float
        SNR in dB
    num_trials : int
        Number of Monte Carlo trials
        
    Returns:
    --------
    results : dict
        Statistics from Monte Carlo simulation
    """
    accuracies = []
    bers = []
    
    for trial in range(num_trials):
        # Create new system instance for each trial (random channel realization)
        system = SemanticCommSystem(
            channel_type=channel_type,
            snr_db=snr_db,
            **channel_kwargs
        )
        
        _, _, summary = system.transmit_batch(sensor_data)
        accuracies.append(summary['semantic_accuracy'])
        bers.append(summary['avg_ber'])
        
        if (trial + 1) % 100 == 0:
            print(f"  Trial {trial + 1}/{num_trials}")
    
    accuracies = np.array(accuracies)
    bers = np.array(bers)
    
    # Calculate statistics
    results = {
        'mean_accuracy': float(np.mean(accuracies)),
        'std_accuracy': float(np.std(accuracies)),
        'ci_95_accuracy': (
            float(np.percentile(accuracies, 2.5)),
            float(np.percentile(accuracies, 97.5))
        ),
        'mean_ber': float(np.mean(bers)),
        'std_ber': float(np.std(bers)),
        'ci_95_ber': (
            float(np.percentile(bers, 2.5)),
            float(np.percentile(bers, 97.5))
        ),
        'num_trials': num_trials,
        'channel_type': channel_type,
        'snr_db': snr_db
    }
    
    return results


def run_snr_sweep_monte_carlo(
    sensor_data: np.ndarray,
    channel_type: str = 'awgn',
    snr_range: List[float] = None,
    num_trials: int = 100,
    **channel_kwargs
) -> Dict:
    """
    Run Monte Carlo for range of SNR values
    """
    if snr_range is None:
        snr_range = [0, 5, 10, 15, 20, 25, 30]
    
    all_results = {
        'snr_db': [],
        'mean_accuracy': [],
        'ci_lower': [],
        'ci_upper': [],
        'mean_ber': [],
        'ber_ci_lower': [],
        'ber_ci_upper': []
    }
    
    for snr in snr_range:
        print(f"\nSNR = {snr} dB")
        results = run_monte_carlo(
            sensor_data, channel_type, snr, num_trials, **channel_kwargs
        )
        
        all_results['snr_db'].append(snr)
        all_results['mean_accuracy'].append(results['mean_accuracy'])
        all_results['ci_lower'].append(results['ci_95_accuracy'][0])
        all_results['ci_upper'].append(results['ci_95_accuracy'][1])
        all_results['mean_ber'].append(results['mean_ber'])
        all_results['ber_ci_lower'].append(results['ci_95_ber'][0])
        all_results['ber_ci_upper'].append(results['ci_95_ber'][1])
    
    return all_results


def plot_monte_carlo_results(results: Dict, output_path: Path):
    """Plot Monte Carlo results with confidence intervals"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    snr = results['snr_db']
    
    # Semantic Accuracy
    ax = axes[0]
    mean_acc = [x * 100 for x in results['mean_accuracy']]
    ci_lower = [x * 100 for x in results['ci_lower']]
    ci_upper = [x * 100 for x in results['ci_upper']]
    
    ax.plot(snr, mean_acc, 'b-o', linewidth=2, markersize=8, label='Mean')
    ax.fill_between(snr, ci_lower, ci_upper, alpha=0.3, color='blue', label='95% CI')
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Semantic Accuracy (%)')
    ax.set_title('Semantic Accuracy vs SNR (Monte Carlo)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 105)
    
    # BER
    ax = axes[1]
    mean_ber = results['mean_ber']
    ber_ci_lower = results['ber_ci_lower']
    ber_ci_upper = results['ber_ci_upper']
    
    ax.semilogy(snr, mean_ber, 'r-o', linewidth=2, markersize=8, label='Mean')
    ax.fill_between(snr, ber_ci_lower, ber_ci_upper, alpha=0.3, color='red', label='95% CI')
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Bit Error Rate (BER)')
    ax.set_title('BER vs SNR (Monte Carlo)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Saved plot: {output_path}")


def compare_channels_monte_carlo(
    sensor_data: np.ndarray,
    channels: List[str] = None,
    snr_range: List[float] = None,
    num_trials: int = 100
) -> Dict:
    """Compare multiple channels using Monte Carlo"""
    
    if channels is None:
        channels = ['awgn', 'rayleigh', 'rician', 'lora']
    
    if snr_range is None:
        snr_range = [0, 5, 10, 15, 20, 25, 30]
    
    all_results = {}
    
    for ch_type in channels:
        print(f"\n{'='*50}")
        print(f"Channel: {ch_type.upper()}")
        print(f"{'='*50}")
        
        kwargs = {}
        if ch_type == 'rician':
            kwargs['k_factor_db'] = 3.0
        
        results = run_snr_sweep_monte_carlo(
            sensor_data, ch_type, snr_range, num_trials, **kwargs
        )
        all_results[ch_type] = results
    
    return all_results


def plot_channel_comparison(results: Dict, output_path: Path):
    """Plot comparison of multiple channels"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {'awgn': 'blue', 'rayleigh': 'red', 'rician': 'green', 'lora': 'orange'}
    markers = {'awgn': 'o', 'rayleigh': 's', 'rician': '^', 'lora': 'D'}
    
    # Semantic Accuracy
    ax = axes[0]
    for ch_type, data in results.items():
        snr = data['snr_db']
        mean_acc = [x * 100 for x in data['mean_accuracy']]
        ci_lower = [x * 100 for x in data['ci_lower']]
        ci_upper = [x * 100 for x in data['ci_upper']]
        
        color = colors.get(ch_type, 'gray')
        ax.plot(snr, mean_acc, '-o', color=color, linewidth=2, markersize=6, 
                label=ch_type.upper())
        ax.fill_between(snr, ci_lower, ci_upper, alpha=0.2, color=color)
    
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Semantic Accuracy (%)')
    ax.set_title('Semantic Accuracy Comparison (Monte Carlo)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 105)
    
    # BER
    ax = axes[1]
    for ch_type, data in results.items():
        snr = data['snr_db']
        mean_ber = data['mean_ber']
        
        color = colors.get(ch_type, 'gray')
        ax.semilogy(snr, mean_ber, '-o', color=color, linewidth=2, markersize=6,
                    label=ch_type.upper())
    
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Bit Error Rate (BER)')
    ax.set_title('BER Comparison (Monte Carlo)')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"Saved plot: {output_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Monte Carlo Analysis for Semantic Communication')
    parser.add_argument('--num_trials', type=int, default=100, help='Number of Monte Carlo trials')
    parser.add_argument('--num_samples', type=int, default=500, help='Number of sensor samples')
    parser.add_argument('--channel', type=str, default='all', help='Channel type (awgn/rayleigh/rician/all)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("MONTE CARLO ANALYSIS - Semantic Communication System")
    print("=" * 70)
    print(f"Num trials: {args.num_trials}")
    print(f"Num samples: {args.num_samples}")
    
    # Generate sensor data
    np.random.seed(42)
    sensor_data = np.column_stack([
        np.random.uniform(5, 50, args.num_samples),
        np.random.uniform(4, 9, args.num_samples),
        np.random.uniform(10, 90, args.num_samples),
        np.random.uniform(15, 40, args.num_samples),
        np.random.uniform(30, 95, args.num_samples)
    ])
    
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.channel == 'all':
        # Compare all channels
        results = compare_channels_monte_carlo(
            sensor_data,
            channels=['awgn', 'rayleigh', 'rician'],
            num_trials=args.num_trials
        )
        
        # Save results
        results_path = output_dir / f"monte_carlo_comparison_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved: {results_path}")
        
        # Plot
        plot_path = output_dir / f"monte_carlo_comparison_{timestamp}.png"
        plot_channel_comparison(results, plot_path)
        
    else:
        # Single channel
        results = run_snr_sweep_monte_carlo(
            sensor_data,
            channel_type=args.channel,
            num_trials=args.num_trials
        )
        
        # Save
        results_path = output_dir / f"monte_carlo_{args.channel}_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved: {results_path}")
        
        # Plot
        plot_path = output_dir / f"monte_carlo_{args.channel}_{timestamp}.png"
        plot_monte_carlo_results(results, plot_path)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if args.channel == 'all':
        for ch_type, data in results.items():
            print(f"\n{ch_type.upper()}:")
            for i, snr in enumerate(data['snr_db']):
                acc = data['mean_accuracy'][i] * 100
                ci_l = data['ci_lower'][i] * 100
                ci_u = data['ci_upper'][i] * 100
                print(f"  SNR={snr:3d}dB: {acc:.2f}% [{ci_l:.2f}%, {ci_u:.2f}%]")
    
    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)


if __name__ == "__main__":
    main()

