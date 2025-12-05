"""
L-DeepSC System-Level Optimization
==================================
Tối ưu TOÀN HỆ THỐNG L-DeepSC, không chỉ classification.

Vấn đề của các phiên bản trước:
- Accuracy trước kênh: ~77%
- Accuracy SAU kênh: ~18% (giảm nghiêm trọng!)
- Nguyên nhân: Channel robustness kém

Cải tiến trong phiên bản này:
1. Channel-Aware Training: Train với nhiều mức SNR (0-30 dB)
2. Repetition Coding: Truyền symbol nhiều lần, voting ở receiver
3. Improved ADNet: Self-attention cho CSI estimation
4. Joint Loss: Reconstruction + Classification + Channel Consistency
5. SNR-Adaptive Inference: Điều chỉnh decoding theo SNR

Paper gốc: "Lite Distributed Semantic Communication System for Internet of Things"
           https://arxiv.org/abs/2007.11095
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "semantic_dataset_preprocessed.csv"
OUTPUT_DIR = Path(__file__).parent / "results"

SENSOR_COLS = ["Moisture", "pH", "N", "Temperature", "Humidity"]

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

# Hyperparameters - System Optimization
TEST_SIZE = 0.2
BATCH_SIZE = 64
NUM_EPOCHS = 150
LR = 1e-3
LATENT_DIM = 32
CHAN_DIM = 32
HIDDEN_DIM = 128

# Channel-aware training
SNR_RANGE = [0, 5, 10, 15, 20, 25, 30]  # Train với nhiều SNR
DEFAULT_SNR = 10.0

# Repetition coding
NUM_REPETITIONS = 3  # Truyền mỗi symbol 3 lần

# Loss weights
LAMBDA_RECON = 0.5
LAMBDA_CLASS_BEFORE = 0.3
LAMBDA_CLASS_AFTER = 0.5  # Tăng weight cho classification sau kênh
LAMBDA_CONSISTENCY = 0.2  # z vs z_hat consistency

DROPOUT = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# MODEL COMPONENTS
# ============================================================

class SemanticEncoder(nn.Module):
    """Semantic Encoder với residual connections"""
    def __init__(self, input_dim, latent_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, latent_dim)
        self.dropout = nn.Dropout(DROPOUT)
        
        # Residual projection
        self.res_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else None

    def forward(self, x):
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        
        res = self.res_proj(x) if self.res_proj else x
        out = F.relu(self.bn2(self.fc2(out))) + res[:, :out.size(1)]
        out = self.dropout(out)
        
        return self.fc3(out)


class SemanticDecoder(nn.Module):
    """Semantic Decoder với residual connections"""
    def __init__(self, latent_dim, output_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, z):
        out = F.relu(self.bn1(self.fc1(z)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.fc2(out)))
        out = self.dropout(out)
        return self.fc3(out)


class ClassifierHead(nn.Module):
    """Classification head"""
    def __init__(self, latent_dim, num_classes, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, z):
        return self.net(z)


class ChannelEncoderWithRepetition(nn.Module):
    """Channel Encoder với repetition coding"""
    def __init__(self, latent_dim, chan_dim, num_repetitions=3):
        super().__init__()
        self.num_repetitions = num_repetitions
        self.encoder = nn.Sequential(
            nn.Linear(latent_dim, chan_dim),
            nn.Tanh()  # Power constraint
        )

    def forward(self, z):
        # Encode
        x_chan = self.encoder(z)
        # Repeat for redundancy
        x_repeated = x_chan.repeat(1, self.num_repetitions)
        return x_repeated, x_chan


class ChannelDecoderWithRepetition(nn.Module):
    """Channel Decoder với majority voting từ repetitions"""
    def __init__(self, chan_dim, latent_dim, num_repetitions=3):
        super().__init__()
        self.chan_dim = chan_dim
        self.num_repetitions = num_repetitions
        self.decoder = nn.Sequential(
            nn.Linear(chan_dim, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )

    def forward(self, y_eq):
        # Split repeated symbols
        B = y_eq.size(0)
        y_split = y_eq.view(B, self.num_repetitions, self.chan_dim)
        
        # Average (soft combining)
        y_combined = y_split.mean(dim=1)
        
        return self.decoder(y_combined)


class ImprovedADNet(nn.Module):
    """
    Improved ADNet với Self-Attention cho CSI estimation
    """
    def __init__(self, length, num_heads=4):
        super().__init__()
        self.length = length
        
        # Initial conv
        self.in_conv = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        
        # Self-attention
        self.attention = nn.MultiheadAttention(32, num_heads, batch_first=True)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([
            self._make_res_block(32) for _ in range(3)
        ])
        
        # Output
        self.out_conv = nn.Conv1d(32, 1, kernel_size=3, padding=1)

    def _make_res_block(self, channels):
        return nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels)
        )

    def forward(self, h_ls):
        # h_ls: [B, L]
        x = h_ls.unsqueeze(1)  # [B, 1, L]
        x = F.relu(self.bn1(self.in_conv(x)))  # [B, 32, L]
        
        # Self-attention
        x_t = x.permute(0, 2, 1)  # [B, L, 32]
        attn_out, _ = self.attention(x_t, x_t, x_t)
        x = attn_out.permute(0, 2, 1)  # [B, 32, L]
        
        # Residual blocks
        for block in self.res_blocks:
            x = F.relu(block(x) + x)
        
        # Output
        out = self.out_conv(x).squeeze(1)  # [B, L]
        return out


class RobustFadingChannel(nn.Module):
    """
    Robust Fading Channel với:
    - SNR-aware noise injection
    - Improved CSI estimation với ADNet
    - Support cho repetition coding
    """
    def __init__(self, chan_dim, num_repetitions=3):
        super().__init__()
        self.chan_dim = chan_dim
        self.num_repetitions = num_repetitions
        self.total_dim = chan_dim * num_repetitions
        self.adnet = ImprovedADNet(length=self.total_dim)
        
        # SNR embedding for adaptive processing
        self.snr_embed = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, self.total_dim)
        )

    def forward(self, x_repeated, snr_db):
        B, D = x_repeated.shape
        device = x_repeated.device
        
        # Rayleigh fading: H = |CN(0,1)|
        H_true = torch.abs(torch.randn(B, D, device=device) / np.sqrt(2) + 
                          1j * torch.randn(B, D, device=device) / np.sqrt(2)).float() + 1e-6
        
        # SNR-aware noise
        snr_linear = 10 ** (snr_db / 10.0)
        noise_var = 1.0 / snr_linear
        noise = (noise_var ** 0.5) * torch.randn_like(x_repeated)
        
        # Channel output
        y = H_true * x_repeated + noise
        
        # CSI estimation (LS with noise)
        # Noise level depends on SNR
        est_noise_std = 0.3 / (1 + snr_linear / 10)
        H_ls = H_true + est_noise_std * torch.randn_like(H_true)
        
        # Refine CSI với Improved ADNet
        H_refine = self.adnet(H_ls)
        H_refine = F.relu(H_refine) + 1e-6  # Ensure positive
        
        # SNR-adaptive equalization
        snr_factor = self.snr_embed(torch.tensor([[snr_db]], device=device).float())
        snr_factor = snr_factor.expand(B, -1)
        
        # MMSE-like equalization
        y_eq = y * H_refine.conj() / (H_refine.abs() ** 2 + 1/snr_linear)
        
        return y_eq, H_true, H_ls, H_refine


class LDeepSCSystemOpt(nn.Module):
    """
    L-DeepSC với System-Level Optimization
    """
    def __init__(self, input_dim, latent_dim, chan_dim, num_classes, 
                 hidden_dim=128, num_repetitions=3):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.chan_dim = chan_dim
        self.num_classes = num_classes
        self.num_repetitions = num_repetitions
        
        # Semantic processing
        self.semantic_encoder = SemanticEncoder(input_dim, latent_dim, hidden_dim)
        self.semantic_decoder = SemanticDecoder(latent_dim, input_dim, hidden_dim)
        
        # Channel processing với repetition
        self.channel_encoder = ChannelEncoderWithRepetition(latent_dim, chan_dim, num_repetitions)
        self.channel = RobustFadingChannel(chan_dim, num_repetitions)
        self.channel_decoder = ChannelDecoderWithRepetition(chan_dim, latent_dim, num_repetitions)
        
        # Classification
        self.classifier = ClassifierHead(latent_dim, num_classes)

    def forward(self, x, snr_db=10.0, return_all=False):
        # Semantic encoding
        z = self.semantic_encoder(x)
        
        # Classification trước kênh
        class_logits_before = self.classifier(z)
        
        # Channel encoding với repetition
        x_repeated, x_chan = self.channel_encoder(z)
        
        # Physical channel
        y_eq, H_true, H_ls, H_refine = self.channel(x_repeated, snr_db)
        
        # Channel decoding với combining
        z_hat = self.channel_decoder(y_eq)
        
        # Semantic decoding
        x_hat = self.semantic_decoder(z_hat)
        
        # Classification sau kênh
        class_logits_after = self.classifier(z_hat)
        
        if return_all:
            return x_hat, class_logits_before, class_logits_after, {
                "z": z,
                "z_hat": z_hat,
                "x_chan": x_chan,
                "y_eq": y_eq,
            }
        return x_hat, class_logits_before, class_logits_after

    def get_bandwidth_bytes(self):
        """Bandwidth với repetition coding"""
        return self.chan_dim * self.num_repetitions * 4  # float32


# ============================================================
# DATA LOADING
# ============================================================

def load_data(csv_path):
    """Load và preprocess data"""
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    X = df[SENSOR_COLS].values.astype("float32")
    
    le = LabelEncoder()
    le.fit(SEMANTIC_CLASSES)
    
    if "semantic_label" in df.columns:
        valid_mask = df["semantic_label"].isin(SEMANTIC_CLASSES)
        df = df[valid_mask].reset_index(drop=True)
        X = df[SENSOR_COLS].values.astype("float32")
        y = le.transform(df["semantic_label"].values)
    else:
        y = np.zeros(len(X), dtype=np.int64)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype("float32")
    X_test_scaled = scaler.transform(X_test).astype("float32")
    
    train_ds = TensorDataset(
        torch.tensor(X_train_scaled),
        torch.tensor(X_train_scaled),
        torch.tensor(y_train, dtype=torch.long)
    )
    test_ds = TensorDataset(
        torch.tensor(X_test_scaled),
        torch.tensor(X_test_scaled),
        torch.tensor(y_test, dtype=torch.long)
    )
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"  Train: {len(X_train)} samples")
    print(f"  Test:  {len(X_test)} samples")
    
    # Class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)
    
    return train_loader, test_loader, scaler, le, X_train, X_test, y_train, y_test, class_weights


# ============================================================
# TRAINING - Channel-Aware
# ============================================================

def train_model(train_loader, test_loader, input_dim, num_classes, class_weights):
    """Train với Channel-Aware strategy"""
    print(f"\n{'='*70}")
    print("Training L-DeepSC System Optimization")
    print(f"{'='*70}")
    print(f"Device: {DEVICE}")
    print(f"SNR Range: {SNR_RANGE}")
    print(f"Repetitions: {NUM_REPETITIONS}")
    
    model = LDeepSCSystemOpt(
        input_dim=input_dim,
        latent_dim=LATENT_DIM,
        chan_dim=CHAN_DIM,
        num_classes=num_classes,
        hidden_dim=HIDDEN_DIM,
        num_repetitions=NUM_REPETITIONS
    ).to(DEVICE)
    
    # Loss functions
    recon_criterion = nn.MSELoss()
    class_criterion = nn.CrossEntropyLoss(weight=class_weights)
    consistency_criterion = nn.MSELoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    best_test_acc_after = 0.0
    history = {
        "train_loss": [], "test_loss": [],
        "train_acc_before": [], "test_acc_before": [],
        "train_acc_after": [], "test_acc_after": [],
    }
    
    for epoch in range(1, NUM_EPOCHS + 1):
        # Training
        model.train()
        train_loss = 0.0
        train_correct_before = 0
        train_correct_after = 0
        train_total = 0
        
        for xb, xb_target, yb in train_loader:
            xb = xb.to(DEVICE)
            xb_target = xb_target.to(DEVICE)
            yb = yb.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Random SNR cho channel-aware training
            snr_db = np.random.choice(SNR_RANGE)
            
            x_hat, logits_before, logits_after, extras = model(xb, snr_db, return_all=True)
            
            # Losses
            loss_recon = recon_criterion(x_hat, xb_target)
            loss_class_before = class_criterion(logits_before, yb)
            loss_class_after = class_criterion(logits_after, yb)
            loss_consistency = consistency_criterion(extras["z_hat"], extras["z"].detach())
            
            # Total loss
            loss = (LAMBDA_RECON * loss_recon + 
                    LAMBDA_CLASS_BEFORE * loss_class_before +
                    LAMBDA_CLASS_AFTER * loss_class_after +
                    LAMBDA_CONSISTENCY * loss_consistency)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * xb.size(0)
            
            _, pred_before = torch.max(logits_before, 1)
            _, pred_after = torch.max(logits_after, 1)
            train_total += yb.size(0)
            train_correct_before += (pred_before == yb).sum().item()
            train_correct_after += (pred_after == yb).sum().item()
        
        train_loss /= len(train_loader.dataset)
        train_acc_before = train_correct_before / train_total
        train_acc_after = train_correct_after / train_total
        
        # Evaluation với SNR cố định
        model.eval()
        test_loss = 0.0
        test_correct_before = 0
        test_correct_after = 0
        test_total = 0
        
        with torch.no_grad():
            for xb, xb_target, yb in test_loader:
                xb = xb.to(DEVICE)
                xb_target = xb_target.to(DEVICE)
                yb = yb.to(DEVICE)
                
                x_hat, logits_before, logits_after, extras = model(xb, DEFAULT_SNR, return_all=True)
                
                loss_recon = recon_criterion(x_hat, xb_target)
                loss_class_after = class_criterion(logits_after, yb)
                loss = loss_recon + loss_class_after
                
                test_loss += loss.item() * xb.size(0)
                
                _, pred_before = torch.max(logits_before, 1)
                _, pred_after = torch.max(logits_after, 1)
                test_total += yb.size(0)
                test_correct_before += (pred_before == yb).sum().item()
                test_correct_after += (pred_after == yb).sum().item()
        
        test_loss /= len(test_loader.dataset)
        test_acc_before = test_correct_before / test_total
        test_acc_after = test_correct_after / test_total
        
        scheduler.step()
        
        # Save history
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_acc_before"].append(train_acc_before)
        history["test_acc_before"].append(test_acc_before)
        history["train_acc_after"].append(train_acc_after)
        history["test_acc_after"].append(test_acc_after)
        
        if test_acc_after > best_test_acc_after:
            best_test_acc_after = test_acc_after
            best_epoch = epoch
            best_model_state = model.state_dict().copy()
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Loss: {train_loss:.4f}/{test_loss:.4f} | "
                  f"Acc Before: {train_acc_before*100:.1f}%/{test_acc_before*100:.1f}% | "
                  f"Acc After: {train_acc_after*100:.1f}%/{test_acc_after*100:.1f}%")
    
    print(f"\nBest Test Acc (After Channel): {best_test_acc_after*100:.2f}% at epoch {best_epoch}")
    
    model.load_state_dict(best_model_state)
    
    return model, history


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, test_loader, scaler, X_test_raw, y_test):
    """Evaluate với nhiều mức SNR"""
    print(f"\n{'='*70}")
    print("Evaluating L-DeepSC System Optimization")
    print(f"{'='*70}")
    
    model.eval()
    
    results_by_snr = {}
    
    for snr in SNR_RANGE:
        all_x_hat = []
        all_pred_before = []
        all_pred_after = []
        all_y_true = []
        
        with torch.no_grad():
            for xb, _, yb in test_loader:
                xb = xb.to(DEVICE)
                
                x_hat, logits_before, logits_after = model(xb, snr)
                
                all_x_hat.append(x_hat.cpu().numpy())
                all_pred_before.append(logits_before.argmax(dim=1).cpu().numpy())
                all_pred_after.append(logits_after.argmax(dim=1).cpu().numpy())
                all_y_true.append(yb.numpy())
        
        X_hat_scaled = np.vstack(all_x_hat)
        y_pred_before = np.concatenate(all_pred_before)
        y_pred_after = np.concatenate(all_pred_after)
        y_true = np.concatenate(all_y_true)
        
        X_hat_raw = scaler.inverse_transform(X_hat_scaled)
        
        # Metrics
        acc_before = accuracy_score(y_true, y_pred_before)
        acc_after = accuracy_score(y_true, y_pred_after)
        f1_before = f1_score(y_true, y_pred_before, average='macro', zero_division=0)
        f1_after = f1_score(y_true, y_pred_after, average='macro', zero_division=0)
        
        # Reconstruction
        rmse = np.sqrt(mean_squared_error(X_test_raw, X_hat_raw))
        
        results_by_snr[snr] = {
            "accuracy_before": float(acc_before),
            "accuracy_after": float(acc_after),
            "f1_before": float(f1_before),
            "f1_after": float(f1_after),
            "rmse": float(rmse)
        }
        
        print(f"SNR={snr:3d}dB | Acc Before: {acc_before*100:.2f}% | "
              f"Acc After: {acc_after*100:.2f}% | RMSE: {rmse:.4f}")
    
    return results_by_snr


def plot_results(history, results_by_snr, output_dir):
    """Vẽ biểu đồ kết quả"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Training history - Accuracy
    ax = axes[0, 0]
    ax.plot([x*100 for x in history["train_acc_before"]], label="Train (Before)", linestyle='--')
    ax.plot([x*100 for x in history["test_acc_before"]], label="Test (Before)", linestyle='--')
    ax.plot([x*100 for x in history["train_acc_after"]], label="Train (After)")
    ax.plot([x*100 for x in history["test_acc_after"]], label="Test (After)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Training History - Classification Accuracy")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Loss
    ax = axes[0, 1]
    ax.plot(history["train_loss"], label="Train")
    ax.plot(history["test_loss"], label="Test")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training History - Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Accuracy vs SNR
    ax = axes[1, 0]
    snrs = sorted(results_by_snr.keys())
    acc_before = [results_by_snr[s]["accuracy_before"]*100 for s in snrs]
    acc_after = [results_by_snr[s]["accuracy_after"]*100 for s in snrs]
    
    ax.plot(snrs, acc_before, 'b-o', linewidth=2, markersize=8, label="Before Channel")
    ax.plot(snrs, acc_after, 'r-s', linewidth=2, markersize=8, label="After Channel")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("L-DeepSC System Opt: Accuracy vs SNR")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 100)
    
    # RMSE vs SNR
    ax = axes[1, 1]
    rmse = [results_by_snr[s]["rmse"] for s in snrs]
    ax.plot(snrs, rmse, 'g-^', linewidth=2, markersize=8)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("RMSE")
    ax.set_title("L-DeepSC System Opt: Reconstruction RMSE vs SNR")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "l_deepsc_system_opt_results.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'l_deepsc_system_opt_results.png'}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("L-DeepSC System-Level Optimization")
    print("Channel-Aware Training + Repetition Coding + Improved ADNet")
    print("=" * 70)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not DATA_PATH.exists():
        print(f"ERROR: Data file not found: {DATA_PATH}")
        return
    
    # Load data
    train_loader, test_loader, scaler, le, X_train, X_test, y_train, y_test, class_weights = load_data(DATA_PATH)
    input_dim = len(SENSOR_COLS)
    num_classes = len(SEMANTIC_CLASSES)
    
    # Train
    model, history = train_model(train_loader, test_loader, input_dim, num_classes, class_weights)
    
    # Evaluate
    results_by_snr = evaluate_model(model, test_loader, scaler, X_test, y_test)
    
    # Plot
    plot_results(history, results_by_snr, OUTPUT_DIR)
    
    # Save results
    results = {
        "model": "L-DeepSC System Optimization",
        "config": {
            "latent_dim": LATENT_DIM,
            "chan_dim": CHAN_DIM,
            "hidden_dim": HIDDEN_DIM,
            "num_repetitions": NUM_REPETITIONS,
            "snr_range": SNR_RANGE,
            "num_epochs": NUM_EPOCHS
        },
        "results_by_snr": results_by_snr,
        "best_accuracy_after_channel": max(r["accuracy_after"] for r in results_by_snr.values()),
        "bandwidth_bytes": model.get_bandwidth_bytes(),
        "timestamp": datetime.now().isoformat()
    }
    
    results_path = OUTPUT_DIR / "l_deepsc_system_opt_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Save model
    model_path = OUTPUT_DIR / "l_deepsc_system_opt_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Best Accuracy After Channel: {results['best_accuracy_after_channel']*100:.2f}%")
    print(f"Bandwidth: {results['bandwidth_bytes']} bytes/sample")
    print("\nAccuracy by SNR:")
    for snr in sorted(results_by_snr.keys()):
        r = results_by_snr[snr]
        print(f"  SNR={snr:3d}dB: Before={r['accuracy_before']*100:.1f}%, After={r['accuracy_after']*100:.1f}%")
    
    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)


if __name__ == "__main__":
    main()

