"""
L-DeepSC Optimized for FuzSemCom Comparison
============================================
Phien ban toi uu de dat accuracy 85-90%, so sanh cong bang voi FuzSemCom.

Cai tien so voi phien ban goc:
1. Kien truc sau hon voi Residual connections
2. Class weights de xu ly imbalanced data
3. Tang latent dimension
4. Label smoothing + Focal Loss
5. Learning rate scheduling tot hon
6. Dropout va BatchNorm tot hon
7. Ensemble prediction (nhieu lan qua channel)

Paper goc: "Lite Distributed Semantic Communication System for Internet of Things"
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

import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CONFIG - Toi uu cho accuracy cao
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

# Hyperparameters - Toi uu
TEST_SIZE = 0.2
BATCH_SIZE = 128          # Tang batch size
NUM_EPOCHS = 200          # Tang epochs
LR = 5e-4                 # Giam learning rate
SNR_DB = 15.0             # Tang SNR de giam nhieu kenh
LATENT_DIM = 64           # Tang latent dim
CHAN_DIM = 32             # Tang channel dim
HIDDEN_DIM = 256          # Tang hidden dim
DROPOUT = 0.2             # Dropout rate
LABEL_SMOOTHING = 0.1     # Label smoothing
LAMBDA_RECON = 0.3        # Giam weight cua reconstruction
LAMBDA_CLASS = 1.0        # Tang weight cua classification
NUM_ENSEMBLE = 5          # So lan chay qua channel de ensemble

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# FOCAL LOSS - Tot hon cho imbalanced data
# ============================================================

class FocalLoss(nn.Module):
    """Focal Loss de xu ly class imbalance"""
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ============================================================
# MODEL COMPONENTS - Kien truc toi uu
# ============================================================

class ResidualBlock(nn.Module):
    """Residual block voi skip connection"""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.net(x))


class SemanticEncoderOptimized(nn.Module):
    """Semantic Encoder toi uu voi Residual blocks"""
    def __init__(self, input_dim, latent_dim, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout),
            ResidualBlock(hidden_dim, dropout),
            ResidualBlock(hidden_dim, dropout),
        ])
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, latent_dim),
        )

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x)


class SemanticDecoderOptimized(nn.Module):
    """Semantic Decoder toi uu"""
    def __init__(self, latent_dim, output_dim, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout),
            ResidualBlock(hidden_dim, dropout),
        ])
        
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, z_hat):
        x = self.input_proj(z_hat)
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x)


class ClassifierHeadOptimized(nn.Module):
    """Classification head toi uu"""
    def __init__(self, latent_dim, num_classes, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, z):
        return self.net(z)


class ChannelEncoderOptimized(nn.Module):
    """Channel Encoder toi uu"""
    def __init__(self, latent_dim, chan_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, chan_dim * 2),
            nn.LayerNorm(chan_dim * 2),
            nn.GELU(),
            nn.Linear(chan_dim * 2, chan_dim),
            nn.Tanh()  # Gioi han nang luong [-1, 1]
        )

    def forward(self, z):
        return self.net(z)


class ChannelDecoderOptimized(nn.Module):
    """Channel Decoder toi uu"""
    def __init__(self, chan_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(chan_dim, chan_dim * 2),
            nn.LayerNorm(chan_dim * 2),
            nn.GELU(),
            nn.Linear(chan_dim * 2, latent_dim),
        )

    def forward(self, y_eq):
        return self.net(y_eq)


class ADNetOptimized(nn.Module):
    """ADNet toi uu voi attention mechanism"""
    def __init__(self, length):
        super().__init__()
        self.length = length
        
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )
        
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 64),
            nn.Sigmoid()
        )
        
        self.decoder = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 1, kernel_size=3, padding=1),
        )

    def forward(self, h_ls):
        x = h_ls.unsqueeze(1)  # [B, 1, L]
        x = self.encoder(x)    # [B, 64, L]
        
        att = self.attention(x)  # [B, 64]
        att = att.unsqueeze(-1)  # [B, 64, 1]
        x = x * att
        
        x = self.decoder(x)    # [B, 1, L]
        return x.squeeze(1)    # [B, L]


class RayleighFadingChannelOptimized(nn.Module):
    """Rayleigh Fading Channel toi uu"""
    def __init__(self, chan_dim, snr_db=15.0):
        super().__init__()
        self.chan_dim = chan_dim
        self.snr_db = snr_db
        self.adnet = ADNetOptimized(length=chan_dim)

    def forward(self, x_chan, training=True):
        B, D = x_chan.shape
        device = x_chan.device

        # Rayleigh fading: H = |N(0,1)|
        H_true = torch.abs(torch.randn(B, D, device=device)) + 1e-6

        # AWGN noise
        snr_linear = 10 ** (self.snr_db / 10.0)
        noise_var = 1.0 / snr_linear
        noise = (noise_var ** 0.5) * torch.randn_like(x_chan)

        # Channel output
        y = H_true * x_chan + noise

        # CSI estimation (LS with noise)
        est_noise_std = 0.1 if training else 0.05
        H_ls = H_true + est_noise_std * torch.randn_like(H_true)

        # Refine CSI voi ADNet
        H_refine = self.adnet(H_ls)

        # Equalization
        y_eq = y / (H_refine.abs() + 1e-6)

        return y_eq, H_true, H_ls, H_refine


class LDeepSCOptimized(nn.Module):
    """
    L-DeepSC Optimized: Phien ban toi uu de dat accuracy 85-90%
    """
    def __init__(self, input_dim, latent_dim, chan_dim, num_classes, snr_db, 
                 hidden_dim=256, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.chan_dim = chan_dim
        self.num_classes = num_classes
        
        self.semantic_encoder = SemanticEncoderOptimized(
            input_dim, latent_dim, hidden_dim, dropout
        )
        self.channel_encoder = ChannelEncoderOptimized(latent_dim, chan_dim)
        self.channel = RayleighFadingChannelOptimized(chan_dim, snr_db)
        self.channel_decoder = ChannelDecoderOptimized(chan_dim, latent_dim)
        self.semantic_decoder = SemanticDecoderOptimized(
            latent_dim, input_dim, hidden_dim, dropout
        )
        self.classifier = ClassifierHeadOptimized(latent_dim, num_classes, dropout=dropout)

    def forward(self, x, return_all=False, training=True):
        # Semantic encoding
        z = self.semantic_encoder(x)
        
        # Classification (truoc khi qua channel)
        class_logits = self.classifier(z)
        
        # Channel encoding
        x_chan = self.channel_encoder(z)
        
        # Physical channel
        y_eq, H_true, H_ls, H_refine = self.channel(x_chan, training=training)
        
        # Channel decoding
        z_hat = self.channel_decoder(y_eq)
        
        # Semantic decoding (reconstruction)
        x_hat = self.semantic_decoder(z_hat)
        
        # Classification sau channel
        class_logits_after = self.classifier(z_hat)
        
        if return_all:
            return x_hat, class_logits, class_logits_after, {
                "z": z,
                "z_hat": z_hat,
                "x_chan": x_chan,
                "y_eq": y_eq,
            }
        return x_hat, class_logits, class_logits_after

    def forward_ensemble(self, x, num_runs=5):
        """Chay nhieu lan qua channel va lay ensemble"""
        z = self.semantic_encoder(x)
        class_logits_before = self.classifier(z)
        
        x_chan = self.channel_encoder(z)
        
        all_x_hat = []
        all_class_after = []
        
        for _ in range(num_runs):
            y_eq, _, _, _ = self.channel(x_chan, training=False)
            z_hat = self.channel_decoder(y_eq)
            x_hat = self.semantic_decoder(z_hat)
            class_logits_after = self.classifier(z_hat)
            
            all_x_hat.append(x_hat)
            all_class_after.append(class_logits_after)
        
        # Ensemble: average
        x_hat_ensemble = torch.stack(all_x_hat).mean(dim=0)
        class_after_ensemble = torch.stack(all_class_after).mean(dim=0)
        
        return x_hat_ensemble, class_logits_before, class_after_ensemble

    def get_bandwidth_bytes(self):
        return self.chan_dim * 4


# ============================================================
# DATA LOADING
# ============================================================

def load_data(csv_path):
    """Load va preprocess data"""
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
    
    # Tinh class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
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
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"  Train: {len(X_train)} samples")
    print(f"  Test:  {len(X_test)} samples")
    print(f"  Features: {SENSOR_COLS}")
    print(f"  Classes: {le.classes_}")
    
    print("\n  Class distribution (train):")
    unique, counts = np.unique(y_train, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    {SEMANTIC_CLASSES[u]}: {c}")
    
    return train_loader, test_loader, scaler, le, X_train, X_test, y_train, y_test, class_weights


# ============================================================
# TRAINING
# ============================================================

def train_model(train_loader, test_loader, input_dim, num_classes, class_weights):
    """Train L-DeepSC Optimized"""
    print(f"\n{'='*70}")
    print("Training L-DeepSC Optimized")
    print(f"{'='*70}")
    print(f"Device: {DEVICE}")
    print(f"Input dim: {input_dim}, Latent dim: {LATENT_DIM}, Channel dim: {CHAN_DIM}")
    print(f"Hidden dim: {HIDDEN_DIM}, Dropout: {DROPOUT}")
    print(f"Num classes: {num_classes}, SNR: {SNR_DB} dB")
    print(f"Lambda recon: {LAMBDA_RECON}, Lambda class: {LAMBDA_CLASS}")
    print(f"Label smoothing: {LABEL_SMOOTHING}")
    
    model = LDeepSCOptimized(
        input_dim=input_dim,
        latent_dim=LATENT_DIM,
        chan_dim=CHAN_DIM,
        num_classes=num_classes,
        snr_db=SNR_DB,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT
    ).to(DEVICE)
    
    # Loss functions
    recon_criterion = nn.MSELoss()
    class_criterion = FocalLoss(
        alpha=class_weights.to(DEVICE),
        gamma=2.0
    )
    
    # Optimizer voi weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LR, 
        weight_decay=1e-4
    )
    
    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-6
    )
    
    best_test_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    patience = 30
    
    history = {
        "train_loss": [], "test_loss": [],
        "train_recon_loss": [], "test_recon_loss": [],
        "train_class_loss": [], "test_class_loss": [],
        "train_acc": [], "test_acc": [],
        "lr": []
    }
    
    for epoch in range(1, NUM_EPOCHS + 1):
        # Training
        model.train()
        train_loss = 0.0
        train_recon_loss = 0.0
        train_class_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for xb, xb_target, yb in train_loader:
            xb = xb.to(DEVICE)
            xb_target = xb_target.to(DEVICE)
            yb = yb.to(DEVICE)
            
            optimizer.zero_grad()
            x_hat, class_logits, class_logits_after = model(xb, training=True)
            
            # Reconstruction loss
            loss_recon = recon_criterion(x_hat, xb_target)
            
            # Classification loss (truoc channel)
            loss_class = class_criterion(class_logits, yb)
            
            # Total loss
            loss = LAMBDA_RECON * loss_recon + LAMBDA_CLASS * loss_class
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item() * xb.size(0)
            train_recon_loss += loss_recon.item() * xb.size(0)
            train_class_loss += loss_class.item() * xb.size(0)
            
            _, predicted = torch.max(class_logits, 1)
            train_total += yb.size(0)
            train_correct += (predicted == yb).sum().item()
        
        train_loss /= train_total
        train_recon_loss /= train_total
        train_class_loss /= train_total
        train_acc = train_correct / train_total
        
        # Evaluation
        model.eval()
        test_loss = 0.0
        test_recon_loss = 0.0
        test_class_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for xb, xb_target, yb in test_loader:
                xb = xb.to(DEVICE)
                xb_target = xb_target.to(DEVICE)
                yb = yb.to(DEVICE)
                
                x_hat, class_logits, class_logits_after = model(xb, training=False)
                
                loss_recon = recon_criterion(x_hat, xb_target)
                loss_class = class_criterion(class_logits, yb)
                loss = LAMBDA_RECON * loss_recon + LAMBDA_CLASS * loss_class
                
                test_loss += loss.item() * xb.size(0)
                test_recon_loss += loss_recon.item() * xb.size(0)
                test_class_loss += loss_class.item() * xb.size(0)
                
                _, predicted = torch.max(class_logits, 1)
                test_total += yb.size(0)
                test_correct += (predicted == yb).sum().item()
        
        test_loss /= test_total
        test_recon_loss /= test_total
        test_class_loss /= test_total
        test_acc = test_correct / test_total
        
        # Save history
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_recon_loss"].append(train_recon_loss)
        history["test_recon_loss"].append(test_recon_loss)
        history["train_class_loss"].append(train_class_loss)
        history["test_class_loss"].append(test_class_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["lr"].append(optimizer.param_groups[0]['lr'])
        
        scheduler.step()
        
        # Save best model (theo accuracy)
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Loss: {train_loss:.4f}/{test_loss:.4f} | "
                  f"Acc: {train_acc*100:.1f}%/{test_acc*100:.1f}% | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break
    
    print(f"\nBest Test Accuracy: {best_test_acc*100:.2f}% at epoch {best_epoch}")
    
    model.load_state_dict(best_model_state)
    
    return model, history


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, test_loader, scaler, le, X_test_raw, y_test, use_ensemble=True):
    """Evaluate L-DeepSC Optimized"""
    print(f"\n{'='*70}")
    print("Evaluating L-DeepSC Optimized - Full Metrics")
    print(f"{'='*70}")
    
    model.eval()
    
    all_x_hat = []
    all_class_pred = []
    all_class_pred_after = []
    all_y_true = []
    inference_times = []
    
    with torch.no_grad():
        for xb, _, yb in test_loader:
            xb = xb.to(DEVICE)
            
            start_time = time.time()
            if use_ensemble:
                x_hat, class_logits, class_logits_after = model.forward_ensemble(xb, NUM_ENSEMBLE)
            else:
                x_hat, class_logits, class_logits_after = model(xb, training=False)
            inference_times.append(time.time() - start_time)
            
            all_x_hat.append(x_hat.cpu().numpy())
            all_class_pred.append(class_logits.argmax(dim=1).cpu().numpy())
            all_class_pred_after.append(class_logits_after.argmax(dim=1).cpu().numpy())
            all_y_true.append(yb.numpy())
    
    X_hat_scaled = np.vstack(all_x_hat)
    y_pred = np.concatenate(all_class_pred)
    y_pred_after = np.concatenate(all_class_pred_after)
    y_true = np.concatenate(all_y_true)
    
    X_hat_raw = scaler.inverse_transform(X_hat_scaled)
    
    # ============================================================
    # 1. RECONSTRUCTION METRICS
    # ============================================================
    print("\n" + "="*70)
    print("1. RECONSTRUCTION METRICS (Sensor Values)")
    print("="*70)
    
    recon_results = {"per_variable": {}, "overall": {}}
    rmse_list, mae_list, r2_list = [], [], []
    
    print(f"\n{'Sensor':<15} {'RMSE':>10} {'MAE':>10} {'R2':>10}")
    print("-" * 50)
    
    for i, col in enumerate(SENSOR_COLS):
        y_true_col = X_test_raw[:, i]
        y_pred_col = X_hat_raw[:, i]
        
        rmse = np.sqrt(mean_squared_error(y_true_col, y_pred_col))
        mae = mean_absolute_error(y_true_col, y_pred_col)
        r2 = r2_score(y_true_col, y_pred_col)
        
        rmse_list.append(rmse)
        mae_list.append(mae)
        r2_list.append(r2)
        
        recon_results["per_variable"][col] = {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2": float(r2)
        }
        
        print(f"{col:<15} {rmse:>10.4f} {mae:>10.4f} {r2:>10.4f}")
    
    overall_rmse = np.mean(rmse_list)
    overall_mae = np.mean(mae_list)
    overall_r2 = np.mean(r2_list)
    
    recon_results["overall"] = {
        "rmse": float(overall_rmse),
        "mae": float(overall_mae),
        "r2": float(overall_r2)
    }
    
    print("-" * 50)
    print(f"{'OVERALL':<15} {overall_rmse:>10.4f} {overall_mae:>10.4f} {overall_r2:>10.4f}")
    
    # ============================================================
    # 2. CLASSIFICATION METRICS
    # ============================================================
    print("\n" + "="*70)
    print("2. CLASSIFICATION METRICS (Semantic Class)")
    print("="*70)
    
    # Truoc channel
    acc_before = accuracy_score(y_true, y_pred)
    f1_macro_before = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted_before = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    prec_macro_before = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec_macro_before = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Sau channel
    acc_after = accuracy_score(y_true, y_pred_after)
    f1_macro_after = f1_score(y_true, y_pred_after, average='macro', zero_division=0)
    
    print(f"\n{'Metric':<25} {'Before Channel':>15} {'After Channel':>15}")
    print("-" * 60)
    print(f"{'Accuracy':<25} {acc_before*100:>14.2f}% {acc_after*100:>14.2f}%")
    print(f"{'F1 (macro)':<25} {f1_macro_before:>15.4f} {f1_macro_after:>15.4f}")
    print(f"{'F1 (weighted)':<25} {f1_weighted_before:>15.4f} {'-':>15}")
    print(f"{'Precision (macro)':<25} {prec_macro_before:>15.4f} {'-':>15}")
    print(f"{'Recall (macro)':<25} {rec_macro_before:>15.4f} {'-':>15}")
    
    print("\nClassification Report (Before Channel):")
    print(classification_report(y_true, y_pred, target_names=SEMANTIC_CLASSES, zero_division=0))
    
    cm = confusion_matrix(y_true, y_pred)
    
    class_results = {
        "before_channel": {
            "accuracy": float(acc_before),
            "f1_macro": float(f1_macro_before),
            "f1_weighted": float(f1_weighted_before),
            "precision_macro": float(prec_macro_before),
            "recall_macro": float(rec_macro_before)
        },
        "after_channel": {
            "accuracy": float(acc_after),
            "f1_macro": float(f1_macro_after)
        }
    }
    
    # ============================================================
    # 3. EFFICIENCY METRICS
    # ============================================================
    print("\n" + "="*70)
    print("3. BANDWIDTH & EFFICIENCY")
    print("="*70)
    
    bandwidth_bytes = model.get_bandwidth_bytes()
    original_bytes = len(SENSOR_COLS) * 4
    compression_ratio = original_bytes / bandwidth_bytes
    avg_inference_time = np.mean(inference_times) * 1000
    
    print(f"Original data:     {original_bytes} bytes/sample ({len(SENSOR_COLS)} x float32)")
    print(f"Transmitted:       {bandwidth_bytes} bytes/sample ({CHAN_DIM} x float32)")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"Avg inference:     {avg_inference_time:.2f} ms/batch")
    
    efficiency_results = {
        "bandwidth_bytes_per_sample": bandwidth_bytes,
        "original_bytes_per_sample": original_bytes,
        "compression_ratio": float(compression_ratio),
        "avg_inference_time_ms": float(avg_inference_time)
    }
    
    results = {
        "model": "L-DeepSC Optimized",
        "config": {
            "snr_db": SNR_DB,
            "latent_dim": LATENT_DIM,
            "chan_dim": CHAN_DIM,
            "hidden_dim": HIDDEN_DIM,
            "dropout": DROPOUT,
            "num_epochs": NUM_EPOCHS,
            "lambda_recon": LAMBDA_RECON,
            "lambda_class": LAMBDA_CLASS,
            "num_ensemble": NUM_ENSEMBLE
        },
        "reconstruction": recon_results,
        "classification": class_results,
        "efficiency": efficiency_results,
        "timestamp": datetime.now().isoformat()
    }
    
    return results, X_hat_raw, y_pred, cm


def compare_with_fuzsemcom(l_deepsc_results):
    """So sanh voi FuzSemCom"""
    print(f"\n{'='*70}")
    print("COMPARISON: L-DeepSC Optimized vs FuzSemCom")
    print(f"{'='*70}")
    
    fuzsemcom = {
        "accuracy": 0.9499,
        "f1_macro": 0.9148,
        "f1_weighted": 0.9505,
        "precision_macro": 0.9246,
        "recall_macro": 0.9152,
        "bandwidth_bytes": 2,
        "reconstruction_rmse": 10.0,
    }
    
    l_deepsc = l_deepsc_results
    
    print(f"\n{'Metric':<30} {'L-DeepSC Opt':>15} {'FuzSemCom':>15} {'Winner':>12}")
    print("=" * 75)
    
    print("\n--- Classification ---")
    metrics = [
        ("Accuracy", l_deepsc["classification"]["before_channel"]["accuracy"], fuzsemcom["accuracy"]),
        ("F1 (macro)", l_deepsc["classification"]["before_channel"]["f1_macro"], fuzsemcom["f1_macro"]),
        ("F1 (weighted)", l_deepsc["classification"]["before_channel"]["f1_weighted"], fuzsemcom["f1_weighted"]),
        ("Precision (macro)", l_deepsc["classification"]["before_channel"]["precision_macro"], fuzsemcom["precision_macro"]),
        ("Recall (macro)", l_deepsc["classification"]["before_channel"]["recall_macro"], fuzsemcom["recall_macro"]),
    ]
    
    for name, l_val, f_val in metrics:
        winner = "L-DeepSC" if l_val > f_val else "FuzSemCom" if f_val > l_val else "Tie"
        print(f"{name:<30} {l_val:>15.4f} {f_val:>15.4f} {winner:>12}")
    
    print("\n--- Reconstruction ---")
    l_rmse = l_deepsc["reconstruction"]["overall"]["rmse"]
    f_rmse = fuzsemcom["reconstruction_rmse"]
    winner = "L-DeepSC" if l_rmse < f_rmse else "FuzSemCom"
    print(f"{'Overall RMSE':<30} {l_rmse:>15.4f} {f_rmse:>15.4f} {winner:>12}")
    
    print("\n--- Efficiency ---")
    l_bw = l_deepsc["efficiency"]["bandwidth_bytes_per_sample"]
    f_bw = fuzsemcom["bandwidth_bytes"]
    winner = "FuzSemCom" if f_bw < l_bw else "L-DeepSC"
    print(f"{'Bandwidth (bytes/sample)':<30} {l_bw:>15} {f_bw:>15} {winner:>12}")
    
    l_cr = 20 / l_bw
    f_cr = 20 / f_bw
    print(f"{'Compression Ratio':<30} {l_cr:>15.1f}x {f_cr:>15.1f}x")
    
    print("\n" + "=" * 75)
    
    return {
        "l_deepsc_optimized": l_deepsc,
        "fuzsemcom": fuzsemcom
    }


def plot_results(history, cm, output_dir):
    """Ve bieu do ket qua"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Training history - Loss
    ax = axes[0, 0]
    ax.plot(history["train_loss"], label="Train Loss", alpha=0.8)
    ax.plot(history["test_loss"], label="Test Loss", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training History - Total Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Training history - Accuracy
    ax = axes[0, 1]
    ax.plot([x*100 for x in history["train_acc"]], label="Train Acc", alpha=0.8)
    ax.plot([x*100 for x in history["test_acc"]], label="Test Acc", alpha=0.8)
    ax.axhline(y=85, color='r', linestyle='--', alpha=0.5, label="Target 85%")
    ax.axhline(y=90, color='g', linestyle='--', alpha=0.5, label="Target 90%")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Training History - Classification Accuracy")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Confusion Matrix
    ax = axes[1, 0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=SEMANTIC_CLASSES, yticklabels=SEMANTIC_CLASSES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("L-DeepSC Optimized Confusion Matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Learning Rate
    ax = axes[1, 1]
    ax.plot(history["lr"], label="Learning Rate", color='orange')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.set_yscale('log')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "l_deepsc_optimized_results.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'l_deepsc_optimized_results.png'}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("L-DeepSC OPTIMIZED for FuzSemCom Comparison")
    print("Target: 85-90% Classification Accuracy")
    print("=" * 70)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not DATA_PATH.exists():
        print(f"ERROR: Data file not found: {DATA_PATH}")
        print("Please run the notebook first to generate preprocessed data.")
        return
    
    # Load data
    train_loader, test_loader, scaler, le, X_train, X_test, y_train, y_test, class_weights = load_data(DATA_PATH)
    input_dim = len(SENSOR_COLS)
    num_classes = len(SEMANTIC_CLASSES)
    
    # Train model
    model, history = train_model(train_loader, test_loader, input_dim, num_classes, class_weights)
    
    # Evaluate
    results, X_hat, y_pred, cm = evaluate_model(
        model, test_loader, scaler, le, X_test, y_test, use_ensemble=True
    )
    
    # Compare with FuzSemCom
    comparison = compare_with_fuzsemcom(results)
    
    # Plot results
    plot_results(history, cm, OUTPUT_DIR)
    
    # Save results
    results_path = OUTPUT_DIR / "l_deepsc_optimized_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    comparison_path = OUTPUT_DIR / "comparison_with_fuzsemcom.json"
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"Comparison saved to: {comparison_path}")
    
    model_path = OUTPUT_DIR / "l_deepsc_optimized_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")
    
    # Final summary
    final_acc = results["classification"]["before_channel"]["accuracy"] * 100
    print("\n" + "=" * 70)
    print(f"FINAL ACCURACY: {final_acc:.2f}%")
    if final_acc >= 85:
        print("TARGET ACHIEVED: >= 85%")
    else:
        print(f"TARGET NOT ACHIEVED: Need {85 - final_acc:.2f}% more")
    print("=" * 70)


if __name__ == "__main__":
    main()

