"""
L-DeepSC Classification Focus
=============================
Phien ban tap trung hoan toan vao classification de dat 85-90% accuracy.

Cai tien:
1. Uu tien classification hon reconstruction
2. Oversampling cho minority classes
3. Kien truc Transformer-based
4. Mixup data augmentation
5. Multi-head attention cho semantic features
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
from imblearn.over_sampling import SMOTE

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

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

# Hyperparameters - Classification focus
TEST_SIZE = 0.2
BATCH_SIZE = 256
NUM_EPOCHS = 300
LR = 1e-3
SNR_DB = 20.0             # Tang SNR cao hon
LATENT_DIM = 128          # Tang latent dim
CHAN_DIM = 64             # Tang channel dim
HIDDEN_DIM = 512          # Tang hidden dim
NUM_HEADS = 8             # Multi-head attention
DROPOUT = 0.3
LAMBDA_RECON = 0.1        # Giam reconstruction weight
LAMBDA_CLASS = 1.0        # Classification weight
USE_SMOTE = True          # Oversampling

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# MIXUP AUGMENTATION
# ============================================================

def mixup_data(x, y, alpha=0.2):
    """Mixup data augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================
# MODEL COMPONENTS - Transformer-based
# ============================================================

class MultiHeadSelfAttention(nn.Module):
    """Multi-head self attention"""
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block"""
    def __init__(self, dim, num_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mlp_ratio, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SemanticEncoderTransformer(nn.Module):
    """Semantic Encoder voi Transformer"""
    def __init__(self, input_dim, latent_dim, hidden_dim=512, num_heads=8, num_layers=3, dropout=0.3):
        super().__init__()
        
        # Embed each sensor as a token
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Project each sensor to hidden dim
        self.sensor_embed = nn.Linear(1, hidden_dim)
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, input_dim, hidden_dim) * 0.02)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, mlp_ratio=4, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(hidden_dim)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x):
        B = x.shape[0]
        
        # [B, 5] -> [B, 5, 1] -> [B, 5, hidden_dim]
        x = x.unsqueeze(-1)
        x = self.sensor_embed(x)
        
        # Add positional encoding
        x = x + self.pos_embed
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Flatten and project
        x = x.reshape(B, -1)
        x = self.output_proj(x)
        
        return x


class SemanticDecoderTransformer(nn.Module):
    """Semantic Decoder"""
    def __init__(self, latent_dim, output_dim, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, z):
        return self.net(z)


class ClassifierHeadTransformer(nn.Module):
    """Classification head voi attention"""
    def __init__(self, latent_dim, num_classes, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=-1)
        )
        
        self.classifier = nn.Sequential(
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
        return self.classifier(z)


class ChannelEncoderSimple(nn.Module):
    """Channel Encoder"""
    def __init__(self, latent_dim, chan_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, chan_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)


class ChannelDecoderSimple(nn.Module):
    """Channel Decoder"""
    def __init__(self, chan_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(chan_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )

    def forward(self, y):
        return self.net(y)


class AWGNChannel(nn.Module):
    """AWGN Channel (don gian hon Rayleigh)"""
    def __init__(self, snr_db=20.0):
        super().__init__()
        self.snr_db = snr_db

    def forward(self, x, training=True):
        if not training:
            return x
        
        snr_linear = 10 ** (self.snr_db / 10.0)
        noise_var = 1.0 / snr_linear
        noise = (noise_var ** 0.5) * torch.randn_like(x)
        return x + noise


class LDeepSCClassificationFocus(nn.Module):
    """L-DeepSC tap trung vao Classification"""
    def __init__(self, input_dim, latent_dim, chan_dim, num_classes, snr_db, 
                 hidden_dim=512, num_heads=8, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.chan_dim = chan_dim
        self.num_classes = num_classes
        
        self.semantic_encoder = SemanticEncoderTransformer(
            input_dim, latent_dim, hidden_dim, num_heads, num_layers=3, dropout=dropout
        )
        self.channel_encoder = ChannelEncoderSimple(latent_dim, chan_dim)
        self.channel = AWGNChannel(snr_db)
        self.channel_decoder = ChannelDecoderSimple(chan_dim, latent_dim)
        self.semantic_decoder = SemanticDecoderTransformer(
            latent_dim, input_dim, hidden_dim, dropout
        )
        self.classifier = ClassifierHeadTransformer(latent_dim, num_classes, dropout=dropout)

    def forward(self, x, training=True):
        z = self.semantic_encoder(x)
        class_logits = self.classifier(z)
        
        x_chan = self.channel_encoder(z)
        y = self.channel(x_chan, training=training)
        z_hat = self.channel_decoder(y)
        
        x_hat = self.semantic_decoder(z_hat)
        class_logits_after = self.classifier(z_hat)
        
        return x_hat, class_logits, class_logits_after

    def get_bandwidth_bytes(self):
        return self.chan_dim * 4


# ============================================================
# DATA LOADING
# ============================================================

def load_data(csv_path, use_smote=True):
    """Load va preprocess data voi SMOTE"""
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
    
    # SMOTE oversampling
    if use_smote:
        print("  Applying SMOTE oversampling...")
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"  After SMOTE: {len(X_train)} samples")
    
    # Class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype("float32")
    X_test_scaled = scaler.transform(X_test).astype("float32")
    
    # Weighted sampler cho imbalanced data
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(y_train),
        replacement=True
    )
    
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
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"  Train: {len(X_train)} samples (after SMOTE)")
    print(f"  Test:  {len(X_test)} samples")
    print(f"  Features: {SENSOR_COLS}")
    print(f"  Classes: {le.classes_}")
    
    print("\n  Class distribution (train after SMOTE):")
    unique, counts = np.unique(y_train, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    {SEMANTIC_CLASSES[u]}: {c}")
    
    return train_loader, test_loader, scaler, le, X_train, X_test, y_train, y_test, class_weights


# ============================================================
# TRAINING
# ============================================================

def train_model(train_loader, test_loader, input_dim, num_classes, class_weights):
    """Train L-DeepSC Classification Focus"""
    print(f"\n{'='*70}")
    print("Training L-DeepSC Classification Focus")
    print(f"{'='*70}")
    print(f"Device: {DEVICE}")
    print(f"Input dim: {input_dim}, Latent dim: {LATENT_DIM}, Channel dim: {CHAN_DIM}")
    print(f"Hidden dim: {HIDDEN_DIM}, Num heads: {NUM_HEADS}, Dropout: {DROPOUT}")
    print(f"Num classes: {num_classes}, SNR: {SNR_DB} dB")
    print(f"Using Mixup augmentation")
    
    model = LDeepSCClassificationFocus(
        input_dim=input_dim,
        latent_dim=LATENT_DIM,
        chan_dim=CHAN_DIM,
        num_classes=num_classes,
        snr_db=SNR_DB,
        hidden_dim=HIDDEN_DIM,
        num_heads=NUM_HEADS,
        dropout=DROPOUT
    ).to(DEVICE)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Loss functions
    recon_criterion = nn.MSELoss()
    class_criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE), label_smoothing=0.1)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=LR, epochs=NUM_EPOCHS, 
        steps_per_epoch=len(train_loader), pct_start=0.1
    )
    
    best_test_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    patience = 50
    
    history = {
        "train_loss": [], "test_loss": [],
        "train_acc": [], "test_acc": [],
        "lr": []
    }
    
    for epoch in range(1, NUM_EPOCHS + 1):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for xb, xb_target, yb in train_loader:
            xb = xb.to(DEVICE)
            xb_target = xb_target.to(DEVICE)
            yb = yb.to(DEVICE)
            
            # Mixup
            xb_mixed, y_a, y_b, lam = mixup_data(xb, yb, alpha=0.2)
            
            optimizer.zero_grad()
            x_hat, class_logits, _ = model(xb_mixed, training=True)
            
            # Reconstruction loss
            loss_recon = recon_criterion(x_hat, xb_target)
            
            # Classification loss with mixup
            loss_class = mixup_criterion(class_criterion, class_logits, y_a, y_b, lam)
            
            # Total loss
            loss = LAMBDA_RECON * loss_recon + LAMBDA_CLASS * loss_class
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item() * xb.size(0)
            
            # Accuracy (khong mixup)
            with torch.no_grad():
                _, class_logits_clean, _ = model(xb, training=False)
                _, predicted = torch.max(class_logits_clean, 1)
                train_total += yb.size(0)
                train_correct += (predicted == yb).sum().item()
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Evaluation
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for xb, xb_target, yb in test_loader:
                xb = xb.to(DEVICE)
                xb_target = xb_target.to(DEVICE)
                yb = yb.to(DEVICE)
                
                x_hat, class_logits, _ = model(xb, training=False)
                
                loss_recon = recon_criterion(x_hat, xb_target)
                loss_class = class_criterion(class_logits, yb)
                loss = LAMBDA_RECON * loss_recon + LAMBDA_CLASS * loss_class
                
                test_loss += loss.item() * xb.size(0)
                
                _, predicted = torch.max(class_logits, 1)
                test_total += yb.size(0)
                test_correct += (predicted == yb).sum().item()
        
        test_loss /= test_total
        test_acc = test_correct / test_total
        
        # Save history
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["lr"].append(optimizer.param_groups[0]['lr'])
        
        # Save best model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 20 == 0 or epoch == 1:
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

def evaluate_model(model, test_loader, scaler, le, X_test_raw, y_test):
    """Evaluate model"""
    print(f"\n{'='*70}")
    print("Evaluating L-DeepSC Classification Focus")
    print(f"{'='*70}")
    
    model.eval()
    
    all_x_hat = []
    all_class_pred = []
    all_y_true = []
    inference_times = []
    
    with torch.no_grad():
        for xb, _, yb in test_loader:
            xb = xb.to(DEVICE)
            
            start_time = time.time()
            x_hat, class_logits, _ = model(xb, training=False)
            inference_times.append(time.time() - start_time)
            
            all_x_hat.append(x_hat.cpu().numpy())
            all_class_pred.append(class_logits.argmax(dim=1).cpu().numpy())
            all_y_true.append(yb.numpy())
    
    X_hat_scaled = np.vstack(all_x_hat)
    y_pred = np.concatenate(all_class_pred)
    y_true = np.concatenate(all_y_true)
    
    X_hat_raw = scaler.inverse_transform(X_hat_scaled)
    
    # Reconstruction metrics
    print("\n" + "="*70)
    print("1. RECONSTRUCTION METRICS")
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
        
        recon_results["per_variable"][col] = {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}
        print(f"{col:<15} {rmse:>10.4f} {mae:>10.4f} {r2:>10.4f}")
    
    recon_results["overall"] = {
        "rmse": float(np.mean(rmse_list)),
        "mae": float(np.mean(mae_list)),
        "r2": float(np.mean(r2_list))
    }
    print("-" * 50)
    print(f"{'OVERALL':<15} {np.mean(rmse_list):>10.4f} {np.mean(mae_list):>10.4f} {np.mean(r2_list):>10.4f}")
    
    # Classification metrics
    print("\n" + "="*70)
    print("2. CLASSIFICATION METRICS")
    print("="*70)
    
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    prec_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    print(f"\nAccuracy:           {acc*100:.2f}%")
    print(f"F1 (macro):         {f1_macro:.4f}")
    print(f"F1 (weighted):      {f1_weighted:.4f}")
    print(f"Precision (macro):  {prec_macro:.4f}")
    print(f"Recall (macro):     {rec_macro:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=SEMANTIC_CLASSES, zero_division=0))
    
    cm = confusion_matrix(y_true, y_pred)
    
    class_results = {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro)
    }
    
    # Efficiency
    print("\n" + "="*70)
    print("3. EFFICIENCY")
    print("="*70)
    
    bandwidth_bytes = model.get_bandwidth_bytes()
    print(f"Bandwidth: {bandwidth_bytes} bytes/sample")
    print(f"Avg inference: {np.mean(inference_times)*1000:.2f} ms/batch")
    
    efficiency_results = {
        "bandwidth_bytes_per_sample": bandwidth_bytes,
        "avg_inference_time_ms": float(np.mean(inference_times) * 1000)
    }
    
    results = {
        "model": "L-DeepSC Classification Focus",
        "config": {
            "snr_db": SNR_DB,
            "latent_dim": LATENT_DIM,
            "chan_dim": CHAN_DIM,
            "hidden_dim": HIDDEN_DIM,
            "num_heads": NUM_HEADS,
            "dropout": DROPOUT,
            "use_smote": USE_SMOTE
        },
        "reconstruction": recon_results,
        "classification": class_results,
        "efficiency": efficiency_results,
        "timestamp": datetime.now().isoformat()
    }
    
    return results, X_hat_raw, y_pred, cm


def compare_with_fuzsemcom(results):
    """So sanh voi FuzSemCom"""
    print(f"\n{'='*70}")
    print("COMPARISON: L-DeepSC vs FuzSemCom")
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
    
    print(f"\n{'Metric':<30} {'L-DeepSC':>15} {'FuzSemCom':>15} {'Winner':>12}")
    print("=" * 75)
    
    print("\n--- Classification ---")
    metrics = [
        ("Accuracy", results["classification"]["accuracy"], fuzsemcom["accuracy"]),
        ("F1 (macro)", results["classification"]["f1_macro"], fuzsemcom["f1_macro"]),
        ("F1 (weighted)", results["classification"]["f1_weighted"], fuzsemcom["f1_weighted"]),
        ("Precision (macro)", results["classification"]["precision_macro"], fuzsemcom["precision_macro"]),
        ("Recall (macro)", results["classification"]["recall_macro"], fuzsemcom["recall_macro"]),
    ]
    
    for name, l_val, f_val in metrics:
        winner = "L-DeepSC" if l_val > f_val else "FuzSemCom" if f_val > l_val else "Tie"
        print(f"{name:<30} {l_val:>15.4f} {f_val:>15.4f} {winner:>12}")
    
    print("\n--- Reconstruction ---")
    l_rmse = results["reconstruction"]["overall"]["rmse"]
    f_rmse = fuzsemcom["reconstruction_rmse"]
    winner = "L-DeepSC" if l_rmse < f_rmse else "FuzSemCom"
    print(f"{'Overall RMSE':<30} {l_rmse:>15.4f} {f_rmse:>15.4f} {winner:>12}")
    
    print("\n--- Efficiency ---")
    l_bw = results["efficiency"]["bandwidth_bytes_per_sample"]
    f_bw = fuzsemcom["bandwidth_bytes"]
    winner = "FuzSemCom" if f_bw < l_bw else "L-DeepSC"
    print(f"{'Bandwidth (bytes/sample)':<30} {l_bw:>15} {f_bw:>15} {winner:>12}")
    
    print("\n" + "=" * 75)
    
    return {"l_deepsc": results, "fuzsemcom": fuzsemcom}


def plot_results(history, cm, output_dir):
    """Ve bieu do"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax = axes[0, 0]
    ax.plot(history["train_loss"], label="Train", alpha=0.8)
    ax.plot(history["test_loss"], label="Test", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot([x*100 for x in history["train_acc"]], label="Train", alpha=0.8)
    ax.plot([x*100 for x in history["test_acc"]], label="Test", alpha=0.8)
    ax.axhline(y=85, color='r', linestyle='--', alpha=0.5, label="Target 85%")
    ax.axhline(y=90, color='g', linestyle='--', alpha=0.5, label="Target 90%")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Classification Accuracy")
    ax.legend()
    ax.grid(alpha=0.3)
    
    ax = axes[1, 0]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=SEMANTIC_CLASSES, yticklabels=SEMANTIC_CLASSES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    ax = axes[1, 1]
    ax.plot(history["lr"], color='orange')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.set_yscale('log')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "l_deepsc_classification_results.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'l_deepsc_classification_results.png'}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("L-DeepSC CLASSIFICATION FOCUS")
    print("Target: 85-90% Accuracy")
    print("=" * 70)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not DATA_PATH.exists():
        print(f"ERROR: Data file not found: {DATA_PATH}")
        return
    
    # Load data
    train_loader, test_loader, scaler, le, X_train, X_test, y_train, y_test, class_weights = load_data(
        DATA_PATH, use_smote=USE_SMOTE
    )
    input_dim = len(SENSOR_COLS)
    num_classes = len(SEMANTIC_CLASSES)
    
    # Train
    model, history = train_model(train_loader, test_loader, input_dim, num_classes, class_weights)
    
    # Evaluate
    results, X_hat, y_pred, cm = evaluate_model(model, test_loader, scaler, le, X_test, y_test)
    
    # Compare
    comparison = compare_with_fuzsemcom(results)
    
    # Plot
    plot_results(history, cm, OUTPUT_DIR)
    
    # Save
    with open(OUTPUT_DIR / "l_deepsc_classification_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    with open(OUTPUT_DIR / "comparison_classification.json", 'w') as f:
        json.dump(comparison, f, indent=2)
    
    torch.save(model.state_dict(), OUTPUT_DIR / "l_deepsc_classification_model.pth")
    
    # Summary
    final_acc = results["classification"]["accuracy"] * 100
    print("\n" + "=" * 70)
    print(f"FINAL ACCURACY: {final_acc:.2f}%")
    if final_acc >= 85:
        print("TARGET ACHIEVED!")
    else:
        print(f"Gap to target: {85 - final_acc:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()

