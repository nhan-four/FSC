"""
L-DeepSC Baseline for FuzSemCom Comparison
==========================================
Phiên bản điều chỉnh để so sánh TOÀN DIỆN với FuzSemCom trên cùng dataset.

So sánh đầy đủ các metrics:
1. RECONSTRUCTION METRICS: RMSE, MAE, R² cho từng sensor + overall
2. CLASSIFICATION METRICS: Accuracy, F1, Precision, Recall (nếu thêm classifier head)
3. BANDWIDTH/COMPRESSION: Số bytes truyền qua kênh
4. LATENCY: Thời gian inference

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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CONFIG - Điều chỉnh để khớp với FuzSemCom
# ============================================================

# Đường dẫn
PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "semantic_dataset_preprocessed.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "l_deepsc_comparison"

# Các cột sensor giống FuzSemCom
SENSOR_COLS = ["Moisture", "pH", "N", "Temperature", "Humidity"]

# Semantic classes (giống FuzSemCom)
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

# Hyperparameters
TEST_SIZE = 0.2
BATCH_SIZE = 64
NUM_EPOCHS = 100
LR = 1e-3
SNR_DB = 10.0           # SNR cho kênh fading
LATENT_DIM = 16         # Kích thước semantic latent
CHAN_DIM = 16           # Kích thước vector sau Channel Encoder
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# MODEL COMPONENTS (theo paper L-DeepSC)
# ============================================================

class SemanticEncoder(nn.Module):
    """Semantic Encoder: Nén sensor data thành semantic representation"""
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )

    def forward(self, x):
        return self.net(x)


class SemanticDecoder(nn.Module):
    """Semantic Decoder: Khôi phục sensor data từ semantic representation"""
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, z_hat):
        return self.net(z_hat)


class ClassifierHead(nn.Module):
    """Classification head để predict semantic class (cho so sánh accuracy với FSE)"""
    def __init__(self, latent_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, z):
        return self.net(z)


class ChannelEncoder(nn.Module):
    """Channel Encoder: Chuẩn bị signal cho truyền qua kênh vật lý"""
    def __init__(self, latent_dim, chan_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, chan_dim),
            nn.Tanh()  # Giới hạn năng lượng [-1, 1]
        )

    def forward(self, z):
        return self.net(z)


class ChannelDecoder(nn.Module):
    """Channel Decoder: Khôi phục semantic từ signal nhận được"""
    def __init__(self, chan_dim, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(chan_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim)
        )

    def forward(self, y_eq):
        return self.net(y_eq)


class ADNetBlock(nn.Module):
    """Residual block cho ADNet"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = out + x  # Residual connection
        return self.relu(out)


class ADNet(nn.Module):
    """Attention-guided Denoising Network (ADNet) cho CSI refinement"""
    def __init__(self, length):
        super().__init__()
        self.length = length
        self.in_conv = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.block1 = ADNetBlock(16)
        self.block2 = ADNetBlock(16)
        self.block3 = ADNetBlock(16)

        self.attention_fc = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.Sigmoid()
        )

        self.out_conv = nn.Conv1d(16, 1, kernel_size=3, padding=1)

    def forward(self, h_ls):
        x = h_ls.unsqueeze(1)
        x = self.in_conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        gap = x.mean(dim=2)
        att = self.attention_fc(gap)
        att = att.unsqueeze(-1)
        x = x * att

        x = self.out_conv(x)
        return x.squeeze(1)


class RayleighFadingChannel(nn.Module):
    """Rayleigh Fading Channel với CSI estimation và ADNet"""
    def __init__(self, chan_dim, snr_db=10.0):
        super().__init__()
        self.chan_dim = chan_dim
        self.snr_db = snr_db
        self.adnet = ADNet(length=chan_dim)

    def forward(self, x_chan):
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
        est_noise_std = 0.2
        H_ls = H_true + est_noise_std * torch.randn_like(H_true)

        # Refine CSI với ADNet
        H_refine = self.adnet(H_ls)

        # Equalization
        y_eq = y / (H_refine.abs() + 1e-6)

        return y_eq, H_true, H_ls, H_refine


class LDeepSC(nn.Module):
    """
    L-DeepSC: Lite Distributed Semantic Communication
    
    Bao gồm cả reconstruction và classification để so sánh với FuzSemCom
    """
    def __init__(self, input_dim, latent_dim, chan_dim, num_classes, snr_db):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.chan_dim = chan_dim
        self.num_classes = num_classes
        
        self.semantic_encoder = SemanticEncoder(input_dim, latent_dim)
        self.channel_encoder = ChannelEncoder(latent_dim, chan_dim)
        self.channel = RayleighFadingChannel(chan_dim, snr_db)
        self.channel_decoder = ChannelDecoder(chan_dim, latent_dim)
        self.semantic_decoder = SemanticDecoder(latent_dim, input_dim)
        self.classifier = ClassifierHead(latent_dim, num_classes)

    def forward(self, x, return_all=False):
        # Semantic encoding
        z = self.semantic_encoder(x)
        
        # Classification (trước khi qua channel)
        class_logits = self.classifier(z)
        
        # Channel encoding
        x_chan = self.channel_encoder(z)
        
        # Physical channel
        y_eq, H_true, H_ls, H_refine = self.channel(x_chan)
        
        # Channel decoding
        z_hat = self.channel_decoder(y_eq)
        
        # Semantic decoding (reconstruction)
        x_hat = self.semantic_decoder(z_hat)
        
        # Classification sau channel (để đo channel effect)
        class_logits_after = self.classifier(z_hat)
        
        if return_all:
            return x_hat, class_logits, class_logits_after, {
                "z": z,
                "z_hat": z_hat,
                "x_chan": x_chan,
                "y_eq": y_eq,
            }
        return x_hat, class_logits, class_logits_after

    def get_bandwidth_bytes(self):
        """Tính số bytes truyền qua kênh"""
        return self.chan_dim * 4  # float32


# ============================================================
# DATA LOADING
# ============================================================

def load_data(csv_path):
    """Load và preprocess data giống FuzSemCom"""
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Sensor features
    X = df[SENSOR_COLS].values.astype("float32")
    
    # Labels (semantic class)
    le = LabelEncoder()
    le.fit(SEMANTIC_CLASSES)
    
    if "semantic_label" in df.columns:
        # Filter chỉ lấy các labels có trong SEMANTIC_CLASSES
        valid_mask = df["semantic_label"].isin(SEMANTIC_CLASSES)
        df = df[valid_mask].reset_index(drop=True)
        X = df[SENSOR_COLS].values.astype("float32")
        y = le.transform(df["semantic_label"].values)
    else:
        # Nếu không có label, tạo dummy
        y = np.zeros(len(X), dtype=np.int64)
    
    # Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y
    )
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype("float32")
    X_test_scaled = scaler.transform(X_test).astype("float32")
    
    # Create DataLoaders
    train_ds = TensorDataset(
        torch.tensor(X_train_scaled),
        torch.tensor(X_train_scaled),  # Reconstruction target
        torch.tensor(y_train, dtype=torch.long)  # Classification target
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
    print(f"  Features: {SENSOR_COLS}")
    print(f"  Classes: {le.classes_}")
    
    # Class distribution
    print("\n  Class distribution (train):")
    unique, counts = np.unique(y_train, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    {SEMANTIC_CLASSES[u]}: {c}")
    
    return train_loader, test_loader, scaler, le, X_train, X_test, y_train, y_test


# ============================================================
# TRAINING
# ============================================================

def train_model(train_loader, test_loader, input_dim, num_classes):
    """Train L-DeepSC model với cả reconstruction và classification loss"""
    print(f"\n{'='*70}")
    print("Training L-DeepSC Model")
    print(f"{'='*70}")
    print(f"Device: {DEVICE}")
    print(f"Input dim: {input_dim}, Latent dim: {LATENT_DIM}, Channel dim: {CHAN_DIM}")
    print(f"Num classes: {num_classes}, SNR: {SNR_DB} dB")
    
    model = LDeepSC(
        input_dim=input_dim,
        latent_dim=LATENT_DIM,
        chan_dim=CHAN_DIM,
        num_classes=num_classes,
        snr_db=SNR_DB
    ).to(DEVICE)
    
    # Loss functions
    recon_criterion = nn.MSELoss()
    class_criterion = nn.CrossEntropyLoss()
    
    # Weights cho multi-task loss
    lambda_recon = 1.0
    lambda_class = 0.5
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )
    
    best_test_loss = float('inf')
    history = {
        "train_loss": [], "test_loss": [],
        "train_recon_loss": [], "test_recon_loss": [],
        "train_class_loss": [], "test_class_loss": [],
        "train_acc": [], "test_acc": []
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
            x_hat, class_logits, class_logits_after = model(xb)
            
            # Reconstruction loss
            loss_recon = recon_criterion(x_hat, xb_target)
            
            # Classification loss (trước channel)
            loss_class = class_criterion(class_logits, yb)
            
            # Total loss
            loss = lambda_recon * loss_recon + lambda_class * loss_class
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * xb.size(0)
            train_recon_loss += loss_recon.item() * xb.size(0)
            train_class_loss += loss_class.item() * xb.size(0)
            
            # Accuracy
            _, predicted = torch.max(class_logits, 1)
            train_total += yb.size(0)
            train_correct += (predicted == yb).sum().item()
        
        train_loss /= len(train_loader.dataset)
        train_recon_loss /= len(train_loader.dataset)
        train_class_loss /= len(train_loader.dataset)
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
                
                x_hat, class_logits, class_logits_after = model(xb)
                
                loss_recon = recon_criterion(x_hat, xb_target)
                loss_class = class_criterion(class_logits, yb)
                loss = lambda_recon * loss_recon + lambda_class * loss_class
                
                test_loss += loss.item() * xb.size(0)
                test_recon_loss += loss_recon.item() * xb.size(0)
                test_class_loss += loss_class.item() * xb.size(0)
                
                _, predicted = torch.max(class_logits, 1)
                test_total += yb.size(0)
                test_correct += (predicted == yb).sum().item()
        
        test_loss /= len(test_loader.dataset)
        test_recon_loss /= len(test_loader.dataset)
        test_class_loss /= len(test_loader.dataset)
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
        
        scheduler.step(test_loss)
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch
            best_model_state = model.state_dict().copy()
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Loss: {train_loss:.4f}/{test_loss:.4f} | "
                  f"Recon: {train_recon_loss:.4f}/{test_recon_loss:.4f} | "
                  f"Acc: {train_acc*100:.1f}%/{test_acc*100:.1f}%")
    
    print(f"\nBest Test Loss: {best_test_loss:.6f} at epoch {best_epoch}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return model, history


# ============================================================
# EVALUATION - ĐẦY ĐỦ METRICS NHƯ FSE
# ============================================================

def evaluate_model(model, test_loader, scaler, le, X_test_raw, y_test):
    """Evaluate L-DeepSC với ĐẦY ĐỦ metrics để so sánh với FuzSemCom"""
    print(f"\n{'='*70}")
    print("Evaluating L-DeepSC Model - Full Metrics")
    print(f"{'='*70}")
    
    model.eval()
    
    all_x_hat = []
    all_class_pred = []
    all_class_pred_after = []
    all_y_true = []
    
    # Đo latency
    inference_times = []
    
    with torch.no_grad():
        for xb, _, yb in test_loader:
            xb = xb.to(DEVICE)
            
            start_time = time.time()
            x_hat, class_logits, class_logits_after = model(xb)
            inference_times.append(time.time() - start_time)
            
            all_x_hat.append(x_hat.cpu().numpy())
            all_class_pred.append(class_logits.argmax(dim=1).cpu().numpy())
            all_class_pred_after.append(class_logits_after.argmax(dim=1).cpu().numpy())
            all_y_true.append(yb.numpy())
    
    # Concatenate
    X_hat_scaled = np.vstack(all_x_hat)
    y_pred = np.concatenate(all_class_pred)
    y_pred_after = np.concatenate(all_class_pred_after)
    y_true = np.concatenate(all_y_true)
    
    # Inverse transform reconstruction
    X_hat_raw = scaler.inverse_transform(X_hat_scaled)
    
    # ============================================================
    # 1. RECONSTRUCTION METRICS
    # ============================================================
    print("\n" + "="*70)
    print("📊 1. RECONSTRUCTION METRICS (Sensor Values)")
    print("="*70)
    
    recon_results = {"per_variable": {}, "overall": {}}
    rmse_list, mae_list, r2_list = [], [], []
    
    print(f"\n{'Sensor':<15} {'RMSE':>10} {'MAE':>10} {'R²':>10}")
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
    # 2. CLASSIFICATION METRICS (như FSE)
    # ============================================================
    print("\n" + "="*70)
    print("📊 2. CLASSIFICATION METRICS (Semantic Class)")
    print("="*70)
    
    # Metrics trước channel (classification từ encoder output)
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Metrics sau channel
    accuracy_after = accuracy_score(y_true, y_pred_after)
    f1_macro_after = f1_score(y_true, y_pred_after, average='macro', zero_division=0)
    
    class_results = {
        "before_channel": {
            "accuracy": float(accuracy),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro)
        },
        "after_channel": {
            "accuracy": float(accuracy_after),
            "f1_macro": float(f1_macro_after)
        }
    }
    
    print(f"\n{'Metric':<25} {'Before Channel':>15} {'After Channel':>15}")
    print("-" * 60)
    print(f"{'Accuracy':<25} {accuracy*100:>14.2f}% {accuracy_after*100:>14.2f}%")
    print(f"{'F1 (macro)':<25} {f1_macro:>15.4f} {f1_macro_after:>15.4f}")
    print(f"{'F1 (weighted)':<25} {f1_weighted:>15.4f} {'-':>15}")
    print(f"{'Precision (macro)':<25} {precision_macro:>15.4f} {'-':>15}")
    print(f"{'Recall (macro)':<25} {recall_macro:>15.4f} {'-':>15}")
    
    # Classification report chi tiết
    print("\nClassification Report (Before Channel):")
    print(classification_report(y_true, y_pred, target_names=SEMANTIC_CLASSES, zero_division=0))
    
    # ============================================================
    # 3. BANDWIDTH & EFFICIENCY
    # ============================================================
    print("\n" + "="*70)
    print("📊 3. BANDWIDTH & EFFICIENCY")
    print("="*70)
    
    bandwidth_bytes = model.get_bandwidth_bytes()
    original_bytes = len(SENSOR_COLS) * 4  # 5 floats * 4 bytes
    compression_ratio = original_bytes / bandwidth_bytes
    
    avg_inference_time = np.mean(inference_times) * 1000  # ms
    
    efficiency_results = {
        "bandwidth_bytes_per_sample": bandwidth_bytes,
        "original_bytes_per_sample": original_bytes,
        "compression_ratio": float(compression_ratio),
        "avg_inference_time_ms": float(avg_inference_time)
    }
    
    print(f"Original data:     {original_bytes} bytes/sample (5 × float32)")
    print(f"Transmitted:       {bandwidth_bytes} bytes/sample ({CHAN_DIM} × float32)")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"Avg inference:     {avg_inference_time:.2f} ms/batch")
    
    # ============================================================
    # TỔNG HỢP KẾT QUẢ
    # ============================================================
    results = {
        "model": "L-DeepSC",
        "config": {
            "snr_db": SNR_DB,
            "latent_dim": LATENT_DIM,
            "chan_dim": CHAN_DIM,
            "num_epochs": NUM_EPOCHS
        },
        "reconstruction": recon_results,
        "classification": class_results,
        "efficiency": efficiency_results,
        "timestamp": datetime.now().isoformat()
    }
    
    return results, X_hat_raw, y_pred, confusion_matrix(y_true, y_pred)


def compare_with_fuzsemcom(l_deepsc_results):
    """So sánh chi tiết L-DeepSC vs FuzSemCom"""
    print(f"\n{'='*70}")
    print("📊 COMPARISON: L-DeepSC vs FuzSemCom")
    print(f"{'='*70}")
    
    # FuzSemCom results (từ notebook outputs)
    fuzsemcom = {
        "accuracy": 0.9499,
        "f1_macro": 0.9148,
        "f1_weighted": 0.9505,
        "precision_macro": 0.9246,
        "recall_macro": 0.9152,
        "bandwidth_bytes": 2,  # 2-byte symbol
        "reconstruction_rmse": 10.0,  # Approximate
    }
    
    l_deepsc = l_deepsc_results
    
    print(f"\n{'Metric':<30} {'L-DeepSC':>15} {'FuzSemCom':>15} {'Winner':>12}")
    print("=" * 75)
    
    # Classification metrics
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
    
    # Reconstruction metrics
    print("\n--- Reconstruction ---")
    l_rmse = l_deepsc["reconstruction"]["overall"]["rmse"]
    f_rmse = fuzsemcom["reconstruction_rmse"]
    winner = "L-DeepSC" if l_rmse < f_rmse else "FuzSemCom"
    print(f"{'Overall RMSE':<30} {l_rmse:>15.4f} {f_rmse:>15.4f} {winner:>12}")
    
    # Efficiency
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
        "l_deepsc": l_deepsc,
        "fuzsemcom": fuzsemcom
    }


def plot_results(history, cm, output_dir):
    """Vẽ các biểu đồ kết quả"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Training history - Loss
    ax = axes[0, 0]
    ax.plot(history["train_loss"], label="Train Loss")
    ax.plot(history["test_loss"], label="Test Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training History - Total Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Training history - Accuracy
    ax = axes[0, 1]
    ax.plot([x*100 for x in history["train_acc"]], label="Train Acc")
    ax.plot([x*100 for x in history["test_acc"]], label="Test Acc")
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
    ax.set_title("L-DeepSC Confusion Matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Reconstruction Loss
    ax = axes[1, 1]
    ax.plot(history["train_recon_loss"], label="Train Recon")
    ax.plot(history["test_recon_loss"], label="Test Recon")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Training History - Reconstruction Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "l_deepsc_training_results.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'l_deepsc_training_results.png'}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("L-DeepSC Baseline for FuzSemCom Comparison")
    print("Full Metrics: Reconstruction + Classification + Efficiency")
    print("=" * 70)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check data file
    if not DATA_PATH.exists():
        print(f"ERROR: Data file not found: {DATA_PATH}")
        print("Please run the notebook first to generate preprocessed data.")
        return
    
    # Load data
    train_loader, test_loader, scaler, le, X_train, X_test, y_train, y_test = load_data(DATA_PATH)
    input_dim = len(SENSOR_COLS)
    num_classes = len(SEMANTIC_CLASSES)
    
    # Train model
    model, history = train_model(train_loader, test_loader, input_dim, num_classes)
    
    # Evaluate
    results, X_hat, y_pred, cm = evaluate_model(
        model, test_loader, scaler, le, X_test, y_test
    )
    
    # Compare with FuzSemCom
    comparison = compare_with_fuzsemcom(results)
    
    # Plot results
    plot_results(history, cm, OUTPUT_DIR)
    
    # Save results
    results_path = OUTPUT_DIR / "l_deepsc_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    # Save comparison
    comparison_path = OUTPUT_DIR / "comparison_with_fuzsemcom.json"
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"Comparison saved to: {comparison_path}")
    
    # Save model
    model_path = OUTPUT_DIR / "l_deepsc_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")
    
    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
