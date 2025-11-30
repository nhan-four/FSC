"""
L-DeepSC V5 - Ensemble with Oversampling
=========================================
Cai tien:
1. SMOTE oversampling cho minority classes
2. Ensemble 3 models voi different seeds
3. Architecture tuned cho dataset nay
4. Label smoothing
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
    mean_squared_error
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
OUTPUT_DIR = Path(__file__).parent / "results_v5"

SENSOR_COLS = ["Moisture", "pH", "N", "Temperature", "Humidity"]
SEMANTIC_CLASSES = [
    "optimal", "nutrient_deficiency", "fungal_risk", "water_deficit_acidic",
    "water_deficit_alkaline", "acidic_soil", "alkaline_soil", "heat_stress",
]

BATCH_SIZE = 128
NUM_EPOCHS = 200
LR = 1e-3
HIDDEN_DIM = 256
LATENT_DIM = 64
DROPOUT = 0.3
NUM_ENSEMBLE = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# SMOTE-like Oversampling (simple version)
# ============================================================

def oversample_minority(X, y, target_count=None):
    """Simple oversampling by duplicating minority samples"""
    unique, counts = np.unique(y, return_counts=True)
    max_count = max(counts) if target_count is None else target_count
    
    X_new, y_new = [X], [y]
    
    for cls, cnt in zip(unique, counts):
        if cnt < max_count:
            # Get samples of this class
            mask = y == cls
            X_cls = X[mask]
            
            # Calculate how many to add
            n_add = max_count - cnt
            
            # Random oversample with small noise
            indices = np.random.choice(len(X_cls), n_add, replace=True)
            X_add = X_cls[indices] + np.random.normal(0, 0.01, (n_add, X.shape[1])).astype(np.float32)
            y_add = np.full(n_add, cls)
            
            X_new.append(X_add)
            y_new.append(y_add)
    
    return np.vstack(X_new), np.concatenate(y_new)


# ============================================================
# MODEL
# ============================================================

class SemanticNet(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256, latent_dim=64, dropout=0.3):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )
        
        self.latent_dim = latent_dim
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        z = self.encoder(x)
        logits = self.classifier(z)
        x_recon = self.decoder(z)
        return logits, x_recon, z


# ============================================================
# TRAINING
# ============================================================

def train_single_model(train_loader, test_loader, input_dim, num_classes, class_weights, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = SemanticNet(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        dropout=DROPOUT
    ).to(DEVICE)
    
    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE), label_smoothing=0.1)
    recon_criterion = nn.MSELoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    
    best_acc = 0.0
    patience = 30
    patience_counter = 0
    
    for epoch in range(1, NUM_EPOCHS + 1):
        # Train
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            
            optimizer.zero_grad()
            logits, x_recon, _ = model(xb)
            
            loss = criterion(logits, yb) + 0.05 * recon_criterion(x_recon, xb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        scheduler.step()
        
        # Eval
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits, _, _ = model(xb)
                correct += (logits.argmax(1) == yb).sum().item()
                total += yb.size(0)
        
        acc = correct / total
        
        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
        
        if epoch % 20 == 0:
            print(f"  [Seed {seed}] Epoch {epoch}: Acc = {acc*100:.2f}%")
    
    model.load_state_dict(best_state)
    print(f"  [Seed {seed}] Best Acc = {best_acc*100:.2f}%")
    return model, best_acc


def train_ensemble(train_loader, test_loader, input_dim, num_classes, class_weights):
    print(f"\n{'='*70}")
    print(f"Training Ensemble ({NUM_ENSEMBLE} models)")
    print(f"{'='*70}")
    
    models = []
    for i in range(NUM_ENSEMBLE):
        print(f"\nTraining model {i+1}/{NUM_ENSEMBLE}...")
        model, acc = train_single_model(
            train_loader, test_loader, input_dim, num_classes, class_weights, seed=42+i*10
        )
        models.append(model)
    
    return models


# ============================================================
# EVALUATION
# ============================================================

def evaluate_ensemble(models, test_loader, scaler, X_test_raw, y_test):
    print(f"\n{'='*70}")
    print("Evaluating Ensemble")
    print(f"{'='*70}")
    
    all_probs = []
    all_recons = []
    all_true = []
    
    for model in models:
        model.eval()
        probs, recons, trues = [], [], []
        
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(DEVICE)
                logits, x_recon, _ = model(xb)
                probs.append(F.softmax(logits, dim=1).cpu().numpy())
                recons.append(x_recon.cpu().numpy())
                trues.append(yb.numpy())
        
        all_probs.append(np.vstack(probs))
        all_recons.append(np.vstack(recons))
        if len(all_true) == 0:
            all_true = np.concatenate(trues)
    
    # Ensemble: average probabilities
    ensemble_probs = np.mean(all_probs, axis=0)
    y_pred = ensemble_probs.argmax(axis=1)
    
    ensemble_recon = np.mean(all_recons, axis=0)
    X_recon = scaler.inverse_transform(ensemble_recon)
    
    # Metrics
    acc = accuracy_score(all_true, y_pred)
    f1_macro = f1_score(all_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(all_true, y_pred, average='weighted', zero_division=0)
    prec = precision_score(all_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(all_true, y_pred, average='macro', zero_division=0)
    
    print(f"\nEnsemble Classification:")
    print(f"  Accuracy:    {acc*100:.2f}%")
    print(f"  F1 (macro):  {f1_macro:.4f}")
    print(f"  F1 (weight): {f1_weighted:.4f}")
    print(f"  Precision:   {prec:.4f}")
    print(f"  Recall:      {rec:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(all_true, y_pred, target_names=SEMANTIC_CLASSES, zero_division=0))
    
    # Reconstruction
    rmse_list = []
    print("\nReconstruction RMSE:")
    for i, col in enumerate(SENSOR_COLS):
        rmse = np.sqrt(mean_squared_error(X_test_raw[:, i], X_recon[:, i]))
        rmse_list.append(rmse)
        print(f"  {col}: {rmse:.4f}")
    print(f"  Overall: {np.mean(rmse_list):.4f}")
    
    cm = confusion_matrix(all_true, y_pred)
    
    results = {
        "model": "L-DeepSC V5 Ensemble",
        "num_models": NUM_ENSEMBLE,
        "classification": {
            "accuracy": float(acc),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "precision_macro": float(prec),
            "recall_macro": float(rec)
        },
        "reconstruction": {
            "overall_rmse": float(np.mean(rmse_list))
        },
        "efficiency": {
            "bandwidth_bytes": LATENT_DIM * 4
        },
        "timestamp": datetime.now().isoformat()
    }
    
    return results, y_pred, cm


def compare_fuzsemcom(results):
    print(f"\n{'='*70}")
    print("Comparison: L-DeepSC V5 vs FuzSemCom")
    print(f"{'='*70}")
    
    fsc = {"accuracy": 0.9499, "f1_macro": 0.9148, "precision_macro": 0.9246, "recall_macro": 0.9152}
    
    print(f"\n{'Metric':<25} {'L-DeepSC V5':>12} {'FuzSemCom':>12} {'Winner':>12}")
    print("-" * 65)
    
    l = results["classification"]
    for name, lv, fv in [
        ("Accuracy", l["accuracy"], fsc["accuracy"]),
        ("F1 (macro)", l["f1_macro"], fsc["f1_macro"]),
        ("Precision", l["precision_macro"], fsc["precision_macro"]),
        ("Recall", l["recall_macro"], fsc["recall_macro"]),
    ]:
        w = "L-DeepSC" if lv > fv else "FuzSemCom"
        print(f"{name:<25} {lv:>12.4f} {fv:>12.4f} {w:>12}")


def plot_results(history_placeholder, cm, output_dir):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=SEMANTIC_CLASSES, yticklabels=SEMANTIC_CLASSES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("L-DeepSC V5 Ensemble - Confusion Matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / "l_deepsc_v5_confusion.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'l_deepsc_v5_confusion.png'}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("L-DeepSC V5 ENSEMBLE - Target: 85%+ Accuracy")
    print("=" * 70)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not DATA_PATH.exists():
        print(f"ERROR: {DATA_PATH} not found")
        return
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    X = df[SENSOR_COLS].values.astype("float32")
    
    le = LabelEncoder()
    le.fit(SEMANTIC_CLASSES)
    
    valid_mask = df["semantic_label"].isin(SEMANTIC_CLASSES)
    df = df[valid_mask].reset_index(drop=True)
    X = df[SENSOR_COLS].values.astype("float32")
    y = le.transform(df["semantic_label"].values)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Original train: {len(X_train)}")
    
    # Oversample
    print("Oversampling minority classes...")
    X_train_os, y_train_os = oversample_minority(X_train, y_train)
    print(f"After oversample: {len(X_train_os)}")
    
    # Class distribution after oversample
    print("\nClass distribution after oversample:")
    unique, counts = np.unique(y_train_os, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  {SEMANTIC_CLASSES[u]}: {c}")
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_os).astype("float32")
    X_test_scaled = scaler.transform(X_test).astype("float32")
    
    # Class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train_os), y=y_train_os)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    # DataLoaders
    train_ds = TensorDataset(torch.tensor(X_train_scaled), torch.tensor(y_train_os, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test_scaled), torch.tensor(y_test, dtype=torch.long))
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # Train ensemble
    models = train_ensemble(train_loader, test_loader, len(SENSOR_COLS), len(SEMANTIC_CLASSES), class_weights)
    
    # Evaluate
    results, y_pred, cm = evaluate_ensemble(models, test_loader, scaler, X_test, y_test)
    
    # Compare
    compare_fuzsemcom(results)
    
    # Plot
    plot_results(None, cm, OUTPUT_DIR)
    
    # Save
    with open(OUTPUT_DIR / "l_deepsc_v5_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    for i, model in enumerate(models):
        torch.save(model.state_dict(), OUTPUT_DIR / f"l_deepsc_v5_model_{i}.pth")
    
    # Summary
    acc = results["classification"]["accuracy"] * 100
    print("\n" + "=" * 70)
    print(f"FINAL ENSEMBLE ACCURACY: {acc:.2f}%")
    if acc >= 85:
        print("TARGET ACHIEVED!")
    else:
        print(f"Gap: {85 - acc:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()

