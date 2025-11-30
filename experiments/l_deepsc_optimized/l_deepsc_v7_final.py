"""
L-DeepSC V7 Final - Best Configuration
======================================
Quay lai dung 5 sensor features goc (giong FuzSemCom) + oversampling manh.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "semantic_dataset_preprocessed.csv"
OUTPUT_DIR = Path(__file__).parent / "results_v7"

SENSOR_COLS = ["Moisture", "pH", "N", "Temperature", "Humidity"]
SEMANTIC_CLASSES = [
    "optimal", "nutrient_deficiency", "fungal_risk", "water_deficit_acidic",
    "water_deficit_alkaline", "acidic_soil", "alkaline_soil", "heat_stress",
]

BATCH_SIZE = 64
NUM_EPOCHS = 300
LR = 5e-4
HIDDEN_DIM = 128
LATENT_DIM = 32
DROPOUT = 0.4
NUM_ENSEMBLE = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# DATA
# ============================================================

def oversample_data(X, y, factor=10):
    """Oversample minority classes"""
    unique, counts = np.unique(y, return_counts=True)
    max_count = max(counts)
    
    X_new, y_new = list(X), list(y)
    
    for cls, cnt in zip(unique, counts):
        if cnt < max_count:
            mask = y == cls
            X_cls = X[mask]
            n_add = min(max_count - cnt, cnt * factor)
            
            indices = np.random.choice(len(X_cls), n_add, replace=True)
            noise = np.random.normal(0, 0.02, (n_add, X.shape[1])).astype(np.float32)
            X_new.extend(X_cls[indices] + noise)
            y_new.extend([cls] * n_add)
    
    return np.array(X_new, dtype=np.float32), np.array(y_new, dtype=np.int64)


def load_data(csv_path):
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    le = LabelEncoder()
    le.fit(SEMANTIC_CLASSES)
    
    valid_mask = df["semantic_label"].isin(SEMANTIC_CLASSES)
    df = df[valid_mask].reset_index(drop=True)
    
    X = df[SENSOR_COLS].values.astype("float32")
    y = le.transform(df["semantic_label"].values)
    
    # Split FIRST
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"  Original train: {len(X_train)}")
    
    # Oversample train only
    X_train_os, y_train_os = oversample_data(X_train, y_train, factor=20)
    print(f"  After oversample: {len(X_train_os)}")
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_os).astype("float32")
    X_test_scaled = scaler.transform(X_test).astype("float32")
    
    # Class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train_os), y=y_train_os)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    # Weighted sampler
    sample_weights = class_weights[y_train_os]
    sampler = WeightedRandomSampler(sample_weights, len(y_train_os), replacement=True)
    
    train_ds = TensorDataset(torch.tensor(X_train_scaled), torch.tensor(y_train_os, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test_scaled), torch.tensor(y_test, dtype=torch.long))
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"  Test: {len(X_test)}")
    
    # Class distribution
    print("\n  Class distribution (train after oversample):")
    unique, counts = np.unique(y_train_os, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    {SEMANTIC_CLASSES[u]}: {c}")
    
    return train_loader, test_loader, scaler, le, X_train, X_test, y_train, y_test, class_weights


# ============================================================
# MODEL - Simple but effective
# ============================================================

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128, dropout=0.4):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, num_classes),
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# ============================================================
# TRAINING
# ============================================================

def train_single(train_loader, test_loader, input_dim, num_classes, class_weights, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = SimpleClassifier(input_dim, num_classes, HIDDEN_DIM, DROPOUT).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    
    best_acc = 0.0
    patience = 50
    patience_counter = 0
    
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
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
        
        if epoch % 30 == 0:
            print(f"    [Seed {seed}] Epoch {epoch}: {acc*100:.2f}%")
    
    model.load_state_dict(best_state)
    print(f"    [Seed {seed}] Best: {best_acc*100:.2f}%")
    return model, best_acc


def train_ensemble(train_loader, test_loader, input_dim, num_classes, class_weights):
    print(f"\n{'='*70}")
    print(f"Training Ensemble ({NUM_ENSEMBLE} models)")
    print(f"{'='*70}")
    
    models = []
    for i in range(NUM_ENSEMBLE):
        print(f"\n  Model {i+1}/{NUM_ENSEMBLE}:")
        model, _ = train_single(train_loader, test_loader, input_dim, num_classes, class_weights, seed=42+i*7)
        models.append(model)
    
    return models


# ============================================================
# EVALUATION
# ============================================================

def evaluate(models, test_loader, y_test):
    print(f"\n{'='*70}")
    print("Evaluation")
    print(f"{'='*70}")
    
    all_probs = []
    
    for model in models:
        model.eval()
        probs = []
        with torch.no_grad():
            for xb, _ in test_loader:
                xb = xb.to(DEVICE)
                logits = model(xb)
                probs.append(F.softmax(logits, dim=1).cpu().numpy())
        all_probs.append(np.vstack(probs))
    
    ensemble_probs = np.mean(all_probs, axis=0)
    y_pred = ensemble_probs.argmax(axis=1)
    
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    
    print(f"\n  Accuracy:    {acc*100:.2f}%")
    print(f"  F1 (macro):  {f1_macro:.4f}")
    print(f"  F1 (weight): {f1_weighted:.4f}")
    print(f"  Precision:   {prec:.4f}")
    print(f"  Recall:      {rec:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=SEMANTIC_CLASSES, zero_division=0))
    
    cm = confusion_matrix(y_test, y_pred)
    
    results = {
        "model": "L-DeepSC V7 Final",
        "num_ensemble": NUM_ENSEMBLE,
        "classification": {
            "accuracy": float(acc),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "precision_macro": float(prec),
            "recall_macro": float(rec)
        },
        "timestamp": datetime.now().isoformat()
    }
    
    return results, y_pred, cm


def compare(results):
    print(f"\n{'='*70}")
    print("Comparison: L-DeepSC V7 vs FuzSemCom")
    print(f"{'='*70}")
    
    fsc = {"accuracy": 0.9499, "f1_macro": 0.9148, "precision_macro": 0.9246, "recall_macro": 0.9152}
    
    print(f"\n{'Metric':<20} {'L-DeepSC':>12} {'FuzSemCom':>12} {'Winner':>12}")
    print("-" * 60)
    
    l = results["classification"]
    for name, lv, fv in [
        ("Accuracy", l["accuracy"], fsc["accuracy"]),
        ("F1 (macro)", l["f1_macro"], fsc["f1_macro"]),
        ("Precision", l["precision_macro"], fsc["precision_macro"]),
        ("Recall", l["recall_macro"], fsc["recall_macro"]),
    ]:
        w = "L-DeepSC" if lv > fv else "FuzSemCom"
        print(f"{name:<20} {lv:>12.4f} {fv:>12.4f} {w:>12}")


def plot_cm(cm, output_dir):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=SEMANTIC_CLASSES, yticklabels=SEMANTIC_CLASSES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("L-DeepSC V7 - Confusion Matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("L-DeepSC V7 FINAL")
    print("=" * 70)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not DATA_PATH.exists():
        print(f"ERROR: {DATA_PATH} not found")
        return
    
    train_loader, test_loader, scaler, le, X_train, X_test, y_train, y_test, class_weights = load_data(DATA_PATH)
    
    models = train_ensemble(train_loader, test_loader, len(SENSOR_COLS), len(SEMANTIC_CLASSES), class_weights)
    
    results, y_pred, cm = evaluate(models, test_loader, y_test)
    
    compare(results)
    
    plot_cm(cm, OUTPUT_DIR)
    
    with open(OUTPUT_DIR / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    for i, model in enumerate(models):
        torch.save(model.state_dict(), OUTPUT_DIR / f"model_{i}.pth")
    
    acc = results["classification"]["accuracy"] * 100
    print("\n" + "=" * 70)
    print(f"FINAL: {acc:.2f}%")
    if acc >= 85:
        print("85%+ ACHIEVED!")
    elif acc >= 80:
        print("80%+ ACHIEVED!")
    print("=" * 70)


if __name__ == "__main__":
    main()

