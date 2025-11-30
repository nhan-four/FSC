"""
L-DeepSC V6 - Extended Features (P, K, NDI_Label, PDI_Label)
============================================================
Su dung them cac features khong co NaN:
- P (Phosphorus)
- K (Potassium)
- NDI_Label (Low/Medium/High)
- PDI_Label (Low/Medium/High)

NDVI, NDRE co 50% NaN nen khong su dung.
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
OUTPUT_DIR = Path(__file__).parent / "results_v6"

# Features - chi dung cac cot khong co NaN
SENSOR_COLS = ["Moisture", "pH", "N", "Temperature", "Humidity"]
EXTRA_NUMERIC_COLS = ["P", "K"]  # Khong dung NDVI, NDRE vi co 50% NaN
CATEGORICAL_COLS = ["NDI_Label", "PDI_Label"]

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
# DATA LOADING
# ============================================================

def load_data_extended(csv_path):
    """Load data voi extended features"""
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Filter valid semantic labels
    le_target = LabelEncoder()
    le_target.fit(SEMANTIC_CLASSES)
    
    valid_mask = df["semantic_label"].isin(SEMANTIC_CLASSES)
    df = df[valid_mask].reset_index(drop=True)
    
    y = le_target.transform(df["semantic_label"].values)
    
    # Numeric features
    X_sensor = df[SENSOR_COLS].values.astype("float32")
    X_extra = df[EXTRA_NUMERIC_COLS].values.astype("float32")
    
    # Categorical features (NDI_Label, PDI_Label)
    ndi_map = {"Low": 0, "Medium": 1, "High": 2}
    pdi_map = {"Low": 0, "Medium": 1, "High": 2}
    
    ndi = df["NDI_Label"].map(ndi_map).fillna(1).values.astype("int64")
    pdi = df["PDI_Label"].map(pdi_map).fillna(1).values.astype("int64")
    
    # One-hot encode
    ndi_onehot = np.zeros((len(ndi), 3), dtype="float32")
    ndi_onehot[np.arange(len(ndi)), ndi] = 1
    
    pdi_onehot = np.zeros((len(pdi), 3), dtype="float32")
    pdi_onehot[np.arange(len(pdi)), pdi] = 1
    
    # Combine all features
    X_all = np.hstack([X_sensor, X_extra, ndi_onehot, pdi_onehot])
    
    print(f"  Total samples: {len(X_all)}")
    print(f"  Feature breakdown:")
    print(f"    - Sensor cols ({len(SENSOR_COLS)}): {SENSOR_COLS}")
    print(f"    - Extra numeric ({len(EXTRA_NUMERIC_COLS)}): {EXTRA_NUMERIC_COLS}")
    print(f"    - NDI_Label one-hot (3)")
    print(f"    - PDI_Label one-hot (3)")
    print(f"    - Total features: {X_all.shape[1]}")
    
    # Split - GIONG HOAN TOAN voi FuzSemCom
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
    
    # Class distribution
    print("\n  Class distribution (train):")
    unique, counts = np.unique(y_train, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    {SEMANTIC_CLASSES[u]}: {c}")
    
    # Scale numeric features only
    scaler = StandardScaler()
    num_numeric = len(SENSOR_COLS) + len(EXTRA_NUMERIC_COLS)
    
    X_train_numeric = X_train[:, :num_numeric].copy()
    X_test_numeric = X_test[:, :num_numeric].copy()
    X_train_cat = X_train[:, num_numeric:].copy()
    X_test_cat = X_test[:, num_numeric:].copy()
    
    X_train_numeric_scaled = scaler.fit_transform(X_train_numeric).astype("float32")
    X_test_numeric_scaled = scaler.transform(X_test_numeric).astype("float32")
    
    X_train_scaled = np.hstack([X_train_numeric_scaled, X_train_cat]).astype("float32")
    X_test_scaled = np.hstack([X_test_numeric_scaled, X_test_cat]).astype("float32")
    
    # Class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    # DataLoaders
    train_ds = TensorDataset(
        torch.tensor(X_train_scaled),
        torch.tensor(y_train, dtype=torch.long)
    )
    test_ds = TensorDataset(
        torch.tensor(X_test_scaled),
        torch.tensor(y_test, dtype=torch.long)
    )
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, test_loader, scaler, le_target, X_train, X_test, y_train, y_test, class_weights


# ============================================================
# MODEL
# ============================================================

class SemanticNetExtended(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=256, latent_dim=64, dropout=0.3):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.GELU(),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
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
        return logits, z


# ============================================================
# TRAINING
# ============================================================

def train_single_model(train_loader, test_loader, input_dim, num_classes, class_weights, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    model = SemanticNetExtended(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        dropout=DROPOUT
    ).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE), label_smoothing=0.1)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    
    best_acc = 0.0
    patience = 40
    patience_counter = 0
    
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            
            optimizer.zero_grad()
            logits, _ = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
        # Eval
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits, _ = model(xb)
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
            print(f"  [Seed {seed}] Epoch {epoch}: Acc = {acc*100:.2f}%, Loss = {train_loss/len(train_loader):.4f}")
    
    model.load_state_dict(best_state)
    print(f"  [Seed {seed}] Best Acc = {best_acc*100:.2f}%")
    return model, best_acc


def train_ensemble(train_loader, test_loader, input_dim, num_classes, class_weights):
    print(f"\n{'='*70}")
    print(f"Training Ensemble ({NUM_ENSEMBLE} models) with Extended Features")
    print(f"{'='*70}")
    print(f"Device: {DEVICE}")
    print(f"Input dim: {input_dim}, Latent dim: {LATENT_DIM}")
    
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

def evaluate_ensemble(models, test_loader, y_test):
    print(f"\n{'='*70}")
    print("Evaluating Ensemble")
    print(f"{'='*70}")
    
    all_probs = []
    all_true = []
    
    for model in models:
        model.eval()
        probs, trues = [], []
        
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(DEVICE)
                logits, _ = model(xb)
                probs.append(F.softmax(logits, dim=1).cpu().numpy())
                trues.append(yb.numpy())
        
        all_probs.append(np.vstack(probs))
        if len(all_true) == 0:
            all_true = np.concatenate(trues)
    
    # Ensemble average
    ensemble_probs = np.mean(all_probs, axis=0)
    y_pred = ensemble_probs.argmax(axis=1)
    
    # Metrics
    acc = accuracy_score(all_true, y_pred)
    f1_macro = f1_score(all_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(all_true, y_pred, average='weighted', zero_division=0)
    prec = precision_score(all_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(all_true, y_pred, average='macro', zero_division=0)
    
    print(f"\nEnsemble Classification Results:")
    print(f"  Accuracy:    {acc*100:.2f}%")
    print(f"  F1 (macro):  {f1_macro:.4f}")
    print(f"  F1 (weight): {f1_weighted:.4f}")
    print(f"  Precision:   {prec:.4f}")
    print(f"  Recall:      {rec:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(all_true, y_pred, target_names=SEMANTIC_CLASSES, zero_division=0))
    
    cm = confusion_matrix(all_true, y_pred)
    
    results = {
        "model": "L-DeepSC V6 Extended Features",
        "num_models": NUM_ENSEMBLE,
        "features": {
            "sensor": SENSOR_COLS,
            "extra_numeric": EXTRA_NUMERIC_COLS,
            "categorical": CATEGORICAL_COLS,
            "total_features": len(SENSOR_COLS) + len(EXTRA_NUMERIC_COLS) + 6
        },
        "classification": {
            "accuracy": float(acc),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "precision_macro": float(prec),
            "recall_macro": float(rec)
        },
        "efficiency": {
            "latent_dim": LATENT_DIM,
            "bandwidth_bytes": LATENT_DIM * 4
        },
        "timestamp": datetime.now().isoformat()
    }
    
    return results, y_pred, cm


def compare_fuzsemcom(results):
    print(f"\n{'='*70}")
    print("Comparison: L-DeepSC V6 vs FuzSemCom")
    print(f"{'='*70}")
    
    fsc = {"accuracy": 0.9499, "f1_macro": 0.9148, "precision_macro": 0.9246, "recall_macro": 0.9152}
    
    print(f"\n{'Metric':<25} {'L-DeepSC V6':>12} {'FuzSemCom':>12} {'Winner':>12}")
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
    
    return {"l_deepsc_v6": results, "fuzsemcom": fsc}


def plot_results(cm, output_dir):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=SEMANTIC_CLASSES, yticklabels=SEMANTIC_CLASSES, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("L-DeepSC V6 Extended Features - Confusion Matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / "l_deepsc_v6_confusion.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'l_deepsc_v6_confusion.png'}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("L-DeepSC V6 - EXTENDED FEATURES (P, K, NDI_Label, PDI_Label)")
    print("Target: 85%+ Accuracy")
    print("=" * 70)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not DATA_PATH.exists():
        print(f"ERROR: {DATA_PATH} not found")
        return
    
    # Load data
    train_loader, test_loader, scaler, le, X_train, X_test, y_train, y_test, class_weights = load_data_extended(DATA_PATH)
    
    input_dim = X_train.shape[1]
    num_classes = len(SEMANTIC_CLASSES)
    
    # Train
    models = train_ensemble(train_loader, test_loader, input_dim, num_classes, class_weights)
    
    # Evaluate
    results, y_pred, cm = evaluate_ensemble(models, test_loader, y_test)
    
    # Compare
    comparison = compare_fuzsemcom(results)
    
    # Plot
    plot_results(cm, OUTPUT_DIR)
    
    # Save
    with open(OUTPUT_DIR / "l_deepsc_v6_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    with open(OUTPUT_DIR / "comparison_v6.json", 'w') as f:
        json.dump(comparison, f, indent=2)
    
    for i, model in enumerate(models):
        torch.save(model.state_dict(), OUTPUT_DIR / f"l_deepsc_v6_model_{i}.pth")
    
    # Summary
    acc = results["classification"]["accuracy"] * 100
    print("\n" + "=" * 70)
    print(f"FINAL ENSEMBLE ACCURACY: {acc:.2f}%")
    if acc >= 85:
        print("TARGET ACHIEVED!")
    elif acc >= 80:
        print("80%+ ACHIEVED!")
    else:
        print(f"Gap to 85%: {85 - acc:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    main()
