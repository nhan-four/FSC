"""
Baseline Models for Semantic Classification
============================================
So sanh cac model AI pho bien voi FuzSemCom va L-DeepSC tren cung dataset.

Models:
1. Decision Tree
2. Random Forest
3. Gradient Boosting (XGBoost)
4. Support Vector Machine (SVM)
5. K-Nearest Neighbors (KNN)
6. Naive Bayes
7. Logistic Regression
8. Multi-Layer Perceptron (MLP)
9. LightGBM
10. AdaBoost

Dataset: Agriculture Sensor Data (5 sensors, 8 classes)
Split: 80% train, 20% test (stratified, random_state=42)
"""

import os
import json
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from sklearn.utils.class_weight import compute_class_weight

# Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    AdaBoostClassifier,
    ExtraTreesClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# Optional: XGBoost, LightGBM
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. Skipping XGBoost baseline.")

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("Warning: LightGBM not installed. Skipping LightGBM baseline.")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "semantic_dataset_preprocessed.csv"
OUTPUT_DIR = Path(__file__).parent / "baseline_results"

SENSOR_COLS = ["Moisture", "pH", "N", "Temperature", "Humidity"]
SEMANTIC_CLASSES = [
    "optimal", "nutrient_deficiency", "fungal_risk", "water_deficit_acidic",
    "water_deficit_alkaline", "acidic_soil", "alkaline_soil", "heat_stress",
]

TEST_SIZE = 0.2
RANDOM_STATE = 42


# ============================================================
# DATA LOADING
# ============================================================

def load_data(csv_path):
    """Load va preprocess data giong FuzSemCom"""
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
        raise ValueError("Column 'semantic_label' not found in dataset")
    
    # Split - GIONG HOAN TOAN voi FuzSemCom va L-DeepSC
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"  Total samples: {len(X)}")
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"  Features: {SENSOR_COLS}")
    print(f"  Classes: {len(SEMANTIC_CLASSES)}")
    
    # Class distribution
    print("\n  Class distribution (train):")
    unique, counts = np.unique(y_train, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"    {SEMANTIC_CLASSES[u]}: {c}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, le, scaler


# ============================================================
# BASELINE MODELS
# ============================================================

def get_baseline_models():
    """Tra ve dict cac baseline models voi hyperparameters tot"""
    
    models = {}
    
    # 1. Decision Tree
    models["Decision Tree"] = {
        "model": DecisionTreeClassifier(
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=RANDOM_STATE
        ),
        "description": "Decision Tree with balanced class weights"
    }
    
    # 2. Random Forest
    models["Random Forest"] = {
        "model": RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            n_jobs=-1,
            random_state=RANDOM_STATE
        ),
        "description": "Random Forest with 200 trees"
    }
    
    # 3. Extra Trees
    models["Extra Trees"] = {
        "model": ExtraTreesClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            class_weight='balanced',
            n_jobs=-1,
            random_state=RANDOM_STATE
        ),
        "description": "Extra Trees Classifier"
    }
    
    # 4. Gradient Boosting
    models["Gradient Boosting"] = {
        "model": GradientBoostingClassifier(
            n_estimators=150,
            max_depth=8,
            learning_rate=0.1,
            min_samples_split=10,
            random_state=RANDOM_STATE
        ),
        "description": "Gradient Boosting Classifier"
    }
    
    # 5. AdaBoost
    models["AdaBoost"] = {
        "model": AdaBoostClassifier(
            n_estimators=100,
            learning_rate=0.5,
            random_state=RANDOM_STATE
        ),
        "description": "AdaBoost Classifier"
    }
    
    # 6. XGBoost (if available)
    if HAS_XGBOOST:
        models["XGBoost"] = {
            "model": XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric='mlogloss',
                n_jobs=-1,
                random_state=RANDOM_STATE
            ),
            "description": "XGBoost Classifier"
        }
    
    # 7. LightGBM (if available)
    if HAS_LIGHTGBM:
        models["LightGBM"] = {
            "model": LGBMClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                num_leaves=31,
                class_weight='balanced',
                n_jobs=-1,
                random_state=RANDOM_STATE,
                verbose=-1
            ),
            "description": "LightGBM Classifier"
        }
    
    # 8. SVM (RBF kernel)
    models["SVM (RBF)"] = {
        "model": SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            class_weight='balanced',
            random_state=RANDOM_STATE
        ),
        "description": "Support Vector Machine with RBF kernel"
    }
    
    # 9. SVM (Linear)
    models["SVM (Linear)"] = {
        "model": SVC(
            kernel='linear',
            C=1,
            class_weight='balanced',
            random_state=RANDOM_STATE
        ),
        "description": "Support Vector Machine with Linear kernel"
    }
    
    # 10. K-Nearest Neighbors
    models["KNN"] = {
        "model": KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            n_jobs=-1
        ),
        "description": "K-Nearest Neighbors (k=5)"
    }
    
    # 11. Naive Bayes
    models["Naive Bayes"] = {
        "model": GaussianNB(),
        "description": "Gaussian Naive Bayes"
    }
    
    # 12. Logistic Regression
    models["Logistic Regression"] = {
        "model": LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            multi_class='multinomial',
            solver='lbfgs',
            random_state=RANDOM_STATE
        ),
        "description": "Logistic Regression (multinomial)"
    }
    
    # 13. MLP (Neural Network)
    models["MLP"] = {
        "model": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=1e-4,
            batch_size=64,
            learning_rate_init=1e-3,
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=RANDOM_STATE
        ),
        "description": "Multi-Layer Perceptron (128-64-32)"
    }
    
    return models


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Train va evaluate mot model"""
    print(f"\n  Training {model_name}...", end=" ")
    
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Predict
    start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - start_time
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    
    print(f"Acc: {acc*100:.2f}%")
    
    return {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "train_time_sec": float(train_time),
        "inference_time_sec": float(inference_time),
        "y_pred": y_pred
    }


def run_all_baselines(X_train, X_test, y_train, y_test):
    """Chay tat ca baseline models"""
    print("\n" + "=" * 70)
    print("Running Baseline Models")
    print("=" * 70)
    
    models = get_baseline_models()
    results = {}
    
    for name, config in models.items():
        result = evaluate_model(
            config["model"], X_train, X_test, y_train, y_test, name
        )
        result["description"] = config["description"]
        results[name] = result
    
    return results


# ============================================================
# COMPARISON
# ============================================================

def add_reference_methods(results):
    """Them FuzSemCom va L-DeepSC vao ket qua"""
    
    # FuzSemCom (from notebook results)
    results["FuzSemCom (Ours)"] = {
        "accuracy": 0.9499,
        "f1_macro": 0.9148,
        "f1_weighted": 0.9505,
        "precision_macro": 0.9246,
        "recall_macro": 0.9152,
        "description": "Fuzzy Semantic Communication (Proposed)",
        "bandwidth_bytes": 2,
        "requires_training": False
    }
    
    # L-DeepSC (from experiments)
    results["L-DeepSC"] = {
        "accuracy": 0.7763,
        "f1_macro": 0.6329,
        "f1_weighted": 0.7482,
        "precision_macro": 0.6342,
        "recall_macro": 0.7093,
        "description": "Lite Deep Semantic Communication",
        "bandwidth_bytes": 64,
        "requires_training": True
    }
    
    return results


def create_comparison_table(results):
    """Tao bang so sanh"""
    print("\n" + "=" * 70)
    print("BASELINE COMPARISON RESULTS")
    print("=" * 70)
    
    # Sort by accuracy
    sorted_results = sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True)
    
    print(f"\n{'Rank':<5} {'Model':<25} {'Accuracy':>10} {'F1 (macro)':>12} {'Precision':>12} {'Recall':>10}")
    print("-" * 80)
    
    for rank, (name, metrics) in enumerate(sorted_results, 1):
        acc = metrics["accuracy"]
        f1 = metrics["f1_macro"]
        prec = metrics["precision_macro"]
        rec = metrics["recall_macro"]
        
        # Highlight FuzSemCom
        marker = " *" if "FuzSemCom" in name else ""
        
        print(f"{rank:<5} {name:<25} {acc*100:>9.2f}% {f1:>12.4f} {prec:>12.4f} {rec:>10.4f}{marker}")
    
    print("-" * 80)
    print("* Proposed method")
    
    return sorted_results


def plot_comparison(results, output_dir):
    """Ve bieu do so sanh"""
    
    # Sort by accuracy
    sorted_results = sorted(results.items(), key=lambda x: x[1]["accuracy"], reverse=True)
    
    names = [r[0] for r in sorted_results]
    accuracies = [r[1]["accuracy"] * 100 for r in sorted_results]
    f1_scores = [r[1]["f1_macro"] for r in sorted_results]
    
    # Colors
    colors = ['#2ecc71' if 'FuzSemCom' in n else '#3498db' if 'L-DeepSC' in n else '#95a5a6' for n in names]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Accuracy bar chart
    ax = axes[0]
    bars = ax.barh(names, accuracies, color=colors)
    ax.set_xlabel('Accuracy (%)')
    ax.set_title('Classification Accuracy Comparison')
    ax.invert_yaxis()
    ax.axvline(x=accuracies[0], color='red', linestyle='--', alpha=0.5, label=f'Best: {accuracies[0]:.2f}%')
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax.text(acc + 0.5, bar.get_y() + bar.get_height()/2, f'{acc:.2f}%', 
                va='center', fontsize=9)
    
    ax.legend()
    ax.set_xlim(0, 105)
    
    # F1 Score bar chart
    ax = axes[1]
    bars = ax.barh(names, f1_scores, color=colors)
    ax.set_xlabel('F1 Score (macro)')
    ax.set_title('F1 Score Comparison')
    ax.invert_yaxis()
    
    for bar, f1 in zip(bars, f1_scores):
        ax.text(f1 + 0.01, bar.get_y() + bar.get_height()/2, f'{f1:.4f}', 
                va='center', fontsize=9)
    
    ax.set_xlim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(output_dir / "baseline_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {output_dir / 'baseline_comparison.png'}")
    
    # Radar chart for top 5 models
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    metrics = ['Accuracy', 'F1 (macro)', 'F1 (weighted)', 'Precision', 'Recall']
    num_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]
    
    top_models = sorted_results[:5]
    colors_radar = plt.cm.Set2(np.linspace(0, 1, len(top_models)))
    
    for i, (name, m) in enumerate(top_models):
        values = [m['accuracy'], m['f1_macro'], m['f1_weighted'], m['precision_macro'], m['recall_macro']]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=name, color=colors_radar[i])
        ax.fill(angles, values, alpha=0.1, color=colors_radar[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.set_title('Top 5 Models - Multi-Metric Comparison')
    
    plt.tight_layout()
    plt.savefig(output_dir / "baseline_radar.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'baseline_radar.png'}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("BASELINE MODELS FOR SEMANTIC CLASSIFICATION")
    print("Comparison with FuzSemCom and L-DeepSC")
    print("=" * 70)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not DATA_PATH.exists():
        print(f"ERROR: Data file not found: {DATA_PATH}")
        return
    
    # Load data
    X_train, X_test, y_train, y_test, le, scaler = load_data(DATA_PATH)
    
    # Run baselines
    results = run_all_baselines(X_train, X_test, y_train, y_test)
    
    # Add reference methods
    results = add_reference_methods(results)
    
    # Create comparison table
    sorted_results = create_comparison_table(results)
    
    # Plot
    plot_comparison(results, OUTPUT_DIR)
    
    # Save results
    # Remove y_pred from results for JSON serialization
    results_json = {}
    for name, metrics in results.items():
        results_json[name] = {k: v for k, v in metrics.items() if k != 'y_pred'}
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as JSON
    json_path = OUTPUT_DIR / f"baseline_comparison_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"\nResults saved to: {json_path}")
    
    # Save as CSV
    df_results = pd.DataFrame([
        {
            "Model": name,
            "Accuracy": metrics["accuracy"],
            "F1 (macro)": metrics["f1_macro"],
            "F1 (weighted)": metrics.get("f1_weighted", 0),
            "Precision": metrics["precision_macro"],
            "Recall": metrics["recall_macro"],
            "Description": metrics.get("description", "")
        }
        for name, metrics in sorted_results
    ])
    
    csv_path = OUTPUT_DIR / f"baseline_comparison_{timestamp}.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    best_baseline = sorted_results[0]
    fuzsemcom_rank = next(i for i, (n, _) in enumerate(sorted_results, 1) if 'FuzSemCom' in n)
    
    print(f"\nBest model: {best_baseline[0]} ({best_baseline[1]['accuracy']*100:.2f}%)")
    print(f"FuzSemCom rank: #{fuzsemcom_rank}")
    
    if fuzsemcom_rank == 1:
        print("\nFuzSemCom OUTPERFORMS all baseline models!")
    else:
        gap = (sorted_results[0][1]['accuracy'] - results['FuzSemCom (Ours)']['accuracy']) * 100
        print(f"\nGap to best: {gap:.2f}%")
    
    print("\n" + "=" * 70)
    print("DONE!")
    print("=" * 70)


if __name__ == "__main__":
    main()

