# ============================================================
# XGNN-Based Intrusion Detection System
# File: evaluation/metrics.py
# Author: Shreyas Santosh Shinde
# MSc Computer Science - Kirti College, Mumbai
# ============================================================

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    classification_report
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


# ============================================================
# Evaluate GNN Models
# ============================================================

def evaluate_gnn_model(model, data, model_name="GCN"):
    """
    Get full evaluation metrics for GNN model
    """
    print(f"\nEvaluating {model_name} Model...")
    print("-" * 40)

    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        probs = torch.exp(out)[:, 1]

    # Get test predictions
    y_true = data.y[data.test_mask].numpy()
    y_pred = pred[data.test_mask].numpy()
    y_prob = probs[data.test_mask].numpy()

    # Calculate metrics
    accuracy  = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred,
                                zero_division=0)
    recall    = recall_score(y_true, y_pred,
                             zero_division=0)
    f1        = f1_score(y_true, y_pred,
                         zero_division=0)
    roc_auc   = roc_auc_score(y_true, y_prob)

    # Print results
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    print(f"\nClassification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=['Normal', 'Attack']
    ))

    return {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'y_true': y_true,
        'y_pred': y_pred,
        'y_prob': y_prob
    }


# ============================================================
# Train and Evaluate Baseline Models
# ============================================================

def evaluate_baseline_models(data):
    """
    Train and evaluate traditional ML models
    for comparison with XGNN
    """
    print("\n" + "=" * 50)
    print("Training Baseline Models for Comparison...")
    print("=" * 50)

    # Get features and labels
    X = data.x.numpy()
    y = data.y.numpy()

    X_train = X[data.train_mask.numpy()]
    y_train = y[data.train_mask.numpy()]
    X_test  = X[data.test_mask.numpy()]
    y_test  = y[data.test_mask.numpy()]

    results = []

    # Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_prob = rf.predict_proba(X_test)[:, 1]

    rf_results = {
        'model': 'Random Forest',
        'accuracy':  accuracy_score(y_test, rf_pred),
        'precision': precision_score(y_test, rf_pred,
                                     zero_division=0),
        'recall':    recall_score(y_test, rf_pred,
                                  zero_division=0),
        'f1':        f1_score(y_test, rf_pred,
                              zero_division=0),
        'roc_auc':   roc_auc_score(y_test, rf_prob),
        'y_true': y_test,
        'y_pred': rf_pred,
        'y_prob': rf_prob
    }
    results.append(rf_results)
    print(f"Random Forest Accuracy: "
          f"{rf_results['accuracy']*100:.2f}%")

    # MLP Neural Network
    print("\nTraining MLP Neural Network...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        max_iter=100,
        random_state=42
    )
    mlp.fit(X_train, y_train)
    mlp_pred = mlp.predict(X_test)
    mlp_prob = mlp.predict_proba(X_test)[:, 1]

    mlp_results = {
        'model': 'MLP Neural Network',
        'accuracy':  accuracy_score(y_test, mlp_pred),
        'precision': precision_score(y_test, mlp_pred,
                                     zero_division=0),
        'recall':    recall_score(y_test, mlp_pred,
                                  zero_division=0),
        'f1':        f1_score(y_test, mlp_pred,
                              zero_division=0),
        'roc_auc':   roc_auc_score(y_test, mlp_prob),
        'y_true': y_test,
        'y_pred': mlp_pred,
        'y_prob': mlp_prob
    }
    results.append(mlp_results)
    print(f"MLP Accuracy: "
          f"{mlp_results['accuracy']*100:.2f}%")

    return results


# ============================================================
# Plot Confusion Matrix
# ============================================================

def plot_confusion_matrix(results_list):
    """
    Plot confusion matrices for all models
    """
    print("\nGenerating confusion matrices...")

    n_models = len(results_list)
    fig, axes = plt.subplots(
        1, n_models,
        figsize=(5 * n_models, 4)
    )

    if n_models == 1:
        axes = [axes]

    for ax, results in zip(axes, results_list):
        cm = confusion_matrix(
            results['y_true'],
            results['y_pred']
        )

        im = ax.imshow(cm, interpolation='nearest',
                      cmap=plt.cm.Blues)
        ax.set_title(f"{results['model']}\n"
                    f"Accuracy: "
                    f"{results['accuracy']*100:.2f}%",
                    fontweight='bold')

        # Add text annotations
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh
                       else "black",
                       fontsize=14, fontweight='bold')

        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Normal', 'Attack'])
        ax.set_yticklabels(['Normal', 'Attack'])

    plt.tight_layout()
    os.makedirs('../outputs', exist_ok=True)
    plt.savefig('../outputs/confusion_matrices.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("Confusion matrices saved to "
          "outputs/confusion_matrices.png")


# ============================================================
# Plot ROC Curves
# ============================================================

def plot_roc_curves(results_list):
    """
    Plot ROC curves for all models
    """
    print("\nGenerating ROC curves...")

    plt.figure(figsize=(10, 7))

    colors = ['blue', 'green', 'red',
              'orange', 'purple']

    for i, results in enumerate(results_list):
        fpr, tpr, _ = roc_curve(
            results['y_true'],
            results['y_prob']
        )
        plt.plot(
            fpr, tpr,
            color=colors[i % len(colors)],
            linewidth=2,
            label=f"{results['model']} "
                  f"(AUC = {results['roc_auc']:.4f})"
        )

    plt.plot([0, 1], [0, 1], 'k--',
             linewidth=1, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Model Comparison',
              fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig('../outputs/roc_curves.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("ROC curves saved to outputs/roc_curves.png")


# ============================================================
# Final Comparison Table
# ============================================================

def plot_comparison_table(results_list):
    """
    Plot bar chart comparing all models
    """
    print("\nGenerating model comparison chart...")

    models = [r['model'] for r in results_list]
    metrics = ['accuracy', 'precision',
               'recall', 'f1', 'roc_auc']
    metric_labels = ['Accuracy', 'Precision',
                     'Recall', 'F1-Score', 'ROC-AUC']

    x = np.arange(len(metrics))
    width = 0.8 / len(models)
    colors = ['steelblue', 'forestgreen',
              'tomato', 'orange', 'purple']

    fig, ax = plt.subplots(figsize=(14, 7))

    for i, (results, color) in enumerate(
        zip(results_list, colors)
    ):
        values = [results[m] for m in metrics]
        offset = (i - len(models)/2 + 0.5) * width
        bars = ax.bar(x + offset, values,
                      width, label=results['model'],
                      color=color, alpha=0.8,
                      edgecolor='black')

        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f'{val:.3f}',
                ha='center', va='bottom',
                fontsize=8, fontweight='bold'
            )

    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('XGNN vs Baseline Models — '
                 'Performance Comparison',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('../outputs/model_comparison.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("Comparison chart saved to "
          "outputs/model_comparison.png")


# ============================================================
# Run Full Evaluation Pipeline
# ============================================================

if __name__ == "__main__":

    from preprocessing.data_loader import preprocess_pipeline
    from preprocessing.graph_builder import build_graph_pipeline
    from models.gcn_model import train_gcn_pipeline
    from models.gat_model import train_gat_pipeline

    base_dir = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
    train_path = os.path.join(
        base_dir, "data", "Train_data.csv"
    )
    test_path = os.path.join(
        base_dir, "data", "Test_data.csv"
    )

    # Step 1: Preprocess
    train_df, test_df, scaler, feature_cols = \
        preprocess_pipeline(train_path, test_path)

    # Step 2: Build graph
    data, G = build_graph_pipeline(train_df, feature_cols)

    # Step 3: Train GNN models
    print("\nTraining GCN...")
    gcn_model, _, _, _, _ = train_gcn_pipeline(
        data, epochs=100
    )

    print("\nTraining GAT...")
    gat_model, _, _, _, _ = train_gat_pipeline(
        data, epochs=100
    )

    # Step 4: Evaluate GNN models
    print("\n" + "=" * 50)
    print("Evaluating All Models")
    print("=" * 50)

    gcn_results = evaluate_gnn_model(
        gcn_model, data, "GCN"
    )
    gat_results = evaluate_gnn_model(
        gat_model, data, "GAT"
    )

    # Step 5: Evaluate baseline models
    baseline_results = evaluate_baseline_models(data)

    # Step 6: Combine all results
    all_results = [gcn_results, gat_results] + \
                  baseline_results

    # Step 7: Plot confusion matrices
    plot_confusion_matrix(all_results)

    # Step 8: Plot ROC curves
    plot_roc_curves(all_results)

    # Step 9: Plot comparison chart
    plot_comparison_table(all_results)

    # Step 10: Print final summary
    print("\n" + "=" * 50)
    print("FINAL RESULTS SUMMARY")
    print("=" * 50)
    print(f"{'Model':<20} {'Accuracy':>10} "
          f"{'Precision':>10} {'Recall':>10} "
          f"{'F1':>10} {'ROC-AUC':>10}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['model']:<20} "
              f"{r['accuracy']:>10.4f} "
              f"{r['precision']:>10.4f} "
              f"{r['recall']:>10.4f} "
              f"{r['f1']:>10.4f} "
              f"{r['roc_auc']:>10.4f}")
    print("=" * 50)
    print("\nAll evaluation plots saved to outputs/ folder!")
    print("Project Complete! 🎉")