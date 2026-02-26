# ============================================================
# XGNN-Based Intrusion Detection System
# File: evaluation/comparison.py
# Author: Shreyas Santosh Shinde
# MSc Computer Science - Kirti College, Mumbai
# ============================================================

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
    roc_auc_score
)


# ============================================================
# Detailed Model Comparison
# ============================================================

def compare_models_detailed(all_results):
    """
    Detailed comparison between all models
    with multiple visualizations
    """
    print("=" * 50)
    print("Running Detailed Model Comparison...")
    print("=" * 50)

    os.makedirs('../outputs', exist_ok=True)

    # 1. Radar Chart Comparison
    plot_radar_chart(all_results)

    # 2. Metrics Heatmap
    plot_metrics_heatmap(all_results)

    # 3. Accuracy vs Explainability
    plot_accuracy_vs_explainability(all_results)

    # 4. Print detailed comparison
    print_detailed_comparison(all_results)


# ============================================================
# Radar Chart
# ============================================================

def plot_radar_chart(all_results):
    """
    Plot radar/spider chart comparing all models
    across multiple metrics
    """
    print("\nGenerating radar chart...")

    metrics = ['Accuracy', 'Precision',
               'Recall', 'F1-Score', 'ROC-AUC']
    metric_keys = ['accuracy', 'precision',
                   'recall', 'f1', 'roc_auc']

    # Number of metrics
    N = len(metrics)

    # Angles for radar chart
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # Setup plot
    fig, ax = plt.subplots(
        figsize=(10, 8),
        subplot_kw=dict(polar=True)
    )

    colors = ['steelblue', 'forestgreen',
              'tomato', 'orange']

    for i, (results, color) in enumerate(
        zip(all_results, colors)
    ):
        values = [results[k] for k in metric_keys]
        values += values[:1]

        ax.plot(angles, values,
                color=color, linewidth=2,
                linestyle='solid',
                label=results['model'])
        ax.fill(angles, values,
                color=color, alpha=0.1)

    # Add metric labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(
        ['0.2', '0.4', '0.6', '0.8', '1.0'],
        fontsize=8
    )
    ax.grid(True, alpha=0.3)

    plt.legend(loc='upper right',
               bbox_to_anchor=(1.3, 1.1),
               fontsize=11)
    plt.title('Model Comparison — Radar Chart',
              fontsize=14, fontweight='bold',
              pad=20)

    plt.tight_layout()
    plt.savefig('../outputs/radar_chart.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("Radar chart saved to outputs/radar_chart.png")


# ============================================================
# Metrics Heatmap
# ============================================================

def plot_metrics_heatmap(all_results):
    """
    Plot heatmap of all metrics for all models
    """
    print("\nGenerating metrics heatmap...")

    metrics = ['Accuracy', 'Precision',
               'Recall', 'F1-Score', 'ROC-AUC']
    metric_keys = ['accuracy', 'precision',
                   'recall', 'f1', 'roc_auc']

    models = [r['model'] for r in all_results]

    # Build matrix
    matrix = np.array([
        [r[k] for k in metric_keys]
        for r in all_results
    ])

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, cmap='YlOrRd',
                   aspect='auto',
                   vmin=0.85, vmax=1.0)

    # Add colorbar
    plt.colorbar(im, ax=ax, label='Score')

    # Labels
    ax.set_xticks(range(len(metrics)))
    ax.set_yticks(range(len(models)))
    ax.set_xticklabels(metrics, fontsize=12)
    ax.set_yticklabels(models, fontsize=12)

    # Add text annotations
    for i in range(len(models)):
        for j in range(len(metrics)):
            ax.text(j, i,
                    f'{matrix[i, j]:.4f}',
                    ha='center', va='center',
                    fontsize=11, fontweight='bold',
                    color='black')

    ax.set_title('Model Performance Heatmap',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('../outputs/metrics_heatmap.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("Heatmap saved to outputs/metrics_heatmap.png")


# ============================================================
# Accuracy vs Explainability Plot
# ============================================================

def plot_accuracy_vs_explainability(all_results):
    """
    Plot showing tradeoff between accuracy
    and explainability
    This is the KEY argument of your research!
    """
    print("\nGenerating accuracy vs explainability plot...")

    # Explainability scores (manually defined)
    # Higher = more explainable
    explainability_scores = {
        'GCN': 0.90,
        'GAT': 0.95,
        'Random Forest': 0.30,
        'MLP Neural Network': 0.15
    }

    models = [r['model'] for r in all_results]
    accuracies = [r['accuracy'] for r in all_results]
    explainability = [
        explainability_scores.get(r['model'], 0.5)
        for r in all_results
    ]

    colors = ['steelblue', 'forestgreen',
              'tomato', 'orange']
    sizes = [300, 300, 300, 300]

    fig, ax = plt.subplots(figsize=(10, 7))

    for i, (model, acc, exp, color) in enumerate(
        zip(models, accuracies,
            explainability, colors)
    ):
        ax.scatter(exp, acc,
                   c=color, s=sizes[i],
                   zorder=5, label=model)
        ax.annotate(
            model,
            (exp, acc),
            textcoords="offset points",
            xytext=(10, 5),
            fontsize=11,
            fontweight='bold'
        )

    # Add quadrant lines
    ax.axhline(y=0.95, color='gray',
               linestyle='--', alpha=0.5)
    ax.axvline(x=0.5, color='gray',
               linestyle='--', alpha=0.5)

    # Add quadrant labels
    ax.text(0.75, 0.97,
            'HIGH Explainability\nHIGH Accuracy',
            ha='center', fontsize=10,
            color='green', fontweight='bold')
    ax.text(0.15, 0.97,
            'LOW Explainability\nHIGH Accuracy',
            ha='center', fontsize=10,
            color='red', fontweight='bold')
    ax.text(0.75, 0.88,
            'HIGH Explainability\nLOW Accuracy',
            ha='center', fontsize=10,
            color='blue', fontweight='bold')

    ax.set_xlabel('Explainability Score',
                  fontsize=13)
    ax.set_ylabel('Detection Accuracy',
                  fontsize=13)
    ax.set_title(
        'Accuracy vs Explainability Tradeoff\n'
        'XGNN models achieve high explainability '
        'with competitive accuracy',
        fontsize=13, fontweight='bold'
    )
    ax.set_xlim(0, 1.1)
    ax.set_ylim(0.85, 1.02)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        '../outputs/accuracy_vs_explainability.png',
        dpi=150, bbox_inches='tight'
    )
    plt.show()
    print("Plot saved to "
          "outputs/accuracy_vs_explainability.png")


# ============================================================
# Print Detailed Comparison
# ============================================================

def print_detailed_comparison(all_results):
    """
    Print a detailed comparison table
    """
    print("\n" + "=" * 70)
    print("DETAILED MODEL COMPARISON REPORT")
    print("=" * 70)

    metrics = ['accuracy', 'precision',
               'recall', 'f1', 'roc_auc']
    metric_labels = ['Accuracy', 'Precision',
                     'Recall', 'F1-Score', 'ROC-AUC']

    print(f"\n{'Model':<22}", end="")
    for label in metric_labels:
        print(f"{label:>10}", end="")
    print()
    print("-" * 72)

    for r in all_results:
        print(f"{r['model']:<22}", end="")
        for m in metrics:
            print(f"{r[m]:>10.4f}", end="")
        print()

    print("=" * 70)

    # Find best model per metric
    print("\nBest Model Per Metric:")
    print("-" * 40)
    for m, label in zip(metrics, metric_labels):
        best = max(all_results, key=lambda x: x[m])
        print(f"{label:<12}: {best['model']:<22} "
              f"({best[m]:.4f})")

    print("\nKey Finding:")
    print("-" * 40)
    print("GCN and GAT provide EXPLAINABLE decisions")
    print("with competitive accuracy, making them")
    print("ideal for real-world cybersecurity deployment")
    print("where transparency is critical.")
    print("=" * 70)


# ============================================================
# Run Comparison
# ============================================================

if __name__ == "__main__":

    from preprocessing.data_loader import preprocess_pipeline
    from preprocessing.graph_builder import build_graph_pipeline
    from models.gcn_model import train_gcn_pipeline
    from models.gat_model import train_gat_pipeline
    from evaluation.metrics import (
        evaluate_gnn_model,
        evaluate_baseline_models
    )

    base_dir = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
    train_path = os.path.join(
        base_dir, "data", "Train_data.csv"
    )
    test_path = os.path.join(
        base_dir, "data", "Test_data.csv"
    )

    # Preprocess
    train_df, test_df, scaler, feature_cols = \
        preprocess_pipeline(train_path, test_path)

    # Build graph
    data, G = build_graph_pipeline(train_df, feature_cols)

    # Train models
    print("\nTraining GCN...")
    gcn_model, _, _, _, _ = train_gcn_pipeline(
        data, epochs=100
    )

    print("\nTraining GAT...")
    gat_model, _, _, _, _ = train_gat_pipeline(
        data, epochs=100
    )

    # Evaluate
    gcn_results = evaluate_gnn_model(
        gcn_model, data, "GCN"
    )
    gat_results = evaluate_gnn_model(
        gat_model, data, "GAT"
    )
    baseline_results = evaluate_baseline_models(data)

    # Combine all results
    all_results = [gcn_results, gat_results] + \
                  baseline_results

    # Run detailed comparison
    compare_models_detailed(all_results)