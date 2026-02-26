# ============================================================
# XGNN-Based Intrusion Detection System
# File: main.py
# Author: Shreyas Santosh Shinde
# MSc Computer Science - Kirti College, Mumbai
# Guide: Dr. Prabha Kadam
# ============================================================

import os
import sys
import time

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from preprocessing.data_loader import preprocess_pipeline
from preprocessing.graph_builder import build_graph_pipeline
from models.gcn_model import train_gcn_pipeline
from models.gat_model import train_gat_pipeline
from explainability.gnn_explainer import run_explainability_pipeline
from explainability.attention_viz import (
    extract_attention_weights,
    plot_attention_per_head,
    plot_top_attended_nodes,
    plot_attack_vs_normal_attention
)
from evaluation.metrics import (
    evaluate_gnn_model,
    evaluate_baseline_models,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_comparison_table
)
from evaluation.comparison import compare_models_detailed


# ============================================================
# Helper
# ============================================================

def print_banner(title):
    print("\n")
    print("=" * 60)
    print(f"   {title}")
    print("=" * 60)


def print_step(step, total, description):
    print(f"\n[{step}/{total}] {description}")
    print("-" * 60)


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():

    start_time = time.time()

    print("\n")
    print("=" * 60)
    print("   XGNN-Based Intrusion Detection System")
    print("   Author : Shreyas Santosh Shinde")
    print("   Guide  : Dr. Prabha Kadam")
    print("   College: Kirti College, Mumbai")
    print("=" * 60)

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------
    base_dir   = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(base_dir, "data",
                              "Train_data.csv")
    test_path  = os.path.join(base_dir, "data",
                              "Test_data.csv")
    os.makedirs(os.path.join(base_dir, "outputs"),
                exist_ok=True)

    # --------------------------------------------------------
    # STEP 1 — Data Preprocessing
    # --------------------------------------------------------
    print_step(1, 7, "Data Preprocessing")
    train_df, test_df, scaler, feature_cols = \
        preprocess_pipeline(train_path, test_path)
    print("✓ Data preprocessing complete")

    # --------------------------------------------------------
    # STEP 2 — Graph Construction
    # --------------------------------------------------------
    print_step(2, 7, "Graph Construction")
    data, G = build_graph_pipeline(train_df, feature_cols)
    print("✓ Graph construction complete")

    # --------------------------------------------------------
    # STEP 3 — Train GCN Model
    # --------------------------------------------------------
    print_step(3, 7, "Training GCN Model")
    gcn_model, gcn_train_losses, gcn_test_losses, \
        gcn_train_accs, gcn_test_accs = \
        train_gcn_pipeline(data, epochs=100)
    print("✓ GCN training complete")

    # --------------------------------------------------------
    # STEP 4 — Train GAT Model
    # --------------------------------------------------------
    print_step(4, 7, "Training GAT Model")
    gat_model, gat_train_losses, gat_test_losses, \
        gat_train_accs, gat_test_accs = \
        train_gat_pipeline(data, epochs=100)
    print("✓ GAT training complete")

    # --------------------------------------------------------
    # STEP 5 — Explainability
    # --------------------------------------------------------
    print_step(5, 7, "Explainability Analysis")

    # GNN Explainer (feature importance + subgraph)
    run_explainability_pipeline(gcn_model, gat_model, data, feature_cols)

    # Attention visualization
    print("\nRunning attention visualizations...")
    edge_index, att_mean, att_per_head = \
        extract_attention_weights(gat_model, data)
    plot_attention_per_head(att_per_head)
    plot_top_attended_nodes(
        edge_index, att_mean, data, top_n=50
    )
    plot_attack_vs_normal_attention(
        edge_index, att_mean, data
    )
    print("✓ Explainability analysis complete")

    # --------------------------------------------------------
    # STEP 6 — Evaluation & Metrics
    # --------------------------------------------------------
    print_step(6, 7, "Model Evaluation")

    gcn_results = evaluate_gnn_model(
        gcn_model, data, "GCN"
    )
    gat_results = evaluate_gnn_model(
        gat_model, data, "GAT"
    )
    baseline_results = evaluate_baseline_models(data)

    all_results = [gcn_results, gat_results] + \
                  baseline_results

    plot_confusion_matrix(all_results)
    plot_roc_curves(all_results)
    plot_comparison_table(all_results)
    print("✓ Evaluation complete")

    # --------------------------------------------------------
    # STEP 7 — Detailed Comparison
    # --------------------------------------------------------
    print_step(7, 7, "Detailed Model Comparison")
    compare_models_detailed(all_results)
    print("✓ Comparison complete")

    # --------------------------------------------------------
    # Final Summary
    # --------------------------------------------------------
    elapsed = time.time() - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print_banner("FINAL RESULTS SUMMARY")
    print(f"{'Model':<22} {'Accuracy':>10} "
          f"{'F1-Score':>10} {'ROC-AUC':>10}")
    print("-" * 54)
    for r in all_results:
        print(f"{r['model']:<22} "
              f"{r['accuracy']:>10.4f} "
              f"{r['f1']:>10.4f} "
              f"{r['roc_auc']:>10.4f}")

    print("=" * 60)
    print(f"\nTotal runtime: {minutes}m {seconds}s")
    print(f"All outputs saved to: outputs/")
    print("\n✓ XGNN IDS Project Complete! 🎉")
    print("=" * 60)


# ============================================================

if __name__ == "__main__":
    main()