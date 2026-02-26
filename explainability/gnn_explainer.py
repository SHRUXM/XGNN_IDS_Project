# ============================================================
# XGNN-Based Intrusion Detection System
# File: explainability/gnn_explainer.py
# Author: Shreyas Santosh Shinde
# MSc Computer Science - Kirti College, Mumbai
# ============================================================

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import networkx as nx
import os
import sys

sys.path.append(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))

from models.gcn_model import GCNModel, train_gcn_pipeline
from models.gat_model import GATModel, train_gat_pipeline
from preprocessing.data_loader import preprocess_pipeline
from preprocessing.graph_builder import build_graph_pipeline


# ============================================================
# Feature Importance Analysis
# ============================================================

def compute_feature_importance(model, data, feature_cols):
    """
    Compute feature importance using gradient based method
    Shows which network features matter most for detection
    """
    print("=" * 50)
    print("Computing Feature Importance...")
    print("=" * 50)

    model.eval()

    # Enable gradients for input
    x = data.x.clone().requires_grad_(True)

    # Forward pass
    out = model(x, data.edge_index)

    # Compute gradients for attack class (class 1)
    out[:, 1].sum().backward()

    # Get gradient magnitudes
    importance = x.grad.abs().mean(dim=0)
    importance = importance.detach().numpy()

    # Normalize
    importance = importance / importance.sum()

    return importance


def plot_feature_importance(importance, feature_cols,
                            model_name="GCN", top_n=15):
    """
    Plot top N most important features
    """
    print(f"\nPlotting top {top_n} important features...")

    # Get top N features
    top_indices = np.argsort(importance)[-top_n:][::-1]
    top_features = [feature_cols[i] for i in top_indices]
    top_importance = importance[top_indices]

    # Plot
    plt.figure(figsize=(12, 6))
    colors = plt.cm.RdYlGn(
        np.linspace(0.2, 0.8, top_n)
    )[::-1]

    bars = plt.barh(range(top_n),
                    top_importance,
                    color=colors)

    plt.yticks(range(top_n), top_features, fontsize=10)
    plt.xlabel('Feature Importance Score', fontsize=12)
    plt.title(f'{model_name} - Top {top_n} Most Important'
              f' Features for Intrusion Detection',
              fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()

    # Save
    os.makedirs('../outputs', exist_ok=True)
    filename = f'../outputs/{model_name.lower()}_feature_importance.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Feature importance plot saved to {filename}")

    # Print top features
    print(f"\nTop {top_n} Important Features:")
    print("-" * 40)
    for i, (feat, imp) in enumerate(
        zip(top_features, top_importance)
    ):
        print(f"{i+1:2d}. {feat:35s} {imp:.4f}")

    return top_features, top_importance


# ============================================================
# Subgraph Explanation
# ============================================================

def explain_subgraph(model, data, node_idx,
                     num_hops=2, model_name="GCN"):
    """
    Extract and visualize the important subgraph
    around a specific node
    Shows which connections led to attack detection
    """
    print(f"\nExplaining prediction for node {node_idx}...")

    model.eval()

    # Get prediction for this node
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)[node_idx].item()
        confidence = torch.exp(out[node_idx]).max().item()

    label = "ATTACK" if pred == 1 else "NORMAL"
    true_label = "ATTACK" if data.y[node_idx].item() == 1 \
                 else "NORMAL"

    print(f"Node {node_idx}:")
    print(f"  Predicted: {label}")
    print(f"  True Label: {true_label}")
    print(f"  Confidence: {confidence:.4f}")

    # Get neighboring nodes
    edge_index = data.edge_index.numpy()
    neighbors = set()
    neighbors.add(node_idx)

    # Get k-hop neighbors
    current_nodes = {node_idx}
    for hop in range(num_hops):
        new_nodes = set()
        for node in current_nodes:
            mask = edge_index[0] == node
            new_nodes.update(edge_index[1][mask].tolist())
        neighbors.update(new_nodes)
        current_nodes = new_nodes

    neighbors = list(neighbors)[:50]

    # Build subgraph
    G_sub = nx.DiGraph()
    for n in neighbors:
        node_label = data.y[n].item()
        G_sub.add_node(n, label=node_label)

    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0][i], edge_index[1][i]
        if src in neighbors and dst in neighbors:
            G_sub.add_edge(src, dst)

    # Visualize subgraph
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G_sub, seed=42)

    # Color nodes
    node_colors = []
    node_sizes = []
    for n in G_sub.nodes():
        if n == node_idx:
            node_colors.append('yellow')  # Target node
            node_sizes.append(500)
        elif data.y[n].item() == 1:
            node_colors.append('red')     # Attack
            node_sizes.append(200)
        else:
            node_colors.append('lightblue')  # Normal
            node_sizes.append(200)

    nx.draw_networkx(
        G_sub, pos=pos,
        node_color=node_colors,
        node_size=node_sizes,
        with_labels=False,
        arrows=True,
        edge_color='gray',
        alpha=0.8
    )

    # Legend
    plt.scatter([], [], c='yellow',
                label=f'Target Node (Predicted: {label})',
                s=100)
    plt.scatter([], [], c='red',
                label='Attack Node', s=100)
    plt.scatter([], [], c='lightblue',
                label='Normal Node', s=100)
    plt.legend(fontsize=10)
    plt.title(f'{model_name} Subgraph Explanation\n'
              f'Node {node_idx} | Predicted: {label} | '
              f'Confidence: {confidence:.2%}',
              fontsize=12, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()

    # Save
    filename = f'../outputs/{model_name.lower()}' \
               f'_subgraph_node_{node_idx}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Subgraph explanation saved to {filename}")

    return neighbors, pred, confidence


# ============================================================
# Attention Weight Visualization (GAT)
# ============================================================

def visualize_attention_weights(gat_model, data,
                                sample_size=500):
    """
    Visualize GAT attention weights
    Shows which edges the model focused on most
    """
    print("\n" + "=" * 50)
    print("Visualizing GAT Attention Weights...")
    print("=" * 50)

    gat_model.eval()

    with torch.no_grad():
        # Get attention weights
        attention_weights = gat_model.get_attention_weights(
            data.x, data.edge_index
        )

    edge_index, att_weights = attention_weights

    # Average across attention heads
    att_weights_mean = att_weights.mean(dim=1).numpy()

    # Plot attention weight distribution
    plt.figure(figsize=(14, 5))

    # Plot 1 - Distribution
    plt.subplot(1, 2, 1)
    plt.hist(att_weights_mean, bins=50,
             color='steelblue', edgecolor='black',
             alpha=0.7)
    plt.xlabel('Attention Weight', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of GAT Attention Weights',
              fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Plot 2 - Top attended edges visualization
    plt.subplot(1, 2, 2)

    # Get top edges by attention weight
    top_indices = np.argsort(att_weights_mean)[-sample_size:]
    top_edges = edge_index.numpy()[:, top_indices]
    top_weights = att_weights_mean[top_indices]

    # Normalize weights for color mapping
    norm_weights = (top_weights - top_weights.min()) / \
                   (top_weights.max() - top_weights.min())

    plt.scatter(range(len(top_weights)),
                top_weights,
                c=norm_weights,
                cmap='RdYlGn',
                alpha=0.6, s=10)
    plt.colorbar(label='Normalized Attention Weight')
    plt.xlabel('Edge Index', fontsize=12)
    plt.ylabel('Attention Weight', fontsize=12)
    plt.title(f'Top {sample_size} Edge Attention Weights',
              fontsize=13, fontweight='bold')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    filename = '../outputs/gat_attention_weights.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Attention weights plot saved to {filename}")

    return att_weights_mean


# ============================================================
# Complete Explainability Pipeline
# ============================================================

def run_explainability_pipeline(gcn_model, gat_model,
                                data, feature_cols):
    """
    Run complete explainability analysis
    """
    print("=" * 50)
    print("Starting Explainability Pipeline")
    print("=" * 50)

    # 1. GCN Feature Importance
    print("\n[1/4] GCN Feature Importance Analysis")
    gcn_importance = compute_feature_importance(
        gcn_model, data, feature_cols
    )
    plot_feature_importance(
        gcn_importance, feature_cols,
        model_name="GCN", top_n=15
    )

    # 2. GAT Feature Importance
    print("\n[2/4] GAT Feature Importance Analysis")
    gat_importance = compute_feature_importance(
        gat_model, data, feature_cols
    )
    plot_feature_importance(
        gat_importance, feature_cols,
        model_name="GAT", top_n=15
    )

    # 3. Subgraph Explanation
    print("\n[3/4] Subgraph Explanation")

    # Find an attack node to explain
    attack_nodes = (data.y == 1).nonzero(as_tuple=True)[0]
    if len(attack_nodes) > 0:
        node_to_explain = attack_nodes[0].item()
        explain_subgraph(
            gcn_model, data,
            node_idx=node_to_explain,
            model_name="GCN"
        )

    # 4. GAT Attention Weights
    print("\n[4/4] GAT Attention Weight Visualization")
    visualize_attention_weights(gat_model, data)

    print("\n" + "=" * 50)
    print("Explainability Pipeline Complete!")
    print("All plots saved to outputs/ folder")
    print("=" * 50)


# ============================================================
# Run Explainability
# ============================================================

if __name__ == "__main__":

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

    # Step 3: Train both models
    print("\nTraining GCN Model...")
    gcn_model, _, _, _, _ = train_gcn_pipeline(
        data, epochs=100
    )

    print("\nTraining GAT Model...")
    gat_model, _, _, _, _ = train_gat_pipeline(
        data, epochs=100
    )

    # Step 4: Run explainability
    run_explainability_pipeline(
        gcn_model, gat_model,
        data, feature_cols
    )