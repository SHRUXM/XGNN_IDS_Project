# ============================================================
# XGNN-Based Intrusion Detection System
# File: explainability/attention_viz.py
# Author: Shreyas Santosh Shinde
# MSc Computer Science - Kirti College, Mumbai
# ============================================================

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import os
import sys

sys.path.append(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))

from models.gat_model import GATModel, train_gat_pipeline
from preprocessing.data_loader import preprocess_pipeline
from preprocessing.graph_builder import build_graph_pipeline


# ============================================================
# Attention Weight Extraction
# ============================================================

def extract_attention_weights(gat_model, data):
    """
    Extract attention weights from all GAT layers
    """
    print("=" * 50)
    print("Extracting GAT Attention Weights...")
    print("=" * 50)

    gat_model.eval()

    with torch.no_grad():
        edge_index, att_weights = \
            gat_model.get_attention_weights(
                data.x, data.edge_index
            )

    # Average across attention heads
    att_mean = att_weights.mean(dim=1).numpy()
    att_per_head = att_weights.numpy()

    print(f"Total edges analyzed: {len(att_mean)}")
    print(f"Attention heads: {att_per_head.shape[1]}")
    print(f"Max attention: {att_mean.max():.4f}")
    print(f"Min attention: {att_mean.min():.4f}")
    print(f"Mean attention: {att_mean.mean():.4f}")

    return edge_index, att_mean, att_per_head


# ============================================================
# Plot Attention Per Head
# ============================================================

def plot_attention_per_head(att_per_head,
                            num_heads=8):
    """
    Plot attention weight distribution
    for each attention head separately
    """
    print("\nPlotting attention per head...")

    fig, axes = plt.subplots(
        2, 4, figsize=(16, 8)
    )
    axes = axes.flatten()

    colors = plt.cm.tab10(
        np.linspace(0, 1, num_heads)
    )

    for head in range(min(num_heads,
                          att_per_head.shape[1])):
        head_weights = att_per_head[:, head]

        axes[head].hist(
            head_weights, bins=50,
            color=colors[head],
            edgecolor='black',
            alpha=0.7
        )
        axes[head].set_title(
            f'Attention Head {head + 1}',
            fontsize=11, fontweight='bold'
        )
        axes[head].set_xlabel(
            'Attention Weight', fontsize=9
        )
        axes[head].set_ylabel(
            'Frequency', fontsize=9
        )
        axes[head].grid(True, alpha=0.3)
        axes[head].axvline(
            x=head_weights.mean(),
            color='red', linestyle='--',
            label=f'Mean: {head_weights.mean():.3f}'
        )
        axes[head].legend(fontsize=8)

    plt.suptitle(
        'GAT Attention Weight Distribution '
        'Per Head',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()

    os.makedirs('../outputs', exist_ok=True)
    plt.savefig(
        '../outputs/attention_per_head.png',
        dpi=150, bbox_inches='tight'
    )
    plt.show()
    print("Saved to outputs/attention_per_head.png")


# ============================================================
# Plot Top Attended Nodes
# ============================================================

def plot_top_attended_nodes(edge_index,
                             att_mean, data,
                             top_n=50):
    """
    Visualize the subgraph of most
    highly attended edges
    """
    print(f"\nPlotting top {top_n} attended edges...")

    # Get top N edges by attention weight
    top_idx = np.argsort(att_mean)[-top_n:]
    top_edges = edge_index.numpy()[:, top_idx]
    top_weights = att_mean[top_idx]

    # Build subgraph
    G = nx.DiGraph()
    unique_nodes = np.unique(top_edges)

    for node in unique_nodes:
        label = data.y[node].item()
        G.add_node(node, label=label)

    for i in range(top_edges.shape[1]):
        src = top_edges[0][i]
        dst = top_edges[1][i]
        weight = top_weights[i]
        G.add_edge(src, dst, weight=weight)

    # Plot
    plt.figure(figsize=(12, 9))
    pos = nx.spring_layout(G, seed=42, k=2)

    # Node colors
    node_colors = [
        'red' if data.y[n].item() == 1
        else 'lightblue'
        for n in G.nodes()
    ]

    # Edge colors based on attention weight
    edge_weights = [
        G[u][v]['weight']
        for u, v in G.edges()
    ]
    edge_colors = plt.cm.YlOrRd(
        np.array(edge_weights) /
        max(edge_weights)
    )

    # Draw
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=300,
        alpha=0.9
    )
    nx.draw_networkx_edges(
        G, pos,
        edge_color=edge_colors,
        width=2,
        arrows=True,
        arrowsize=15,
        alpha=0.8
    )

    # Legend
    plt.scatter([], [], c='red',
                label='Attack Node', s=100)
    plt.scatter([], [], c='lightblue',
                label='Normal Node', s=100)
    plt.plot([], [], color='darkred',
             linewidth=2,
             label='High Attention Edge')
    plt.plot([], [], color='lightyellow',
             linewidth=2,
             label='Low Attention Edge')
    plt.legend(fontsize=10, loc='upper left')

    plt.title(
        f'Top {top_n} Most Attended Edges\n'
        f'in GAT Network — '
        f'Red edges = High Attention',
        fontsize=13, fontweight='bold'
    )
    plt.axis('off')
    plt.tight_layout()

    plt.savefig(
        '../outputs/top_attended_nodes.png',
        dpi=150, bbox_inches='tight'
    )
    plt.show()
    print("Saved to outputs/top_attended_nodes.png")


# ============================================================
# Attack vs Normal Attention Comparison
# ============================================================

def plot_attack_vs_normal_attention(edge_index,
                                     att_mean,
                                     data):
    """
    Compare attention weights for
    attack vs normal connections
    """
    print("\nComparing attack vs normal attention...")

    edge_idx = edge_index.numpy()

    attack_att = []
    normal_att = []

    for i in range(edge_idx.shape[1]):
        src = edge_idx[0][i]
        dst = edge_idx[1][i]
        weight = att_mean[i]

        # Check if either node is attack
        if (data.y[src].item() == 1 or
                data.y[dst].item() == 1):
            attack_att.append(weight)
        else:
            normal_att.append(weight)

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram comparison
    axes[0].hist(
        normal_att, bins=50,
        alpha=0.7, color='steelblue',
        label=f'Normal ({len(normal_att)} edges)',
        edgecolor='black'
    )
    axes[0].hist(
        attack_att, bins=50,
        alpha=0.7, color='red',
        label=f'Attack ({len(attack_att)} edges)',
        edgecolor='black'
    )
    axes[0].set_xlabel('Attention Weight',
                       fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title(
        'Attention Weight Distribution\n'
        'Attack vs Normal Connections',
        fontsize=12, fontweight='bold'
    )
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Box plot comparison
    axes[1].boxplot(
        [normal_att, attack_att],
        labels=['Normal', 'Attack'],
        patch_artist=True,
        boxprops=dict(facecolor='lightblue'),
        medianprops=dict(color='red',
                         linewidth=2)
    )
    axes[1].set_ylabel('Attention Weight',
                       fontsize=12)
    axes[1].set_title(
        'Attention Weight Boxplot\n'
        'Attack vs Normal',
        fontsize=12, fontweight='bold'
    )
    axes[1].grid(True, alpha=0.3)

    # Print statistics
    print(f"\nNormal connections:")
    print(f"  Mean attention: "
          f"{np.mean(normal_att):.4f}")
    print(f"  Max attention:  "
          f"{np.max(normal_att):.4f}")
    print(f"\nAttack connections:")
    print(f"  Mean attention: "
          f"{np.mean(attack_att):.4f}")
    print(f"  Max attention:  "
          f"{np.max(attack_att):.4f}")

    plt.tight_layout()
    plt.savefig(
        '../outputs/attack_vs_normal_attention.png',
        dpi=150, bbox_inches='tight'
    )
    plt.show()
    print("Saved to "
          "outputs/attack_vs_normal_attention.png")


# ============================================================
# Run Full Attention Visualization Pipeline
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

    # Preprocess
    train_df, test_df, scaler, feature_cols = \
        preprocess_pipeline(train_path, test_path)

    # Build graph
    data, G = build_graph_pipeline(
        train_df, feature_cols
    )

    # Train GAT
    print("\nTraining GAT Model...")
    gat_model, _, _, _, _ = train_gat_pipeline(
        data, epochs=100
    )

    # Extract attention weights
    edge_index, att_mean, att_per_head = \
        extract_attention_weights(gat_model, data)

    # Plot per head
    plot_attention_per_head(att_per_head)

    # Plot top attended nodes
    plot_top_attended_nodes(
        edge_index, att_mean, data, top_n=50
    )

    # Attack vs normal comparison
    plot_attack_vs_normal_attention(
        edge_index, att_mean, data
    )

    print("\n" + "=" * 50)
    print("Attention Visualization Complete!")
    print("All plots saved to outputs/ folder")
    print("=" * 50)