# ============================================================
# XGNN-Based Intrusion Detection System
# File: preprocessing/graph_builder.py
# Author: Shreyas Santosh Shinde
# MSc Computer Science - Kirti College, Mumbai
# ============================================================

import pandas as pd
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os

def build_networkx_graph(df, feature_cols):
    """
    Build a NetworkX graph from the dataset
    Each unique connection type becomes a node
    Edges represent relationships between connections
    """
    print("=" * 50)
    print("Building NetworkX Graph...")
    print("=" * 50)

    # Create empty directed graph
    G = nx.DiGraph()

    # Add nodes - each row becomes a node
    print(f"\nAdding {len(df)} nodes to graph...")
    for idx, row in df.iterrows():
        G.add_node(idx,
                   protocol=row['proto'],
                   service=row['service'],
                   state=row['state'],
                   label=row['label'])

        # Add edges based on similar proto and service
    print("Adding edges between similar connections...")
    edge_count = 0

    # Group by proto and service
    # Connect nodes that share same proto and service
    groups = df.groupby(['proto', 'service'])

    for name, group in groups:
        indices = group.index.tolist()
        # Connect consecutive nodes in same group
        for i in range(len(indices) - 1):
            G.add_edge(indices[i], indices[i+1],
                      weight=1.0)
            edge_count += 1

    print(f"Total nodes: {G.number_of_nodes()}")
    print(f"Total edges: {G.number_of_edges()}")

    return G


def visualize_graph(G, title="Network Traffic Graph",
                   sample_size=100):
    import random

    # Get attack and normal nodes
    attack_nodes = [n for n, d in G.nodes(data=True)
                    if d.get('label') == 1]
    normal_nodes = [n for n, d in G.nodes(data=True)
                    if d.get('label') == 0]

    # Pick seed nodes that HAVE edges
    nodes_with_edges = [n for n in G.nodes()
                        if G.degree(n) > 0]

    # Pick 10 random seed nodes and expand their neighbors
    seeds = random.sample(
        nodes_with_edges,
        min(10, len(nodes_with_edges))
    )

    # Collect seeds + their neighbors
    expanded = set(seeds)
    for s in seeds:
        expanded.update(list(G.neighbors(s))[:15])

    # Now balance attack vs normal within expanded set
    exp_attack = [n for n in expanded
                  if G.nodes[n].get('label') == 1]
    exp_normal = [n for n in expanded
                  if G.nodes[n].get('label') == 0]

    # Fill remaining from full graph if needed
    need_attack = max(0, 50 - len(exp_attack))
    need_normal = max(0, 50 - len(exp_normal))

    if need_attack > 0:
        extras = [n for n in attack_nodes
                  if n not in expanded]
        exp_attack += random.sample(
            extras, min(need_attack, len(extras))
        )
    if need_normal > 0:
        extras = [n for n in normal_nodes
                  if n not in expanded]
        exp_normal += random.sample(
            extras, min(need_normal, len(extras))
        )

    sampled = list(set(exp_attack[:50] + exp_normal[:50]))
    subgraph = G.subgraph(sampled)

    # Node colors
    colors = []
    for node in subgraph.nodes():
        label = G.nodes[node].get('label', 0)
        colors.append('red' if label == 1 else 'blue')

    # Plot with spring layout for natural flow
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(subgraph, seed=42, k=2.5)

    # Draw edges first
    nx.draw_networkx_edges(
        subgraph, pos,
        edge_color='gray',
        arrows=True,
        alpha=0.5,
        width=0.8,
        arrowsize=10
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        subgraph, pos,
        node_color=colors,
        node_size=120,
        alpha=0.9
    )

    # Legend
    plt.scatter([], [], c='red', s=80,
                label='Attack (Anomaly)')
    plt.scatter([], [], c='blue', s=80,
                label='Normal Traffic')
    plt.legend(fontsize=12)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.axis('off')

    # Fix save path
    base_dir = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
    out_dir = os.path.join(base_dir, 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'network_graph.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Graph saved to {save_path}")

def convert_to_pytorch_geometric(df, feature_cols):
    """
    Convert dataframe to PyTorch Geometric Data object
    This is the format our GCN and GAT models need
    """
    print("\nConverting to PyTorch Geometric format...")

    # Node features matrix
    X = torch.tensor(
        df[feature_cols].values,
        dtype=torch.float
    )

    # Labels
    y = torch.tensor(
        df['label'].values,
        dtype=torch.long
    )

    # Build edge index
    # Connect nodes based on proto and service similarity
    edge_list = []
    groups = df.groupby(['proto', 'service'])

    for name, group in groups:
        indices = group.index.tolist()
        # Reset indices to 0-based
        local_indices = [
            df.index.get_loc(i) for i in indices
        ]
        for i in range(len(local_indices) - 1):
            edge_list.append(
                [local_indices[i], local_indices[i+1]]
            )
            edge_list.append(
                [local_indices[i+1], local_indices[i]]
            )

    # Convert to tensor
    if edge_list:
        edge_index = torch.tensor(
            edge_list, dtype=torch.long
        ).t().contiguous()
    else:
        edge_index = torch.zeros(
            (2, 0), dtype=torch.long
        )

    print(f"Node feature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Edge index shape: {edge_index.shape}")

    # Create PyG Data object
    data = Data(x=X, edge_index=edge_index, y=y)

    return data


def create_train_test_masks(data, df, test_size=0.2):
    """
    Create train and test masks for the graph
    """
    print("\nCreating train/test masks...")

    num_nodes = data.num_nodes
    indices = list(range(num_nodes))

    # Split indices
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=42,
        stratify=df['label'].values
    )

    # Create boolean masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    test_mask[test_idx] = True

    data.train_mask = train_mask
    data.test_mask = test_mask

    print(f"Training nodes: {train_mask.sum().item()}")
    print(f"Testing nodes: {test_mask.sum().item()}")

    return data


def build_graph_pipeline(df, feature_cols):
    """
    Complete graph building pipeline
    """
    print("=" * 50)
    print("Starting Graph Building Pipeline")
    print("=" * 50)

    # Step 1: Build NetworkX graph for visualization
    G = build_networkx_graph(df, feature_cols)

    # Step 2: Visualize the graph
    visualize_graph(G, title="UNSW-NB15 Network Intrusion Graph")

    # Step 3: Convert to PyTorch Geometric
    data = convert_to_pytorch_geometric(df, feature_cols)

    # Step 4: Create train/test masks
    data = create_train_test_masks(data, df)

    print("\n" + "=" * 50)
    print("Graph Building Complete!")
    print(f"Graph Summary:")
    print(f"  Nodes: {data.num_nodes}")
    print(f"  Edges: {data.num_edges}")
    print(f"  Features per node: {data.num_features}")
    print(f"  Classes: {data.y.unique()}")
    print("=" * 50)

    return data, G


# ============================================================
# Test the graph builder
# ============================================================
if __name__ == "__main__":

    import os
    from data_loader import preprocess_pipeline

    # Get the correct path automatically
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, "data", "Train_data.csv")
    test_path = os.path.join(base_dir, "data", "Test_data.csv")

    print(f"Looking for dataset at: {train_path}")

    # Run preprocessing first
    train_df, test_df, scaler, feature_cols = preprocess_pipeline(
        train_path, test_path
    )

    # Build graph
    data, G = build_graph_pipeline(train_df, feature_cols)

    # Save the PyG data object
    outputs_dir = os.path.join(base_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    torch.save(data, os.path.join(outputs_dir, 'graph_data.pt'))
    print("\nGraph data saved to outputs/graph_data.pt")
    print("\nReady to train GCN and GAT models!")