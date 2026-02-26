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
                   protocol=row['protocol_type'],
                   service=row['service'],
                   flag=row['flag'],
                   label=row['label'])

    # Add edges based on similar protocol and service
    print("Adding edges between similar connections...")
    edge_count = 0

    # Group by protocol_type and service
    # Connect nodes that share same protocol and service
    groups = df.groupby(['protocol_type', 'service'])

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
    """
    Visualize a sample of the graph
    Red nodes = Attack
    Blue nodes = Normal
    """
    print("\nGenerating graph visualization...")

    # Take a small sample for visualization
    sample_nodes = list(G.nodes())[:sample_size]
    subgraph = G.subgraph(sample_nodes)

    # Get node colors based on label
    colors = []
    for node in subgraph.nodes():
        label = G.nodes[node].get('label', 0)
        if label == 1:
            colors.append('red')      # Attack
        else:
            colors.append('blue')     # Normal

    # Plot
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(subgraph, seed=42)

    nx.draw_networkx(
        subgraph,
        pos=pos,
        node_color=colors,
        node_size=100,
        with_labels=False,
        arrows=True,
        edge_color='gray',
        alpha=0.7
    )

    # Add legend
    plt.scatter([], [], c='red',
                label='Attack (Anomaly)')
    plt.scatter([], [], c='blue',
                label='Normal Traffic')
    plt.legend(fontsize=12)
    plt.title(title, fontsize=16, fontweight='bold')
    plt.axis('off')

    # Save the plot
    os.makedirs('../outputs', exist_ok=True)
    plt.savefig('../outputs/network_graph.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("Graph saved to outputs/network_graph.png")


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
    # Connect nodes based on protocol and service similarity
    edge_list = []
    groups = df.groupby(['protocol_type', 'service'])

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
    visualize_graph(G, title="XGNN Network Intrusion Graph")

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