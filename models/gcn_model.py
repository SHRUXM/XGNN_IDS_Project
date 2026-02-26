# ============================================================
# XGNN-Based Intrusion Detection System
# File: models/gcn_model.py
# Author: Shreyas Santosh Shinde
# MSc Computer Science - Kirti College, Mumbai
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================================
# GCN Model Architecture
# ============================================================

class GCNModel(nn.Module):
    """
    Graph Convolutional Network for Intrusion Detection
    Architecture:
    Input -> GCNConv1 -> ReLU -> Dropout
          -> GCNConv2 -> ReLU -> Dropout
          -> GCNConv3 -> Output
    """

    def __init__(self, input_dim, hidden_dim=64,
                 output_dim=2, dropout=0.5):
        super(GCNModel, self).__init__()

        # Three GCN layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)

        # Dropout for regularization
        self.dropout = dropout

        print(f"GCN Model initialized!")
        print(f"Input dimensions: {input_dim}")
        print(f"Hidden dimensions: {hidden_dim}")
        print(f"Output dimensions: {output_dim}")

    def forward(self, x, edge_index):
        """
        Forward pass through GCN layers
        """
        # Layer 1
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout,
                      training=self.training)

        # Layer 2
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout,
                      training=self.training)

        # Layer 3 - Output layer
        x = self.conv3(x, edge_index)

        return F.log_softmax(x, dim=1)

    def get_embeddings(self, x, edge_index):
        """
        Get node embeddings from second layer
        Used for visualization
        """
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


# ============================================================
# Training Function
# ============================================================

def train_gcn(model, data, optimizer):
    """
    Single training step
    """
    model.train()
    optimizer.zero_grad()

    # Forward pass
    out = model(data.x, data.edge_index)

    # Calculate loss on training nodes only
    loss = F.nll_loss(
        out[data.train_mask],
        data.y[data.train_mask]
    )

    # Backward pass
    loss.backward()
    optimizer.step()

    return loss.item()


def evaluate_gcn(model, data, mask):
    """
    Evaluate model accuracy
    """
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)

        # Calculate accuracy
        correct = pred[mask] == data.y[mask]
        accuracy = correct.sum().item() / mask.sum().item()

    return accuracy, pred


# ============================================================
# Training Pipeline
# ============================================================

def train_gcn_pipeline(data, epochs=100,
                       hidden_dim=64, lr=0.01):
    """
    Complete GCN training pipeline
    """
    print("=" * 50)
    print("Starting GCN Training Pipeline")
    print("=" * 50)

    # Initialize model
    model = GCNModel(
        input_dim=data.num_features,
        hidden_dim=hidden_dim,
        output_dim=2
    )

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=5e-4
    )

    # Training history
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    print(f"\nTraining for {epochs} epochs...")
    print("-" * 50)

    # Training loop
    for epoch in range(1, epochs + 1):

        # Train
        loss = train_gcn(model, data, optimizer)

        # Evaluate
        train_acc, _ = evaluate_gcn(
            model, data, data.train_mask
        )
        test_acc, pred = evaluate_gcn(
            model, data, data.test_mask
        )

        # Store history
        train_losses.append(loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        # Print every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | "
                  f"Loss: {loss:.4f} | "
                  f"Train Acc: {train_acc:.4f} | "
                  f"Test Acc: {test_acc:.4f}")

    print("-" * 50)
    print(f"Training Complete!")
    print(f"Final Test Accuracy: {test_acc:.4f}")

    return model, train_losses, train_accuracies, \
           test_accuracies, pred


# ============================================================
# Plot Training Results
# ============================================================

def plot_training_results(train_losses,
                          train_accuracies,
                          test_accuracies):
    """
    Plot training loss and accuracy curves
    """
    print("\nGenerating training plots...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(train_losses) + 1)

    # Plot 1 - Loss curve
    axes[0].plot(epochs, train_losses,
                 'b-', linewidth=2, label='Training Loss')
    axes[0].set_title('GCN Training Loss',
                      fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2 - Accuracy curves
    axes[1].plot(epochs, train_accuracies,
                 'b-', linewidth=2, label='Train Accuracy')
    axes[1].plot(epochs, test_accuracies,
                 'r-', linewidth=2, label='Test Accuracy')
    axes[1].set_title('GCN Accuracy Curves',
                      fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    os.makedirs('../outputs', exist_ok=True)
    plt.savefig('../outputs/gcn_training_results.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved to outputs/gcn_training_results.png")


# ============================================================
# Run GCN Training
# ============================================================

if __name__ == "__main__":

    import sys
    import os
    sys.path.append(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    ))

    from preprocessing.data_loader import preprocess_pipeline
    from preprocessing.graph_builder import build_graph_pipeline

    # Paths
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

    # Step 3: Train GCN
    model, losses, train_accs, test_accs, predictions = \
        train_gcn_pipeline(data, epochs=100)

    # Step 4: Plot results
    plot_training_results(losses, train_accs, test_accs)

    # Step 5: Save model
    outputs_dir = os.path.join(base_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    torch.save(model.state_dict(),
               os.path.join(outputs_dir, 'gcn_model.pt'))
    print("\nGCN Model saved to outputs/gcn_model.pt")
    print("\nReady to build GAT Model next!")
