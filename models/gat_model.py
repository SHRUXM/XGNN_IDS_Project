# ============================================================
# XGNN-Based Intrusion Detection System
# File: models/gat_model.py
# Author: Shreyas Santosh Shinde
# MSc Computer Science - Kirti College, Mumbai
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import numpy as np
import os

# ============================================================
# GAT Model Architecture
# ============================================================

class GATModel(nn.Module):
    """
    Graph Attention Network for Intrusion Detection
    Architecture:
    Input -> GATConv1 (8 heads) -> ELU -> Dropout
          -> GATConv2 (8 heads) -> ELU -> Dropout
          -> GATConv3 (1 head)  -> Output
    """

    def __init__(self, input_dim, hidden_dim=64,
                 output_dim=2, heads=8, dropout=0.5):
        super(GATModel, self).__init__()

        # Three GAT layers with attention heads
        self.conv1 = GATConv(input_dim,
                             hidden_dim,
                             heads=heads,
                             dropout=dropout)

        self.conv2 = GATConv(hidden_dim * heads,
                             hidden_dim,
                             heads=heads,
                             dropout=dropout)

        self.conv3 = GATConv(hidden_dim * heads,
                             output_dim,
                             heads=1,
                             concat=False,
                             dropout=dropout)

        self.dropout = dropout

        print(f"GAT Model initialized!")
        print(f"Input dimensions: {input_dim}")
        print(f"Hidden dimensions: {hidden_dim}")
        print(f"Attention heads: {heads}")
        print(f"Output dimensions: {output_dim}")

    def forward(self, x, edge_index):
        """
        Forward pass through GAT layers
        """
        # Layer 1
        x = F.dropout(x, p=self.dropout,
                      training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)

        # Layer 2
        x = F.dropout(x, p=self.dropout,
                      training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)

        # Layer 3 - Output
        x = F.dropout(x, p=self.dropout,
                      training=self.training)
        x = self.conv3(x, edge_index)

        return F.log_softmax(x, dim=1)

    def get_attention_weights(self, x, edge_index):
        """
        Get attention weights from first GAT layer
        These are used for explainability visualization
        """
        x = F.dropout(x, p=self.dropout,
                      training=self.training)
        x, attention_weights = self.conv1(
            x, edge_index,
            return_attention_weights=True
        )
        return attention_weights


# ============================================================
# Training Function
# ============================================================

def train_gat(model, data, optimizer):
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


def evaluate_gat(model, data, mask):
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

def train_gat_pipeline(data, epochs=100,
                       hidden_dim=64, lr=0.005):
    """
    Complete GAT training pipeline
    """
    print("=" * 50)
    print("Starting GAT Training Pipeline")
    print("=" * 50)

    # Initialize model
    model = GATModel(
        input_dim=data.num_features,
        hidden_dim=64,
        output_dim=2,
        heads=8,
        dropout=0.5
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
        loss = train_gat(model, data, optimizer)

        # Evaluate
        train_acc, _ = evaluate_gat(
            model, data, data.train_mask
        )
        test_acc, pred = evaluate_gat(
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
                 'g-', linewidth=2, label='Training Loss')
    axes[0].set_title('GAT Training Loss',
                      fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2 - Accuracy curves
    axes[1].plot(epochs, train_accuracies,
                 'g-', linewidth=2, label='Train Accuracy')
    axes[1].plot(epochs, test_accuracies,
                 'r-', linewidth=2, label='Test Accuracy')
    axes[1].set_title('GAT Accuracy Curves',
                      fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    os.makedirs('../outputs', exist_ok=True)
    plt.savefig('../outputs/gat_training_results.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot saved to outputs/gat_training_results.png")


# ============================================================
# Run GAT Training
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

    # Step 3: Train GAT
    model, losses, train_accs, test_accs, predictions = \
        train_gat_pipeline(data, epochs=100)

    # Step 4: Plot results
    plot_training_results(losses, train_accs, test_accs)

    # Step 5: Save model
    outputs_dir = os.path.join(base_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    torch.save(model.state_dict(),
               os.path.join(outputs_dir, 'gat_model.pt'))
    print("\nGAT Model saved to outputs/gat_model.pt")
    print("\nReady for Explainability next!")