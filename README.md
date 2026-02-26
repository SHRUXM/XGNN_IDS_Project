# Explainable Graph Neural Networks (XGNNs) for Network Intrusion Detection

## Author
Shreyas Santosh Shinde
MSc Computer Science — Kirti College, Mumbai

## Project Overview
This project implements an Explainable GNN-based 
Intrusion Detection System using GCN and GAT architectures
with XAI techniques for transparent threat detection.

## Dataset
- CICIDS2017
- UNSW-NB15

## Technologies Used
- Python 3.10
- PyTorch & PyTorch Geometric
- Scikit-learn
- NetworkX
- Captum (Explainability)

## Installation
pip install -r requirements.txt

## Project Structure
- data/ — datasets
- models/ — GCN and GAT model files
- preprocessing/ — data loading and graph construction
- explainability/ — GNNExplainer and attention visualization
- evaluation/ — metrics and comparison
- notebooks/ — Jupyter notebooks