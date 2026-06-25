"""
src/models/gnn_baseline.py

Baseline GNN (GCN ou GAT) para predicao de expressao genica t -> t+1.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv


class GNNBaseline(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        hidden_dim: int = 16,
        num_layers: int = 2,
        architecture: str = "GCN",
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.architecture = architecture

        conv_cls = GCNConv if architecture == "GCN" else GATConv

        layers = []
        in_dim = 1
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else 1
            layers.append(conv_cls(in_dim, out_dim))
            in_dim = out_dim
        self.convs = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < len(self.convs) - 1:
                h = F.relu(h)
        return h

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_gnn(
    model: GNNBaseline,
    train_data: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    val_data: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    lr: float = 0.001,
    weight_decay: float = 1e-5,
    epochs: int = 200,
    seed: int = 0,
    patience: int = 15,
) -> dict:
    """
    OTIMIZACAO (2026-06-24): early stopping com paciencia configuravel.
    Para de treinar se val_loss nao melhorar por `patience` epochs seguidas.
    Otimizacao de velocidade, nao de criterio cientifico.
    """
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_mse = float("inf")
    history = {"train_loss": [], "val_loss": []}
    diverged = False
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x_t, edge_index, x_next in train_data:
            optimizer.zero_grad()
            pred = model(x_t, edge_index)
            loss = F.mse_loss(pred, x_next)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / max(len(train_data), 1)
        history["train_loss"].append(avg_train_loss)

        if not torch.isfinite(torch.tensor(avg_train_loss)) or avg_train_loss > 1e6:
            diverged = True
            print(f"AVISO: GNN divergiu na epoch {epoch} (train_loss={avg_train_loss:.2e}).")
            break

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for x_t, edge_index, x_next in val_data:
                pred = model(x_t, edge_index)
                val_loss += F.mse_loss(pred, x_next).item()
            val_loss /= max(len(val_data), 1)
            history["val_loss"].append(val_loss)
            if val_loss < best_val_mse - 1e-12:
                best_val_mse = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    return {"history": history, "best_val_mse": best_val_mse, "diverged": diverged}
