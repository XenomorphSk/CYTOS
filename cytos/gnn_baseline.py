"""
src/models/gnn_baseline.py

Baseline GNN (GCN ou GAT) - VECTORIZED + MINI-BATCH (2026-06-24).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv


class GNNBaseline(nn.Module):
    def __init__(self, num_nodes: int, hidden_dim: int = 16, num_layers: int = 2, architecture: str = "GCN"):
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


def make_batched_edge_index(edge_index: torch.Tensor, num_nodes: int, batch_size: int) -> torch.Tensor:
    if batch_size == 1:
        return edge_index
    offsets = (torch.arange(batch_size, device=edge_index.device) * num_nodes).repeat_interleave(edge_index.size(1))
    src = edge_index[0].repeat(batch_size) + offsets
    dst = edge_index[1].repeat(batch_size) + offsets
    return torch.stack([src, dst], dim=0)


def train_gnn(
    model: GNNBaseline,
    x_train: torch.Tensor,
    x_train_next: torch.Tensor,
    x_val: torch.Tensor,
    x_val_next: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    lr: float = 0.001,
    weight_decay: float = 1e-5,
    epochs: int = 200,
    seed: int = 0,
    patience: int = 15,
    batch_size: int = 16,
) -> dict:
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    n_train = x_train.shape[0]
    n_val = x_val.shape[0]
    batched_edge_val = make_batched_edge_index(edge_index, num_nodes, n_val)
    x_val_flat = x_val.reshape(-1, 1)
    x_val_next_flat = x_val_next.reshape(-1, 1)

    full_batch_edge = make_batched_edge_index(edge_index, num_nodes, batch_size)
    remainder = n_train % batch_size
    remainder_batch_edge = make_batched_edge_index(edge_index, num_nodes, remainder) if remainder > 0 else None

    best_val_mse = float("inf")
    history = {"train_loss": [], "val_loss": []}
    diverged = False
    epochs_without_improvement = 0
    generator = torch.Generator().manual_seed(seed)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train, generator=generator)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            xb, xb_next = x_train[idx], x_train_next[idx]
            cur_batch_size = xb.shape[0]
            batched_edge = full_batch_edge if cur_batch_size == batch_size else remainder_batch_edge
            xb_flat = xb.reshape(-1, 1)
            xb_next_flat = xb_next.reshape(-1, 1)

            optimizer.zero_grad()
            pred = model(xb_flat, batched_edge)
            loss = F.mse_loss(pred, xb_next_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_train_loss = total_loss / max(n_batches, 1)
        history["train_loss"].append(avg_train_loss)

        if not torch.isfinite(torch.tensor(avg_train_loss)) or avg_train_loss > 1e6:
            diverged = True
            print(f"AVISO: GNN divergiu na epoch {epoch} (train_loss={avg_train_loss:.2e}).")
            break

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val_flat, batched_edge_val)
            val_loss = F.mse_loss(val_pred, x_val_next_flat).item()
            history["val_loss"].append(val_loss)
            if val_loss < best_val_mse - 1e-12:
                best_val_mse = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    return {"history": history, "best_val_mse": best_val_mse, "diverged": diverged}
