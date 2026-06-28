"""
cytos/topology_inference.py

Fase 3: inferencia de topologia via sensibilidade a perturbacao.

Diferente de H1/H1b/H2, aqui o grafo verdadeiro NAO e usado para definir
a hierarquia da TTN (seria circular). A hierarquia e arbitraria
(agrupamento sequencial dos genes). O baseline e um MLP simples, nao uma
GNN - GNN exige edge_index conhecido a priori, o que tambem seria
circular para esta tarefa.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score


def arbitrary_hierarchy(gene_names: list, group_size: int = None) -> dict:
    n = len(gene_names)
    if group_size is None:
        group_size = max(2, int(round(n ** 0.5)))

    partition = {}
    for idx, gene in enumerate(gene_names):
        partition[gene] = idx // group_size

    return {"level_0": partition}


class MLPBaseline(nn.Module):
    def __init__(self, n_genes: int, hidden_dim: int = 32, num_layers: int = 2):
        super().__init__()
        layers = []
        in_dim = n_genes
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else n_genes
            layers.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i < len(self.layers) - 1:
                h = torch.relu(h)
        return h

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_mlp(model, x_train, x_train_next, x_val, x_val_next, lr=0.001, weight_decay=1e-5, epochs=200, seed=0, patience=15, batch_size=16):
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    n_train = x_train.shape[0]
    best_val_mse = float("inf")
    epochs_without_improvement = 0
    diverged = False
    generator = torch.Generator().manual_seed(seed)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train, generator=generator)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            xb, xb_next = x_train[idx], x_train_next[idx]
            optimizer.zero_grad()
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, xb_next)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_train_loss = total_loss / max(n_batches, 1)

        if not torch.isfinite(torch.tensor(avg_train_loss)) or avg_train_loss > 1e6:
            diverged = True
            break

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val)
            val_loss = nn.functional.mse_loss(val_pred, x_val_next).item()
            if val_loss < best_val_mse - 1e-12:
                best_val_mse = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break

    return {"best_val_mse": best_val_mse, "diverged": diverged}


def edge_scores_via_perturbation(model, n_genes, x_test):
    scores = np.zeros((n_genes, n_genes))
    with torch.no_grad():
        pred_baseline = model(x_test)
        for i in range(n_genes):
            x_knockout = x_test.clone()
            x_knockout[:, i] = 0.0
            pred_knockout = model(x_knockout)
            diff = (pred_knockout - pred_baseline).abs().mean(dim=0)
            scores[i, :] = diff.numpy()
            scores[i, i] = 0.0
    return scores


def evaluate_against_gold_standard(scores, gold_graph, gene_names):
    n = len(gene_names)
    y_true = np.zeros(n * n - n)
    y_score = np.zeros(n * n - n)
    k = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            gi, gj = gene_names[i], gene_names[j]
            y_true[k] = 1.0 if gold_graph.has_edge(gi, gj) else 0.0
            y_score[k] = scores[i, j]
            k += 1

    aupr = average_precision_score(y_true, y_score)
    auroc = roc_auc_score(y_true, y_score)
    return {"aupr": float(aupr), "auroc": float(auroc), "n_true_edges": int(y_true.sum())}
