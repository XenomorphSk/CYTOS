"""
cytos/rollout.py

Fase II (piloto exploratorio, Secao 27 do pre-registro): testa se a
vantagem de predicao de 1 passo (H1/H1b) persiste quando o modelo e
usado em rollout autoregressivo, sem retreinar especificamente para
essa tarefa.

IMPORTANTE: pre-requisito da Fase II completa (validacao em dados reais)
nao foi atendido. Piloto exploratorio em benchmark simulado.
"""

from __future__ import annotations

import numpy as np
import torch

from cytos.gnn_baseline import make_batched_edge_index


def rollout_ttn(model, x_start, n_steps):
    model.eval()
    trajectory = []
    current = x_start.clone()

    with torch.no_grad():
        for _ in range(n_steps):
            pred = model(current.unsqueeze(0)).squeeze(0)
            trajectory.append(pred.clone())
            current = pred

    return torch.stack(trajectory, dim=0)


def rollout_gnn(model, x_start, edge_index, num_nodes, n_steps):
    model.eval()
    trajectory = []
    current = x_start.clone()
    batched_edge = make_batched_edge_index(edge_index, num_nodes, 1)

    with torch.no_grad():
        for _ in range(n_steps):
            pred = model(current.unsqueeze(-1), batched_edge).squeeze(-1)
            trajectory.append(pred.clone())
            current = pred

    return torch.stack(trajectory, dim=0)


def rollout_mse_per_step(predicted_trajectory, true_trajectory):
    n_steps = min(predicted_trajectory.shape[0], true_trajectory.shape[0])
    pred = predicted_trajectory[:n_steps]
    true = true_trajectory[:n_steps]
    mse_per_step = ((pred - true) ** 2).mean(dim=1).numpy()
    return mse_per_step
