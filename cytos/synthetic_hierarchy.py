"""
cytos/synthetic_hierarchy.py

Gera dados sinteticos de dinamica com hierarquia de comunidade
VERDADEIRA POR CONSTRUCAO - para o experimento H6 (Secao 33), que isola
se o problema da Fase I era ruido de medicao e/ou tamanho de amostra.

Sistema: x_{t+1} = x_t + dt*(-decay*x_t + tanh(A @ x_t)) + ruido, onde A
e bloco-diagonal por construcao. Saturacao via tanh garante estabilidade
numerica independente da escala de A.
"""

from __future__ import annotations

import numpy as np


def build_true_hierarchy(n_genes, comm_size):
    gene_names = [f"G{i}" for i in range(n_genes)]
    partition = {gene_names[i]: i // comm_size for i in range(n_genes)}
    return gene_names, {"level_0": partition}


def build_coupling_matrix(gene_names, partition, within_std=1.0, between_std=0.05, seed=0):
    rng = np.random.default_rng(seed)
    n = len(gene_names)
    A = np.zeros((n, n))
    for i, gi in enumerate(gene_names):
        for j, gj in enumerate(gene_names):
            if partition[gi] == partition[gj]:
                A[i, j] = rng.standard_normal() * within_std
            else:
                A[i, j] = rng.standard_normal() * between_std
    return A


def generate_trajectory(A, n_genes, n_steps, dt=0.1, decay=0.5, noise_std=0.01, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n_genes) * 0.5
    traj = [x.copy()]
    for _ in range(n_steps):
        dx = -decay * x + np.tanh(A @ x)
        x = x + dt * dx + rng.standard_normal(n_genes) * noise_std
        traj.append(x.copy())
    return np.array(traj, dtype=np.float32)


def generate_dataset(n_genes=100, comm_size=10, noise_std=0.01, n_train_trajectories=10,
                      n_steps_per_traj=20, dt=0.1, decay=0.5, coupling_seed=0, data_seed=0):
    gene_names, hierarchy = build_true_hierarchy(n_genes, comm_size)
    partition = hierarchy["level_0"]
    A = build_coupling_matrix(gene_names, partition, seed=coupling_seed)

    train_trajs = [
        generate_trajectory(A, n_genes, n_steps_per_traj, dt, decay, noise_std, seed=data_seed * 1000 + i)
        for i in range(n_train_trajectories)
    ]
    val_traj = generate_trajectory(A, n_genes, n_steps_per_traj, dt, decay, noise_std, seed=data_seed * 1000 + 9000)
    test_traj = generate_trajectory(A, n_genes, n_steps_per_traj, dt, decay, noise_std, seed=data_seed * 1000 + 9001)

    return {
        "gene_names": gene_names, "hierarchy": hierarchy, "coupling_matrix": A,
        "train_trajectories": train_trajs, "val_trajectory": val_traj, "test_trajectory": test_traj,
    }
