"""
cytos/dynamic_hierarchy.py

Constroi hierarquia para a TTN a partir de correlacao com lag temporal
(x_i(t) vs x_j(t+1)), em vez de topologia estatica ou anotacao
funcional estatica. Secao 35 do pre-registro (H7).
"""

from __future__ import annotations

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def compute_lag_correlation_matrix(trajectories, gene_names):
    n = len(gene_names)
    xs, x_nexts = [], []
    for traj in trajectories:
        traj = np.asarray(traj)
        for t in range(len(traj) - 1):
            xs.append(traj[t])
            x_nexts.append(traj[t + 1])
    X = np.array(xs)
    X_next = np.array(x_nexts)

    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            xi = X[:, i]
            xj_next = X_next[:, j]
            if np.std(xi) < 1e-10 or np.std(xj_next) < 1e-10:
                C[i, j] = 0.0
            else:
                C[i, j] = np.corrcoef(xi, xj_next)[0, 1]
    return C


def build_lag_correlation_hierarchy(trajectories, gene_names, n_clusters=None):
    n = len(gene_names)
    if n_clusters is None:
        n_clusters = max(2, n // 15)

    C = compute_lag_correlation_matrix(trajectories, gene_names)
    dist = 1 - np.abs(C)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2
    dist = np.clip(dist, 0, None)

    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="ward")
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")

    partition = {gene_names[i]: int(labels[i]) for i in range(n)}
    return {"level_0": partition}
