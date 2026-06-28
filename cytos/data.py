"""
cytos/data.py

Utilidades de dados agnosticas de dataset: deteccao de comunidades,
split de trajetorias, empacotamento em arrays. Nenhuma dependencia do
DREAM4 ou de qualquer formato de arquivo especifico - o usuario traz seu
proprio grafo (networkx.DiGraph) e suas proprias trajetorias (lista de
arrays numpy, cada um de shape (n_timepoints, n_genes)).
"""

from __future__ import annotations

import networkx as nx
import numpy as np

try:
    import community as community_louvain
except ImportError:
    community_louvain = None


def detect_hierarchy(graph: nx.DiGraph, method: str = "louvain") -> dict:
    undirected = graph.to_undirected()

    if method == "louvain":
        if community_louvain is None:
            raise ImportError(
                "Instale 'python-louvain' para usar method='louvain': "
                "pip install python-louvain"
            )
        partition = community_louvain.best_partition(undirected)
    elif method == "spectral":
        from sklearn.cluster import SpectralClustering

        adj = nx.to_numpy_array(undirected)
        n_clusters = max(2, len(undirected.nodes) // 4)
        labels = SpectralClustering(
            n_clusters=n_clusters, affinity="precomputed", random_state=0
        ).fit_predict(adj)
        partition = {node: int(label) for node, label in zip(undirected.nodes, labels)}
    else:
        raise ValueError(f"Metodo de clustering desconhecido: {method}")

    return {"level_0": partition}


def split_trajectories_by_fraction(
    trajectories: list,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    test_frac: float = 0.2,
):
    n = len(trajectories)
    if n < 3:
        raise ValueError(
            f"Sao necessarias pelo menos 3 trajetorias (treino/val/teste), "
            f"mas apenas {n} foram fornecidas. Se voce so tem uma serie "
            f"temporal longa, divida-a manualmente em sub-trajetorias "
            f"antes de usar esta biblioteca."
        )
    n_train = max(1, int(round(n * train_frac)))
    n_val = max(1, int(round(n * val_frac)))
    n_test = max(1, n - n_train - n_val)
    if n_train + n_val + n_test > n:
        n_test = n - n_train - n_val

    train = trajectories[:n_train]
    val = trajectories[n_train : n_train + n_val]
    test = trajectories[n_train + n_val : n_train + n_val + n_test]
    return train, val, test


def trajectories_to_stacked_arrays(trajectories: list):
    xs, x_nexts = [], []
    for traj in trajectories:
        for t in range(len(traj) - 1):
            xs.append(traj[t])
            x_nexts.append(traj[t + 1])
    return np.array(xs, dtype=np.float32), np.array(x_nexts, dtype=np.float32)
