"""
src/data/pipeline.py

Pipeline de dados para o experimento CYTOS (TTN vs GNN), usando os dados
REAIS do DREAM4 In Silico Networks Challenge (obtidos via Synapse).

REESCRITO (2026-06-24): nao existe "noise_levels: low/high" no DREAM4 -
existem 5 REDES distintas por tamanho. Cada arquivo de timeseries contem
multiplas TRAJETORIAS separadas por linha em branco, com Time reiniciando
em 0.0 a cada bloco. O pipeline parseia isso em lista de trajetorias e
NUNCA constroi pares (t, t+1) atravessando trajetorias diferentes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import yaml

try:
    import community as community_louvain
except ImportError:
    community_louvain = None


@dataclass
class DreamNetworkData:
    size: int
    network: int
    trajectories: list
    graph: nx.DiGraph
    hierarchy: dict
    gene_names: list


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _timeseries_path(root: Path, size: int, network: int) -> Path:
    return (
        root
        / f"DREAM4_InSilico_Size{size}"
        / f"insilico_size{size}_{network}"
        / f"insilico_size{size}_{network}_timeseries.tsv"
    )


def _goldstandard_path(root: Path, size: int, network: int) -> Path:
    return (
        root
        / "DREAM4_Challenge2_GoldStandards"
        / f"Size {size}"
        / f"DREAM4_GoldStandard_InSilico_Size{size}_{network}.tsv"
    )


def load_dream4_trajectories(root: Path, size: int, network: int):
    path = _timeseries_path(root, size, network)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de timeseries nao encontrado em {path}.")

    with open(path, "r") as f:
        lines = f.read().splitlines()

    header_line = lines[0]
    gene_names = [
        c.strip().strip('"') for c in header_line.split("\t")
        if c.strip().strip('"').lower() != "time"
    ]

    trajectories = []
    current_block = []
    for line in lines[1:]:
        if line.strip() == "":
            if current_block:
                trajectories.append(current_block)
                current_block = []
            continue
        values = [float(v) for v in line.split("\t")]
        current_block.append(values[1:])
    if current_block:
        trajectories.append(current_block)

    trajectory_arrays = [np.array(block, dtype=np.float32) for block in trajectories]
    return trajectory_arrays, gene_names


def load_dream4_goldstandard(root: Path, size: int, network: int) -> nx.DiGraph:
    path = _goldstandard_path(root, size, network)
    if not path.exists():
        raise FileNotFoundError(f"Gold standard nao encontrado em {path}.")
    df = pd.read_csv(path, sep="\t", header=None, names=["source", "target", "weight"])
    g = nx.DiGraph()
    for _, row in df.iterrows():
        if row["weight"] > 0:
            g.add_edge(row["source"], row["target"])
    return g


def detect_hierarchy(graph: nx.DiGraph, method: str = "louvain") -> dict:
    undirected = graph.to_undirected()

    if method == "louvain":
        if community_louvain is None:
            raise ImportError("Instale 'python-louvain': pip install python-louvain --break-system-packages")
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


def split_trajectories_by_fraction(trajectories, train_frac, val_frac, test_frac):
    # DECISAO (2026-06-24): trocado de contagem fixa (3/1/1) para fracao
    # uniforme. size=10 (5 trajetorias) continua 3/1/1; size=100
    # (10 trajetorias) passa a usar 6/2/2 em vez de desperdicar metade.
    n = len(trajectories)
    n_train = max(1, int(round(n * train_frac)))
    n_val = max(1, int(round(n * val_frac)))
    n_test = max(1, n - n_train - n_val)
    if n_train + n_val + n_test > n:
        n_test = n - n_train - n_val
    train = trajectories[:n_train]
    val = trajectories[n_train : n_train + n_val]
    test = trajectories[n_train + n_val : n_train + n_val + n_test]
    return train, val, test


def build_dataset(config_path: str = "config/config.yaml"):
    cfg = load_config(config_path)
    root = Path(cfg["data"]["dream4_root"])

    datasets = []
    for size in cfg["data"]["sizes"]:
        for network in cfg["data"]["networks"]:
            trajectories, gene_names = load_dream4_trajectories(root, size, network)
            graph = load_dream4_goldstandard(root, size, network)
            hierarchy = detect_hierarchy(graph, method=cfg["clustering"]["method"])

            datasets.append(
                DreamNetworkData(
                    size=size, network=network, trajectories=trajectories,
                    graph=graph, hierarchy=hierarchy, gene_names=gene_names,
                )
            )

    return datasets


if __name__ == "__main__":
    datasets = build_dataset()
    for d in datasets:
        print(
            f"size={d.size} network={d.network} "
            f"n_trajetorias={len(d.trajectories)} "
            f"shape_cada={d.trajectories[0].shape} "
            f"n_genes={len(d.gene_names)} n_edges_goldstandard={d.graph.number_of_edges()}"
        )
