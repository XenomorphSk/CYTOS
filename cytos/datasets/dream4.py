"""
cytos/datasets/dream4.py

Loader de CONVENIENCIA para o benchmark DREAM4 (opcional - a biblioteca
principal nao depende disso, qualquer grafo+trajetorias do usuario
funciona via cytos.TTNvsGNN diretamente).
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


def _timeseries_path(root, size, network):
    return (
        root / f"DREAM4_InSilico_Size{size}" / f"insilico_size{size}_{network}"
        / f"insilico_size{size}_{network}_timeseries.tsv"
    )


def _goldstandard_path(root, size, network):
    return (
        root / "DREAM4_Challenge2_GoldStandards" / f"Size {size}"
        / f"DREAM4_GoldStandard_InSilico_Size{size}_{network}.tsv"
    )


def load_dream4(size: int, network: int, root: str = "."):
    root_path = Path(root)

    ts_path = _timeseries_path(root_path, size, network)
    if not ts_path.exists():
        raise FileNotFoundError(
            f"Arquivo de timeseries nao encontrado em {ts_path}. "
            "Baixe o DREAM4 via Synapse (syn3049712) e extraia na raiz "
            "do projeto, ou aponte `root` para o diretorio correto."
        )

    with open(ts_path, "r") as f:
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
    trajectories = [np.array(block, dtype=np.float32) for block in trajectories]

    gs_path = _goldstandard_path(root_path, size, network)
    if not gs_path.exists():
        raise FileNotFoundError(f"Gold standard nao encontrado em {gs_path}.")
    df = pd.read_csv(gs_path, sep="\t", header=None, names=["source", "target", "weight"])
    graph = nx.DiGraph()
    for _, row in df.iterrows():
        if row["weight"] > 0:
            graph.add_edge(row["source"], row["target"])

    return graph, trajectories, gene_names
