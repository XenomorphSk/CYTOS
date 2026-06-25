"""
src/data/synthetic_dream4.py

Gera dados sinteticos NO FORMATO do DREAM4, para permitir desenvolvimento
e teste do pipeline ANTES de ter acesso aos dados reais do Synapse.

IMPORTANTE: isso e so para destravar desenvolvimento/debug. Os resultados do
pre-registro so valem com o DREAM4 real.

Uso:
    python -m src.data.synthetic_dream4 --size 10
    python -m src.data.synthetic_dream4 --size 100
"""

from __future__ import annotations

import argparse
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


def generate_synthetic_network(size: int, seed: int = 0) -> nx.DiGraph:
    """Gera um grafo direcionado scale-free simples como proxy de rede regulatoria."""
    g = nx.scale_free_graph(size, alpha=0.4, beta=0.5, gamma=0.1, seed=seed)
    g = nx.DiGraph(g)
    g.remove_edges_from(nx.selfloop_edges(g))
    return g


def simulate_timeseries(
    graph: nx.DiGraph, size: int, n_timepoints: int = 200, noise_level: str = "low", seed: int = 0
) -> np.ndarray:
    """
    Simulacao tipo modelo linear com decaimento + acoplamento via matriz de
    adjacencia + ruido gaussiano.

    BUG CORRIGIDO (2026-06-24): a versao anterior so limitava x por baixo,
    sem limite superior. Em grafos com ciclos, o acoplamento podia criar
    realimentacao positiva e a serie explodia exponencialmente (valores
    chegando a ~3e7 em 200 passos). Correcoes:
    1. Normalizacao espectral da matriz de acoplamento (raio espectral < 1).
    2. Clip simetrico (0 a 1).
    """
    rng = np.random.default_rng(seed)
    adj = nx.to_numpy_array(graph)

    eigenvalues = np.linalg.eigvals(adj)
    spectral_radius = np.max(np.abs(eigenvalues)) if len(eigenvalues) > 0 else 0.0
    if spectral_radius > 1e-8:
        adj = adj * (0.5 / spectral_radius)

    noise_std = 0.02 if noise_level == "low" else 0.10
    decay = 0.9

    x = rng.uniform(0.2, 0.8, size=size)
    series = [x.copy()]
    for _ in range(n_timepoints - 1):
        coupling = adj.T @ x
        x = decay * x + 0.3 * coupling + rng.normal(0, noise_std, size=size)
        x = np.clip(x, 0.0, 1.0)
        series.append(x.copy())

    return np.array(series, dtype=np.float32)


def write_dream4_files(raw_dir: Path, size: int, seed: int = 0) -> None:
    out_dir = raw_dir / "dream4"
    out_dir.mkdir(parents=True, exist_ok=True)

    graph = generate_synthetic_network(size, seed=seed)
    gene_names = [f"G{i+1}" for i in range(size)]
    relabel = {i: gene_names[i] for i in range(size)}
    graph = nx.relabel_nodes(graph, relabel)

    edges_df = pd.DataFrame(
        [(u, v, 1) for u, v in graph.edges()], columns=["source", "target", "weight"]
    )
    edges_df.to_csv(
        out_dir / f"insilico_size{size}_goldstandard.tsv", sep="\t", header=False, index=False
    )

    for noise_level in ["low", "high"]:
        ts = simulate_timeseries(graph, size, noise_level=noise_level, seed=seed)
        df = pd.DataFrame(ts, columns=gene_names)
        df.insert(0, "Time", np.arange(len(ts)))
        suffix = "" if noise_level == "low" else f"_{noise_level}"
        df.to_csv(
            out_dir / f"insilico_size{size}_timeseries{suffix}.tsv", sep="\t", index=False
        )

    print(f"Dados sinteticos gerados em {out_dir} para size={size}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--raw-dir", type=str, default="data/raw")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    write_dream4_files(Path(args.raw_dir), args.size, seed=args.seed)
