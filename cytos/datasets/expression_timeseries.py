"""
cytos/datasets/expression_timeseries.py

Carregador GENERICO de dados de expressao genica temporal REAL (GEO,
ArrayExpress, etc.), para combinar com uma rede real (string_db) e usar
com cytos.TTNvsGNN.

Formato esperado do CSV: linhas = timepoints, colunas = genes. Multiplas
replicas/condicoes = um CSV separado por trajetoria.

IMPORTANTE: nenhuma limpeza automatica de dados reais e feita aqui
(normalizacao, imputacao) - isso e responsabilidade do usuario, para
nao mascarar silenciosamente problemas de qualidade dos dados.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_expression_trajectory(csv_path, gene_columns=None, time_column=None):
    df = pd.read_csv(csv_path)

    if time_column is not None and time_column in df.columns:
        df = df.drop(columns=[time_column])

    if gene_columns is not None:
        missing = [g for g in gene_columns if g not in df.columns]
        if missing:
            raise ValueError(f"Genes nao encontrados no CSV: {missing}")
        df = df[gene_columns]

    if df.isnull().any().any():
        raise ValueError(
            "CSV contem valores faltantes (NaN). Trate isso explicitamente "
            "antes de carregar - esta funcao nao faz isso automaticamente."
        )

    return df.values.astype(np.float32), list(df.columns)


def load_expression_trajectories(csv_paths, gene_columns=None, time_column=None):
    trajectories = []
    gene_names_ref = None

    for path in csv_paths:
        traj, gene_names = load_expression_trajectory(path, gene_columns, time_column)
        if gene_names_ref is None:
            gene_names_ref = gene_names
        elif gene_names != gene_names_ref:
            raise ValueError(
                f"Arquivo {path} tem genes/ordem diferente dos anteriores."
            )
        trajectories.append(traj)

    return trajectories, gene_names_ref


def align_graph_and_expression_genes(graph, gene_names):
    graph_genes = set(graph.nodes())
    common_genes = [g for g in gene_names if g in graph_genes]

    if len(common_genes) < 3:
        raise ValueError(
            f"Apenas {len(common_genes)} genes em comum entre o grafo e os "
            f"dados de expressao - verifique se os identificadores batem."
        )

    subgraph = graph.subgraph(common_genes).copy()
    return subgraph, common_genes
