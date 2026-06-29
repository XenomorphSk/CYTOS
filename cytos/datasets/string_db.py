"""
cytos/datasets/string_db.py

Carregador de redes de interacao proteina-proteina REAIS do STRING
database (https://string-db.org). So tem TOPOLOGIA - nao tem dinamica
temporal embutida. Combine com cytos.datasets.expression_timeseries.

Como obter os arquivos:
1. https://string-db.org/cgi/download
2. Escolher organismo (ex: 9606 = Homo sapiens, 4932 = S. cerevisiae)
3. Baixar:
   - {species}.protein.links.v12.0.txt.gz
   - {species}.protein.info.v12.0.txt.gz
4. Descomprimir (gunzip) antes de usar.
"""

from __future__ import annotations

import networkx as nx
import pandas as pd


def load_string_id_mapping(info_path: str) -> dict:
    df = pd.read_csv(info_path, sep="\t")
    id_col = "#string_protein_id" if "#string_protein_id" in df.columns else df.columns[0]
    name_col = "preferred_name" if "preferred_name" in df.columns else df.columns[1]
    return dict(zip(df[id_col], df[name_col]))


def load_string_network(links_path, info_path, confidence_threshold=700, gene_whitelist=None):
    id_to_symbol = load_string_id_mapping(info_path)

    df = pd.read_csv(links_path, sep=" ")
    score_col = "combined_score" if "combined_score" in df.columns else df.columns[-1]
    df = df[df[score_col] >= confidence_threshold]

    graph = nx.DiGraph()
    n_skipped_unmapped = 0
    for _, row in df.iterrows():
        p1, p2 = row.iloc[0], row.iloc[1]
        g1, g2 = id_to_symbol.get(p1), id_to_symbol.get(p2)
        if g1 is None or g2 is None:
            n_skipped_unmapped += 1
            continue
        if gene_whitelist is not None and (g1 not in gene_whitelist or g2 not in gene_whitelist):
            continue
        graph.add_edge(g1, g2)
        graph.add_edge(g2, g1)

    if n_skipped_unmapped > 0:
        print(f"AVISO: {n_skipped_unmapped} interacoes puladas por falta de mapeamento ID->simbolo.")

    return graph
