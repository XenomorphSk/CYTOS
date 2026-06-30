"""
cytos/kegg_hierarchy.py

Constroi hierarquia para a TTN a partir de KEGG pathways, em vez de
clustering estatistico (Louvain). Secao 31 do pre-registro (H5).
"""

from __future__ import annotations


def load_kegg_gene_pathway_map(path):
    gene_to_pathways = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            gene_raw, pathway_raw = line.split("\t")
            gene = gene_raw.split(":")[-1]
            pathway = pathway_raw.split(":")[-1]
            gene_to_pathways.setdefault(gene, []).append(pathway)
    return gene_to_pathways


def build_kegg_hierarchy(gene_names, gene_to_pathways):
    partition = {}
    for gene in gene_names:
        pathways = gene_to_pathways.get(gene, [])
        if pathways:
            partition[gene] = pathways[0]
        else:
            partition[gene] = "sem_pathway"

    return {"level_0": partition}
