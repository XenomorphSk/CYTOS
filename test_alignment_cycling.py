import pandas as pd
import numpy as np
from cytos.datasets import (
    load_string_network,
    load_alias_to_preferred_name,
    rename_genes_via_aliases,
    align_graph_and_expression_genes,
)
from cytos.datasets.expression_timeseries import load_expression_trajectory

if __name__ == "__main__":
    # carrega a lista dos 800 genes ciclo-regulados
    orf800 = pd.read_csv("orf800_cycling_genes.csv")["orf"].tolist()
    print(f"Genes ciclo-regulados (orf800): {len(orf800)}")

    # carrega grafo STRING
    graph = load_string_network(
        links_path="4932.protein.links.v12.0.txt",
        info_path="4932.protein.info.v12.0.txt",
        confidence_threshold=900,
    )

    # carrega expressao (todos os 4489 genes completos)
    expr, gene_names = load_expression_trajectory("spellman_alpha_complete_genes.csv")

    # filtra a expressao pros 800 ciclo-regulados
    orf800_set = set(orf800)
    cycling_idx = [i for i, g in enumerate(gene_names) if g in orf800_set]
    expr_cycling = expr[:, cycling_idx]
    gene_names_cycling = [gene_names[i] for i in cycling_idx]
    print(f"Genes ciclo-regulados com dados de expressao completos: {len(gene_names_cycling)}")

    # mapeia nomes sistematicos -> nomes padrao do STRING via aliases
    alias_map = load_alias_to_preferred_name(
        info_path="4932.protein.info.v12.0.txt",
        aliases_path="4932.protein.aliases.v12.0.txt",
    )
    gene_names_mapped = rename_genes_via_aliases(gene_names_cycling, alias_map)

    # alinha com o grafo STRING
    subgraph, common_genes = align_graph_and_expression_genes(graph, gene_names_mapped)
    print(f"\nGenes em comum com STRING (pos alias): {len(common_genes)}")
    print(f"Subgrafo final: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")
    print(f"Exemplos de genes finais: {common_genes[:10]}")

    # salva a expressao final so com esses genes, na ordem certa
    common_set = set(common_genes)
    final_idx = [i for i, g in enumerate(gene_names_mapped) if g in common_set]
    expr_final = expr_cycling[:, final_idx]
    gene_names_final = [gene_names_mapped[i] for i in final_idx]

    df_final = pd.DataFrame(expr_final, columns=gene_names_final)
    df_final.to_csv("spellman_cycling_string_aligned.csv", index=False)
    print(f"\nSalvo: spellman_cycling_string_aligned.csv ({expr_final.shape[0]} timepoints x {expr_final.shape[1]} genes)")
