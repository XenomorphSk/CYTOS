from cytos.datasets import (
    load_string_network,
    load_alias_to_preferred_name,
    rename_genes_via_aliases,
    align_graph_and_expression_genes,
)
from cytos.datasets.expression_timeseries import load_expression_trajectory

if __name__ == "__main__":
    graph = load_string_network(
        links_path="4932.protein.links.v12.0.txt",
        info_path="4932.protein.info.v12.0.txt",
        confidence_threshold=900,
    )
    print(f"Grafo STRING: {graph.number_of_nodes()} genes")

    expr, gene_names = load_expression_trajectory("spellman_alpha_complete_genes.csv")
    print(f"Expressao: {len(gene_names)} genes, exemplos: {gene_names[:5]}")

    alias_map = load_alias_to_preferred_name(
        info_path="4932.protein.info.v12.0.txt",
        aliases_path="4932.protein.aliases.v12.0.txt",
    )
    gene_names_mapped = rename_genes_via_aliases(gene_names, alias_map)
    print(f"\nExemplos apos mapeamento via aliases:")
    for orig, mapped in zip(gene_names[:8], gene_names_mapped[:8]):
        changed = " <-- MUDOU" if orig != mapped else ""
        print(f"  {orig} -> {mapped}{changed}")

    subgraph, common_genes = align_graph_and_expression_genes(graph, gene_names_mapped)
    print(f"\nGenes em comum (pos alias): {len(common_genes)}")
    print(f"Subgrafo: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")
