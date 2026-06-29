from cytos.datasets import load_string_network, align_graph_and_expression_genes
from cytos.datasets.expression_timeseries import load_expression_trajectory

if __name__ == "__main__":
    graph = load_string_network(
        links_path="4932.protein.links.v12.0.txt",
        info_path="4932.protein.info.v12.0.txt",
        confidence_threshold=900,
    )
    print(f"Grafo STRING: {graph.number_of_nodes()} genes")

    expr, gene_names = load_expression_trajectory(
        "spellman_alpha_complete_genes.csv",
    )
    print(f"Expressao: {len(gene_names)} genes, formato dos primeiros nomes: {gene_names[:5]}")

    subgraph, common_genes = align_graph_and_expression_genes(graph, gene_names)
    print(f"\nGenes em comum (match direto por nome): {len(common_genes)}")
    print(f"Exemplos: {common_genes[:10]}")
