from cytos.datasets import load_string_network

if __name__ == "__main__":
    graph = load_string_network(
        links_path="4932.protein.links.v12.0.txt",
        info_path="4932.protein.info.v12.0.txt",
        confidence_threshold=900,  # comeca alto (confianca maxima) pra rede menor/mais manejavel
    )
    print(f"Grafo: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print(f"Exemplos de genes: {list(graph.nodes())[:10]}")
