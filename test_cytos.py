from cytos import TTNvsGNN
from cytos.datasets import load_dream4

if __name__ == "__main__":
    graph, trajectories, gene_names = load_dream4(size=10, network=1, root=".")
    print(f"Grafo: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    print(f"Trajetorias: {len(trajectories)}, shape de cada: {trajectories[0].shape}")

    exp = TTNvsGNN(graph=graph, trajectories=trajectories, gene_names=gene_names)
    print(f"hidden_dim={exp.gnn_hidden_dim}, bond_dim={exp.bond_dim}, gnn_params={exp.gnn_params}, ttn_params={exp.ttn_params}")

    result = exp.run(seeds=5)
    print(result.summary())
