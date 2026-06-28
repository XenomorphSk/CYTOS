from cytos import TTNvsGNN
from cytos.datasets import load_dream4

if __name__ == "__main__":
    graph, trajectories, gene_names = load_dream4(size=10, network=1, root=".")
    exp = TTNvsGNN(graph=graph, trajectories=trajectories, gene_names=gene_names)
    result = exp.run(seeds=20)
    print(result.summary())
