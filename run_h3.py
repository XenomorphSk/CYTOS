from cytos.datasets import load_dream4
from cytos.topology_experiment import run_topology_experiment

if __name__ == "__main__":
    configs = [(10, 1), (10, 2), (100, 1)]

    for size, network in configs:
        print(f"\n=== H3: {size} genes / rede {network} ===")
        graph, trajectories, gene_names = load_dream4(size=size, network=network, root=".")
        result = run_topology_experiment(graph, trajectories, gene_names, seeds=20)
        print(f"TTN: AUPR={result['ttn_aupr_mean']:.4f}, AUROC={result['ttn_auroc_mean']:.4f}")
        print(f"MLP: AUPR={result['mlp_aupr_mean']:.4f}, AUROC={result['mlp_auroc_mean']:.4f}")
        print(f"p(AUPR)={result['p_aupr']:.4e}, p(AUROC)={result['p_auroc']:.4e}")
        print(f"H3 AUPR: {'PASSOU' if result['h3_aupr_pass'] else 'FALHOU'}, H3 AUROC: {'PASSOU' if result['h3_auroc_pass'] else 'FALHOU'}")
