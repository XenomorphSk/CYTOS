import torch
from cytos.datasets import load_dream4
from cytos.data import split_trajectories_by_fraction, trajectories_to_stacked_arrays
from cytos.ttn_model import TTNModel, train_ttn
from cytos.topology_inference import (
    arbitrary_hierarchy, MLPBaseline, train_mlp,
    edge_scores_via_perturbation, evaluate_against_gold_standard,
)

if __name__ == "__main__":
    graph, trajectories, gene_names = load_dream4(size=10, network=1, root=".")
    n_genes = len(gene_names)

    train_trajs, val_trajs, test_trajs = split_trajectories_by_fraction(trajectories)
    x_train, x_train_next = trajectories_to_stacked_arrays(train_trajs)
    x_val, x_val_next = trajectories_to_stacked_arrays(val_trajs)
    x_test, _ = trajectories_to_stacked_arrays(test_trajs)

    x_train_t, x_train_next_t = torch.tensor(x_train), torch.tensor(x_train_next)
    x_val_t, x_val_next_t = torch.tensor(x_val), torch.tensor(x_val_next)
    x_test_t = torch.tensor(x_test)

    # TTN com hierarquia ARBITRARIA (sem usar o grafo verdadeiro)
    hierarchy = arbitrary_hierarchy(gene_names)
    print("Hierarquia arbitraria:", hierarchy)

    ttn = TTNModel(hierarchy=hierarchy, gene_names=gene_names, bond_dim=4)
    train_ttn(ttn, x_train_t, x_train_next_t, x_val_t, x_val_next_t, epochs=200, seed=0)
    ttn.eval()
    print(f"TTN params: {ttn.count_parameters()}")

    ttn_scores = edge_scores_via_perturbation(ttn, n_genes, x_test_t)
    ttn_eval = evaluate_against_gold_standard(ttn_scores, graph, gene_names)
    print(f"TTN: AUPR={ttn_eval['aupr']:.4f}, AUROC={ttn_eval['auroc']:.4f} (n_true_edges={ttn_eval['n_true_edges']})")

    # MLP baseline (parametros aproximadamente casados)
    mlp = MLPBaseline(n_genes=n_genes, hidden_dim=8, num_layers=2)
    print(f"MLP params: {mlp.count_parameters()}")
    train_mlp(mlp, x_train_t, x_train_next_t, x_val_t, x_val_next_t, epochs=200, seed=0)
    mlp.eval()

    mlp_scores = edge_scores_via_perturbation(mlp, n_genes, x_test_t)
    mlp_eval = evaluate_against_gold_standard(mlp_scores, graph, gene_names)
    print(f"MLP: AUPR={mlp_eval['aupr']:.4f}, AUROC={mlp_eval['auroc']:.4f}")
