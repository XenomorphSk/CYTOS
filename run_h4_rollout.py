"""
run_h4_rollout.py

Piloto exploratorio H4 (Secao 27 do pre-registro): treina TTN e GNN
exatamente como em H1/H1b (1 passo), depois testa ambos em rollout
multi-passo nas trajetorias de teste, sem retreinar.
"""

import torch
import numpy as np
from scipy.stats import wilcoxon

from cytos.datasets import load_dream4
from cytos.data import detect_hierarchy, split_trajectories_by_fraction, trajectories_to_stacked_arrays
from cytos.ttn_model import TTNModel, train_ttn
from cytos.gnn_baseline import GNNBaseline, train_gnn
from cytos.experiment import match_parameter_counts
from cytos.rollout import rollout_ttn, rollout_gnn, rollout_mse_per_step


def run_config(size, network, seeds):
    graph, trajectories, gene_names = load_dream4(size=size, network=network, root=".")
    hierarchy = detect_hierarchy(graph, method="louvain")
    n_genes = len(gene_names)

    train_trajs, val_trajs, test_trajs = split_trajectories_by_fraction(trajectories)
    x_train, x_train_next = trajectories_to_stacked_arrays(train_trajs)
    x_val, x_val_next = trajectories_to_stacked_arrays(val_trajs)

    node_to_idx = {n: i for i, n in enumerate(gene_names)}
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in graph.edges() if u in node_to_idx and v in node_to_idx]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    gnn_hidden_dim, bond_dim, gnn_params, ttn_params = match_parameter_counts(
        hierarchy, gene_names, num_nodes=n_genes
    )

    x_train_t, x_train_next_t = torch.tensor(x_train), torch.tensor(x_train_next)
    x_val_t, x_val_next_t = torch.tensor(x_val), torch.tensor(x_val_next)

    ttn_rollout_mses = []
    gnn_rollout_mses = []
    ttn_curves = []
    gnn_curves = []

    for seed in seeds:
        ttn = TTNModel(hierarchy=hierarchy, gene_names=gene_names, bond_dim=bond_dim)
        train_ttn(ttn, x_train_t, x_train_next_t, x_val_t, x_val_next_t, epochs=200, seed=seed)
        ttn.eval()

        gnn = GNNBaseline(num_nodes=n_genes, hidden_dim=gnn_hidden_dim, num_layers=2)
        train_gnn(
            gnn, x_train_t.unsqueeze(-1), x_train_next_t.unsqueeze(-1),
            x_val_t.unsqueeze(-1), x_val_next_t.unsqueeze(-1),
            edge_index=edge_index, num_nodes=n_genes, epochs=200, seed=seed,
        )
        gnn.eval()

        for traj in test_trajs:
            traj_t = torch.tensor(traj)
            x_start = traj_t[0]
            n_steps = traj_t.shape[0] - 1
            true_future = traj_t[1:]

            ttn_traj = rollout_ttn(ttn, x_start, n_steps)
            gnn_traj = rollout_gnn(gnn, x_start, edge_index, n_genes, n_steps)

            ttn_curve = rollout_mse_per_step(ttn_traj, true_future)
            gnn_curve = rollout_mse_per_step(gnn_traj, true_future)

            ttn_rollout_mses.append(ttn_curve.mean())
            gnn_rollout_mses.append(gnn_curve.mean())
            ttn_curves.append(ttn_curve)
            gnn_curves.append(gnn_curve)

    return {
        "ttn_rollout_mses": ttn_rollout_mses,
        "gnn_rollout_mses": gnn_rollout_mses,
        "ttn_curves": ttn_curves,
        "gnn_curves": gnn_curves,
    }


if __name__ == "__main__":
    seeds = list(range(20))
    configs = [(10, 1), (100, 1)]

    for size, network in configs:
        print(f"\n=== H4: {size} genes / rede {network} ===")
        result = run_config(size, network, seeds)

        ttn_m = np.array(result["ttn_rollout_mses"])
        gnn_m = np.array(result["gnn_rollout_mses"])

        _, p = wilcoxon(ttn_m, gnn_m)
        h4_pass = bool(p < 0.05 and ttn_m.mean() < gnn_m.mean())

        print(f"TTN rollout MSE medio: {ttn_m.mean():.4e}")
        print(f"GNN rollout MSE medio: {gnn_m.mean():.4e}")
        print(f"p={p:.4e}, H4: {'PASSOU' if h4_pass else 'FALHOU'}")

        min_len = min(len(c) for c in result["ttn_curves"])
        ttn_curve_avg = np.mean([c[:min_len] for c in result["ttn_curves"]], axis=0)
        gnn_curve_avg = np.mean([c[:min_len] for c in result["gnn_curves"]], axis=0)

        print(f"\nCurva de erro por passo (k=1..{min_len}):")
        print(f"  TTN: {np.array2string(ttn_curve_avg, precision=4)}")
        print(f"  GNN: {np.array2string(gnn_curve_avg, precision=4)}")
