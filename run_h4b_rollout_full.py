"""
run_h4b_rollout_full.py

H4b (Secao 29 do pre-registro): protocolo completo de rollout multi-passo,
todas as 5 redes, 2 tamanhos, paralelizado.
"""

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import numpy as np
from scipy.stats import wilcoxon

from cytos.datasets import load_dream4
from cytos.data import detect_hierarchy, split_trajectories_by_fraction, trajectories_to_stacked_arrays
from cytos.ttn_model import TTNModel, train_ttn
from cytos.gnn_baseline import GNNBaseline, train_gnn
from cytos.experiment import match_parameter_counts
from cytos.rollout import rollout_ttn, rollout_gnn, rollout_mse_per_step


def _run_one_seed_rollout(args):
    torch.set_num_threads(1)

    seed = args["seed"]
    gene_names = args["gene_names"]
    hierarchy = args["hierarchy"]
    edge_index = args["edge_index"]
    gnn_hidden_dim = args["gnn_hidden_dim"]
    bond_dim = args["bond_dim"]
    n_genes = len(gene_names)

    x_train = torch.tensor(args["x_train"])
    x_train_next = torch.tensor(args["x_train_next"])
    x_val = torch.tensor(args["x_val"])
    x_val_next = torch.tensor(args["x_val_next"])
    test_trajs = [torch.tensor(t) for t in args["test_trajs"]]

    ttn = TTNModel(hierarchy=hierarchy, gene_names=gene_names, bond_dim=bond_dim)
    train_ttn(ttn, x_train, x_train_next, x_val, x_val_next, epochs=200, seed=seed)
    ttn.eval()

    gnn = GNNBaseline(num_nodes=n_genes, hidden_dim=gnn_hidden_dim, num_layers=2)
    train_gnn(
        gnn, x_train.unsqueeze(-1), x_train_next.unsqueeze(-1),
        x_val.unsqueeze(-1), x_val_next.unsqueeze(-1),
        edge_index=edge_index, num_nodes=n_genes, epochs=200, seed=seed,
    )
    gnn.eval()

    ttn_mses, gnn_mses = [], []
    for traj_t in test_trajs:
        x_start = traj_t[0]
        n_steps = traj_t.shape[0] - 1
        true_future = traj_t[1:]

        ttn_traj = rollout_ttn(ttn, x_start, n_steps)
        gnn_traj = rollout_gnn(gnn, x_start, edge_index, n_genes, n_steps)

        ttn_mses.append(rollout_mse_per_step(ttn_traj, true_future).mean())
        gnn_mses.append(rollout_mse_per_step(gnn_traj, true_future).mean())

    return {"seed": seed, "ttn_mse": float(np.mean(ttn_mses)), "gnn_mse": float(np.mean(gnn_mses))}


def run_config(size, network, seeds, n_workers):
    graph, trajectories, gene_names = load_dream4(size=size, network=network, root=".")
    hierarchy = detect_hierarchy(graph, method="louvain")
    n_genes = len(gene_names)

    train_trajs, val_trajs, test_trajs = split_trajectories_by_fraction(trajectories)
    x_train, x_train_next = trajectories_to_stacked_arrays(train_trajs)
    x_val, x_val_next = trajectories_to_stacked_arrays(val_trajs)

    node_to_idx = {n: i for i, n in enumerate(gene_names)}
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in graph.edges() if u in node_to_idx and v in node_to_idx]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    gnn_hidden_dim, bond_dim, _, _ = match_parameter_counts(hierarchy, gene_names, num_nodes=n_genes)

    seed_args = [
        {
            "seed": seed, "gene_names": gene_names, "hierarchy": hierarchy,
            "edge_index": edge_index, "gnn_hidden_dim": gnn_hidden_dim, "bond_dim": bond_dim,
            "x_train": x_train, "x_train_next": x_train_next,
            "x_val": x_val, "x_val_next": x_val_next,
            "test_trajs": [np.array(t) for t in test_trajs],
        }
        for seed in seeds
    ]

    results = []
    with ProcessPoolExecutor(max_workers=n_workers, max_tasks_per_child=4) as executor:
        futures = {executor.submit(_run_one_seed_rollout, a): a["seed"] for a in seed_args}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"\nAVISO seed falhou: {e}")

    return results


if __name__ == "__main__":
    seeds = list(range(20))
    n_workers = min(len(seeds), os.cpu_count() or 4)
    configs = [(10, n) for n in range(1, 6)] + [(100, n) for n in range(1, 6)]
    confirmatory = {1, 2}

    all_results = {}
    start = time.time()

    for size, network in configs:
        label = "CONFIRMATORIO" if network in confirmatory else "EXPLORATORIO"
        print(f"\n=== {size} genes / rede {network} [{label}] ===")
        results = run_config(size, network, seeds, n_workers)

        ttn_m = np.array([r["ttn_mse"] for r in results])
        gnn_m = np.array([r["gnn_mse"] for r in results])
        _, p = wilcoxon(ttn_m, gnn_m)
        h4b_pass = bool(p < 0.05 and ttn_m.mean() < gnn_m.mean())

        all_results[(size, network)] = {
            "label": label, "ttn_mean": float(ttn_m.mean()), "gnn_mean": float(gnn_m.mean()),
            "p": float(p), "pass": h4b_pass,
        }
        print(f"TTN={ttn_m.mean():.4e}, GNN={gnn_m.mean():.4e}, p={p:.4e}, {'PASSOU' if h4b_pass else 'FALHOU'}")
        print(f"Tempo decorrido: {time.time()-start:.1f}s")

    print("\n\n=== RESUMO FINAL ===")
    n_pass_100 = sum(1 for (s, n), r in all_results.items() if s == 100 and r["pass"])
    n_pass_10 = sum(1 for (s, n), r in all_results.items() if s == 10 and r["pass"])
    print(f"100 genes: {n_pass_100}/5 configs passaram")
    print(f"10 genes: {n_pass_10}/5 configs passaram (sem criterio de sucesso/falha, descritivo)")

    n_pass_100_confirm = sum(1 for (s, n), r in all_results.items() if s == 100 and n in confirmatory and r["pass"])
    n_pass_100_explor = sum(1 for (s, n), r in all_results.items() if s == 100 and n not in confirmatory and r["pass"])
    h4b_simples_pass = n_pass_100_confirm >= 1 and n_pass_100_explor >= 2
    print(f"\nH4b-simples (criterio pre-registrado): {'PASSOU' if h4b_simples_pass else 'FALHOU'}")
    print(f"  (confirmatorias 100g passando: {n_pass_100_confirm}/2, exploratorias 100g passando: {n_pass_100_explor}/3)")
