"""
run_h6_factorial.py

H6 (Secao 33): desenho fatorial 2x2 (ruido x tamanho de amostra),
hierarquia VERDADEIRA por construcao.
"""

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import numpy as np
from scipy.stats import wilcoxon

from cytos.synthetic_hierarchy import generate_dataset
from cytos.data import trajectories_to_stacked_arrays
from cytos.experiment import match_parameter_counts, _run_one_seed


def run_condition(noise_std, n_train_trajectories, n_steps_per_traj, seeds, n_workers):
    data = generate_dataset(
        n_genes=100, comm_size=10, noise_std=noise_std,
        n_train_trajectories=n_train_trajectories, n_steps_per_traj=n_steps_per_traj,
    )
    gene_names = data["gene_names"]
    hierarchy = data["hierarchy"]
    n_genes = len(gene_names)

    x_train, x_train_next = trajectories_to_stacked_arrays(data["train_trajectories"])
    x_val, x_val_next = trajectories_to_stacked_arrays([data["val_trajectory"]])
    x_test, x_test_next = trajectories_to_stacked_arrays([data["test_trajectory"]])

    partition = hierarchy["level_0"]
    node_to_idx = {g: i for i, g in enumerate(gene_names)}
    edges = [
        (node_to_idx[gi], node_to_idx[gj])
        for gi in gene_names for gj in gene_names
        if gi != gj and partition[gi] == partition[gj]
    ]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    gnn_hidden_dim, bond_dim, _, _ = match_parameter_counts(hierarchy, gene_names, num_nodes=n_genes)

    train_cfg = {"gnn_architecture": "GCN", "gnn_num_layers": 2, "lr": 0.001,
                 "weight_decay": 1e-5, "epochs": 200, "patience": 15, "batch_size": 16}

    seed_args = [
        {"seed": s, "gene_names": gene_names, "hierarchy": hierarchy,
         "edge_index": edge_index, "gnn_hidden_dim": gnn_hidden_dim, "bond_dim": bond_dim,
         "train_cfg": train_cfg,
         "x_train": x_train, "x_train_next": x_train_next,
         "x_val": x_val, "x_val_next": x_val_next,
         "x_test": x_test, "x_test_next": x_test_next}
        for s in seeds
    ]

    results = []
    with ProcessPoolExecutor(max_workers=n_workers, max_tasks_per_child=4) as executor:
        futures = {executor.submit(_run_one_seed, a): a["seed"] for a in seed_args}
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                print(f"\nAVISO seed falhou: {e}")
    return results


if __name__ == "__main__":
    seeds = list(range(20))
    n_workers = min(len(seeds), os.cpu_count() or 4)

    conditions = [
        ("ruido_baixo_amostra_grande", 0.01, 10, 20),
        ("ruido_baixo_amostra_pequena", 0.01, 1, 10),
        ("ruido_alto_amostra_grande", 0.3, 10, 20),
        ("ruido_alto_amostra_pequena", 0.3, 1, 10),
    ]

    for name, noise, n_traj, n_steps in conditions:
        n_pairs = n_traj * n_steps
        print(f"\n=== {name} (noise_std={noise}, n_pares_treino={n_pairs}) ===")
        start = time.time()
        results = run_condition(noise, n_traj, n_steps, seeds, n_workers)

        ttn_mses = [r["ttn_mse_per_param"] for r in results]
        gnn_mses = [r["gnn_mse_per_param"] for r in results]
        _, p = wilcoxon(ttn_mses, gnn_mses)
        ttn_wins = np.mean(ttn_mses) < np.mean(gnn_mses)

        print(f"TTN={np.mean(ttn_mses):.4e}, GNN={np.mean(gnn_mses):.4e}, p={p:.4e}")
        print(f"TTN {'venceu' if ttn_wins else 'perdeu'} ({'significativo' if p<0.05 else 'nao significativo'})")
        print(f"Tempo: {time.time()-start:.1f}s")
