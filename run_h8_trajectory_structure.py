"""
run_h8_trajectory_structure.py

H8 (Secao 37): isola se a estrutura de trajetoria (unica contigua vs
multiplas independentes) explica o gap em dados reais.
"""

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import numpy as np
from scipy.stats import wilcoxon

from cytos.synthetic_hierarchy import build_true_hierarchy, build_coupling_matrix, generate_trajectory
from cytos.data import trajectories_to_stacked_arrays
from cytos.experiment import match_parameter_counts, _run_one_seed


def run_multi_trajectory_condition(seeds, n_workers):
    n_genes, comm_size = 100, 10
    gene_names, hierarchy = build_true_hierarchy(n_genes, comm_size)
    partition = hierarchy["level_0"]
    A = build_coupling_matrix(gene_names, partition, seed=0)

    train_trajs = [generate_trajectory(A, n_genes, n_steps=20, noise_std=0.01, seed=i) for i in range(10)]
    val_traj = generate_trajectory(A, n_genes, n_steps=20, noise_std=0.01, seed=9000)
    test_traj = generate_trajectory(A, n_genes, n_steps=20, noise_std=0.01, seed=9001)

    x_train, x_train_next = trajectories_to_stacked_arrays(train_trajs)
    x_val, x_val_next = trajectories_to_stacked_arrays([val_traj])
    x_test, x_test_next = trajectories_to_stacked_arrays([test_traj])

    return _run_all_seeds(gene_names, hierarchy, x_train, x_train_next, x_val, x_val_next, x_test, x_test_next, seeds, n_workers)


def run_single_trajectory_condition(seeds, n_workers):
    n_genes, comm_size = 100, 10
    gene_names, hierarchy = build_true_hierarchy(n_genes, comm_size)
    partition = hierarchy["level_0"]
    A = build_coupling_matrix(gene_names, partition, seed=0)

    single_traj = generate_trajectory(A, n_genes, n_steps=334, noise_std=0.01, seed=0)
    n = single_traj.shape[0]
    n_train = int(n * 0.6)
    n_val = int(n * 0.2)
    train_seg = single_traj[:n_train]
    val_seg = single_traj[n_train:n_train + n_val]
    test_seg = single_traj[n_train + n_val:]

    x_train, x_train_next = trajectories_to_stacked_arrays([train_seg])
    x_val, x_val_next = trajectories_to_stacked_arrays([val_seg])
    x_test, x_test_next = trajectories_to_stacked_arrays([test_seg])

    return _run_all_seeds(gene_names, hierarchy, x_train, x_train_next, x_val, x_val_next, x_test, x_test_next, seeds, n_workers)


def _run_all_seeds(gene_names, hierarchy, x_train, x_train_next, x_val, x_val_next, x_test, x_test_next, seeds, n_workers):
    n_genes = len(gene_names)
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

    print("=== Condicao: MULTIPLAS trajetorias independentes (replica H6/DREAM4) ===")
    start = time.time()
    results_multi = run_multi_trajectory_condition(seeds, n_workers)
    ttn_m = np.array([r["ttn_mse_per_param"] for r in results_multi])
    gnn_m = np.array([r["gnn_mse_per_param"] for r in results_multi])
    _, p_multi = wilcoxon(ttn_m, gnn_m)
    print(f"TTN={ttn_m.mean():.4e}, GNN={gnn_m.mean():.4e}, p={p_multi:.4e}, tempo={time.time()-start:.1f}s")
    multi_ttn_wins = bool(p_multi < 0.05 and ttn_m.mean() < gnn_m.mean())

    print("\n=== Condicao: TRAJETORIA UNICA contigua (replica Spellman/H5/H7) ===")
    start = time.time()
    results_single = run_single_trajectory_condition(seeds, n_workers)
    ttn_s = np.array([r["ttn_mse_per_param"] for r in results_single])
    gnn_s = np.array([r["gnn_mse_per_param"] for r in results_single])
    _, p_single = wilcoxon(ttn_s, gnn_s)
    print(f"TTN={ttn_s.mean():.4e}, GNN={gnn_s.mean():.4e}, p={p_single:.4e}, tempo={time.time()-start:.1f}s")
    single_ttn_wins = bool(p_single < 0.05 and ttn_s.mean() < gnn_s.mean())

    print(f"\n=== Resultado H8 ===")
    print(f"Multiplas trajetorias: TTN {'venceu' if multi_ttn_wins else 'NAO venceu'}")
    print(f"Trajetoria unica: TTN {'venceu' if single_ttn_wins else 'NAO venceu'}")
    h8_confirmed = multi_ttn_wins and not single_ttn_wins
    print(f"H8 {'CONFIRMADO' if h8_confirmed else 'NAO confirmado'} (estrutura de trajetoria explica o gap: {h8_confirmed})")
