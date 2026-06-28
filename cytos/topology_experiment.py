"""
cytos/topology_experiment.py

Orquestra o experimento completo da Fase 3 (H3): casa parametros entre
TTN (hierarquia arbitraria) e MLP, treina ambos com multiplas seeds,
pontua arestas via sensibilidade a perturbacao, avalia contra o gold
standard (AUPR/AUROC), testa diferenca via Wilcoxon pareado.
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from scipy.stats import wilcoxon

from cytos.data import split_trajectories_by_fraction, trajectories_to_stacked_arrays
from cytos.ttn_model import TTNModel, train_ttn
from cytos.topology_inference import (
    arbitrary_hierarchy, MLPBaseline, train_mlp,
    edge_scores_via_perturbation, evaluate_against_gold_standard,
)


def match_ttn_mlp_params(gene_names, num_layers=2, tolerance=0.10):
    hierarchy = arbitrary_hierarchy(gene_names)
    n_genes = len(gene_names)
    bond_dim_candidates = list(range(2, 33))
    hidden_dim_candidates = [2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 256]

    best = None
    for bond_dim in bond_dim_candidates:
        ttn_probe = TTNModel(hierarchy=hierarchy, gene_names=gene_names, bond_dim=bond_dim)
        ttn_params = ttn_probe.count_parameters()
        for hidden_dim in hidden_dim_candidates:
            mlp_probe = MLPBaseline(n_genes=n_genes, hidden_dim=hidden_dim, num_layers=num_layers)
            mlp_params = mlp_probe.count_parameters()
            diff = abs(ttn_params - mlp_params) / max(mlp_params, 1)
            if best is None or diff < best[0]:
                best = (diff, bond_dim, hidden_dim, ttn_params, mlp_params)
            if diff <= tolerance:
                break
        if best is not None and best[0] <= tolerance:
            break

    diff, bond_dim, hidden_dim, ttn_params, mlp_params = best
    if diff > tolerance:
        print(f"AVISO: casamento de parametros TTN/MLP ficou em {diff:.2%} (> {tolerance:.2%}). bond_dim={bond_dim}, hidden_dim={hidden_dim}")
    return hierarchy, bond_dim, hidden_dim, ttn_params, mlp_params


def _run_one_seed(args):
    torch.set_num_threads(1)

    seed = args["seed"]
    gene_names = args["gene_names"]
    hierarchy = args["hierarchy"]
    bond_dim = args["bond_dim"]
    hidden_dim = args["hidden_dim"]
    graph = args["graph"]
    train_cfg = args["train_cfg"]

    x_train = torch.tensor(args["x_train"])
    x_train_next = torch.tensor(args["x_train_next"])
    x_val = torch.tensor(args["x_val"])
    x_val_next = torch.tensor(args["x_val_next"])
    x_test = torch.tensor(args["x_test"])

    n_genes = len(gene_names)

    ttn = TTNModel(hierarchy=hierarchy, gene_names=gene_names, bond_dim=bond_dim)
    ttn_result = train_ttn(
        ttn, x_train, x_train_next, x_val, x_val_next,
        lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"], epochs=train_cfg["epochs"],
        seed=seed, patience=train_cfg["patience"], batch_size=train_cfg["batch_size"],
    )
    ttn.eval()
    ttn_scores = edge_scores_via_perturbation(ttn, n_genes, x_test)
    ttn_eval = evaluate_against_gold_standard(ttn_scores, graph, gene_names)

    mlp = MLPBaseline(n_genes=n_genes, hidden_dim=hidden_dim, num_layers=train_cfg["mlp_num_layers"])
    mlp_result = train_mlp(
        mlp, x_train, x_train_next, x_val, x_val_next,
        lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"], epochs=train_cfg["epochs"],
        seed=seed, patience=train_cfg["patience"], batch_size=train_cfg["batch_size"],
    )
    mlp.eval()
    mlp_scores = edge_scores_via_perturbation(mlp, n_genes, x_test)
    mlp_eval = evaluate_against_gold_standard(mlp_scores, graph, gene_names)

    return {
        "seed": seed,
        "ttn_aupr": ttn_eval["aupr"], "ttn_auroc": ttn_eval["auroc"],
        "mlp_aupr": mlp_eval["aupr"], "mlp_auroc": mlp_eval["auroc"],
        "ttn_diverged": ttn_result["diverged"], "mlp_diverged": mlp_result["diverged"],
    }


def run_topology_experiment(
    graph, trajectories, gene_names,
    seeds=20, n_workers=None,
    epochs=200, batch_size=16, patience=15,
    lr=0.001, weight_decay=1e-5,
    mlp_num_layers=2, significance_alpha=0.05,
    show_progress=True,
):
    seed_list = list(range(seeds)) if isinstance(seeds, int) else list(seeds)
    n_workers = n_workers or min(len(seed_list), os.cpu_count() or 4)

    train_trajs, val_trajs, test_trajs = split_trajectories_by_fraction(trajectories)
    x_train, x_train_next = trajectories_to_stacked_arrays(train_trajs)
    x_val, x_val_next = trajectories_to_stacked_arrays(val_trajs)
    x_test, _ = trajectories_to_stacked_arrays(test_trajs)

    hierarchy, bond_dim, hidden_dim, ttn_params, mlp_params = match_ttn_mlp_params(gene_names, num_layers=mlp_num_layers)
    print(f"  TTN params={ttn_params} (bond_dim={bond_dim}), MLP params={mlp_params} (hidden_dim={hidden_dim})")

    train_cfg = {
        "lr": lr, "weight_decay": weight_decay, "epochs": epochs,
        "patience": patience, "batch_size": batch_size, "mlp_num_layers": mlp_num_layers,
    }

    seed_args = [
        {
            "seed": seed, "gene_names": gene_names, "hierarchy": hierarchy,
            "bond_dim": bond_dim, "hidden_dim": hidden_dim, "graph": graph, "train_cfg": train_cfg,
            "x_train": x_train, "x_train_next": x_train_next,
            "x_val": x_val, "x_val_next": x_val_next, "x_test": x_test,
        }
        for seed in seed_list
    ]

    seed_results = []
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=n_workers, max_tasks_per_child=4) as executor:
        futures = {executor.submit(_run_one_seed, args): args["seed"] for args in seed_args}
        for future in as_completed(futures):
            seed = futures[future]
            try:
                result = future.result()
                seed_results.append(result)
            except Exception as e:
                print(f"\nAVISO: seed {seed} falhou: {e}")
                continue
            if show_progress:
                n_done = len(seed_results)
                n_total = len(seed_args)
                elapsed = time.time() - start_time
                eta = (elapsed / n_done) * (n_total - n_done) if n_done > 0 else 0
                bar = "#" * int(30 * n_done / n_total) + "-" * (30 - int(30 * n_done / n_total))
                sys.stdout.write(f"\r  [{bar}] {n_done}/{n_total} seeds | {elapsed:.1f}s | ETA {eta:.1f}s   ")
                sys.stdout.flush()
    if show_progress:
        print()

    ttn_auprs = [r["ttn_aupr"] for r in seed_results]
    mlp_auprs = [r["mlp_aupr"] for r in seed_results]
    ttn_aurocs = [r["ttn_auroc"] for r in seed_results]
    mlp_aurocs = [r["mlp_auroc"] for r in seed_results]

    _, p_aupr = wilcoxon(ttn_auprs, mlp_auprs)
    _, p_auroc = wilcoxon(ttn_aurocs, mlp_aurocs)

    h3_aupr_pass = bool(p_aupr < significance_alpha and np.mean(ttn_auprs) > np.mean(mlp_auprs))
    h3_auroc_pass = bool(p_auroc < significance_alpha and np.mean(ttn_aurocs) > np.mean(mlp_aurocs))

    return {
        "ttn_aupr_mean": float(np.mean(ttn_auprs)), "mlp_aupr_mean": float(np.mean(mlp_auprs)),
        "ttn_auroc_mean": float(np.mean(ttn_aurocs)), "mlp_auroc_mean": float(np.mean(mlp_aurocs)),
        "p_aupr": float(p_aupr), "p_auroc": float(p_auroc),
        "h3_aupr_pass": h3_aupr_pass, "h3_auroc_pass": h3_auroc_pass,
        "ttn_params": ttn_params, "mlp_params": mlp_params,
        "n_seeds": len(seed_results),
    }
