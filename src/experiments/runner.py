"""
src/experiments/runner.py

Orquestra o experimento completo do pre-registro, usando dados REAIS do
DREAM4 (rede 1 e 2, por tamanho), com seeds paralelizadas via
ProcessPoolExecutor.

Rodar com: python -m src.experiments.runner
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from scipy.stats import wilcoxon

from src.data.pipeline import build_dataset, load_config, split_trajectories_by_fraction
from src.models.gnn_baseline import GNNBaseline, train_gnn
from src.models.ttn_model import TTNModel, train_ttn


def match_parameter_counts(hierarchy, gene_names, num_nodes, cfg):
    tol = cfg["param_match_tolerance"]
    num_layers = cfg["gnn"]["num_layers"]
    architecture = cfg["gnn"]["architecture"]

    hidden_dim_candidates = [2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 144, 160, 176, 192, 224, 256]
    bond_dim_candidates = list(range(2, 65))

    best = None

    for hidden_dim in hidden_dim_candidates:
        gnn_probe = GNNBaseline(num_nodes=num_nodes, hidden_dim=hidden_dim, num_layers=num_layers, architecture=architecture)
        gnn_params = gnn_probe.count_parameters()

        for bond_dim in bond_dim_candidates:
            ttn_probe = TTNModel(hierarchy=hierarchy, gene_names=gene_names, bond_dim=bond_dim)
            ttn_params = ttn_probe.count_parameters()
            diff_ratio = abs(ttn_params - gnn_params) / max(gnn_params, 1)

            if best is None or diff_ratio < best[0]:
                best = (diff_ratio, hidden_dim, bond_dim, gnn_params, ttn_params)
            if diff_ratio <= tol:
                break
        if best is not None and best[0] <= tol:
            break

    diff_ratio, hidden_dim, bond_dim, gnn_params, ttn_params = best

    if diff_ratio > tol:
        print(f"AVISO: nao foi possivel casar parametros dentro da tolerancia ({diff_ratio:.2%} > {tol:.2%}). Melhor par: hidden_dim={hidden_dim} (GNN, {gnn_params} params), bond_dim={bond_dim} (TTN, {ttn_params} params).")

    return hidden_dim, bond_dim, gnn_params, ttn_params


def long_range_correlation(pred, true, hierarchy, gene_names):
    pred = np.asarray(pred).flatten()
    true = np.asarray(true).flatten()

    partition = hierarchy["level_0"]
    pairs = []
    for i, gi in enumerate(gene_names):
        for j, gj in enumerate(gene_names):
            if i >= j:
                continue
            if partition.get(gi, -1) != partition.get(gj, -1):
                pairs.append((i, j))

    if not pairs:
        return float("nan")

    pred_diffs = np.array([pred[i] - pred[j] for i, j in pairs])
    true_diffs = np.array([true[i] - true[j] for i, j in pairs])
    if np.std(pred_diffs) == 0 or np.std(true_diffs) == 0:
        return float("nan")
    return float(np.corrcoef(pred_diffs, true_diffs)[0, 1])


def trajectories_to_gnn_pairs(trajectories, edge_index):
    pairs = []
    for traj in trajectories:
        for t in range(len(traj) - 1):
            x_t = torch.tensor(traj[t], dtype=torch.float32).unsqueeze(-1)
            x_next = torch.tensor(traj[t + 1], dtype=torch.float32).unsqueeze(-1)
            pairs.append((x_t, edge_index, x_next))
    return pairs


def trajectories_to_ttn_pairs(trajectories):
    pairs = []
    for traj in trajectories:
        for t in range(len(traj) - 1):
            x_t = torch.tensor(traj[t], dtype=torch.float32)
            x_next = torch.tensor(traj[t + 1], dtype=torch.float32)
            pairs.append((x_t, x_next))
    return pairs


def run_one_seed(args):
    torch.set_num_threads(1)

    seed = args["seed"]
    gene_names = args["gene_names"]
    hierarchy = args["hierarchy"]
    edge_index = args["edge_index"]
    gnn_hidden_dim = args["gnn_hidden_dim"]
    bond_dim = args["bond_dim"]
    cfg = args["cfg"]
    gnn_train, gnn_val, gnn_test = args["gnn_train"], args["gnn_val"], args["gnn_test"]
    ttn_train, ttn_val, ttn_test = args["ttn_train"], args["ttn_val"], args["ttn_test"]

    gnn = GNNBaseline(num_nodes=len(gene_names), hidden_dim=gnn_hidden_dim, num_layers=cfg["gnn"]["num_layers"], architecture=cfg["gnn"]["architecture"])
    gnn_result = train_gnn(gnn, gnn_train, gnn_val, lr=cfg["gnn"]["lr"], weight_decay=cfg["gnn"]["weight_decay"], epochs=cfg["gnn"]["epochs"], seed=seed)
    gnn_mse_per_param = gnn_result["best_val_mse"] / gnn.count_parameters()

    ttn = TTNModel(hierarchy=hierarchy, gene_names=gene_names, bond_dim=bond_dim)
    ttn_result = train_ttn(ttn, ttn_train, ttn_val, lr=cfg["ttn"]["lr"], weight_decay=cfg["ttn"]["weight_decay"], epochs=cfg["ttn"]["epochs"], seed=seed)
    ttn_mse_per_param = ttn_result["best_val_mse"] / ttn.count_parameters()

    with torch.no_grad():
        gnn.eval()
        ttn.eval()
        gnn_preds = [gnn(x_t, edge_index).squeeze(-1).numpy() for x_t, _, _ in gnn_test]
        ttn_preds = [ttn(x_t).numpy() for x_t, _ in ttn_test]
        trues = [x_next.numpy() for _, _, x_next in gnn_test]

        gnn_lr_corrs = [long_range_correlation(p, t, hierarchy, gene_names) for p, t in zip(gnn_preds, trues)]
        ttn_lr_corrs = [long_range_correlation(p, t, hierarchy, gene_names) for p, t in zip(ttn_preds, trues)]

    return {
        "seed": seed,
        "gnn_mse_per_param": gnn_mse_per_param,
        "ttn_mse_per_param": ttn_mse_per_param,
        "gnn_lr_corrs": gnn_lr_corrs,
        "ttn_lr_corrs": ttn_lr_corrs,
        "gnn_diverged": gnn_result["diverged"],
        "ttn_diverged": ttn_result["diverged"],
    }


def run_single_config(dataset, cfg):
    gene_names = dataset.gene_names
    hierarchy = dataset.hierarchy
    graph = dataset.graph

    train_trajs, val_trajs, test_trajs = split_trajectories_by_fraction(
        dataset.trajectories,
        cfg["data"]["train_traj_frac"],
        cfg["data"]["val_traj_frac"],
        cfg["data"]["test_traj_frac"],
    )

    node_to_idx = {n: i for i, n in enumerate(gene_names)}
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in graph.edges() if u in node_to_idx and v in node_to_idx]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    gnn_train = trajectories_to_gnn_pairs(train_trajs, edge_index)
    gnn_val = trajectories_to_gnn_pairs(val_trajs, edge_index)
    gnn_test = trajectories_to_gnn_pairs(test_trajs, edge_index)
    ttn_train = trajectories_to_ttn_pairs(train_trajs)
    ttn_val = trajectories_to_ttn_pairs(val_trajs)
    ttn_test = trajectories_to_ttn_pairs(test_trajs)

    gnn_hidden_dim, bond_dim, gnn_params_matched, ttn_params_matched = match_parameter_counts(
        hierarchy, gene_names, num_nodes=len(gene_names), cfg=cfg
    )

    seed_args = [
        {
            "seed": seed, "gene_names": gene_names, "hierarchy": hierarchy, "edge_index": edge_index,
            "gnn_hidden_dim": gnn_hidden_dim, "bond_dim": bond_dim, "cfg": cfg,
            "gnn_train": gnn_train, "gnn_val": gnn_val, "gnn_test": gnn_test,
            "ttn_train": ttn_train, "ttn_val": ttn_val, "ttn_test": ttn_test,
        }
        for seed in cfg["seed_list"]
    ]

    n_workers = min(len(seed_args), os.cpu_count() or 4)
    print(f"  Rodando {len(seed_args)} seeds em paralelo ({n_workers} workers)...")

    seed_results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(run_one_seed, args): args["seed"] for args in seed_args}
        for future in as_completed(futures):
            seed = futures[future]
            try:
                result = future.result()
                seed_results.append(result)
                print(f"    seed {seed} concluida ({len(seed_results)}/{len(seed_args)})")
            except Exception as e:
                print(f"    AVISO: seed {seed} falhou com erro: {e}")

    gnn_mses = [r["gnn_mse_per_param"] for r in seed_results]
    ttn_mses = [r["ttn_mse_per_param"] for r in seed_results]
    gnn_lr_corrs = [c for r in seed_results for c in r["gnn_lr_corrs"]]
    ttn_lr_corrs = [c for r in seed_results for c in r["ttn_lr_corrs"]]
    n_gnn_diverged = sum(r["gnn_diverged"] for r in seed_results)
    n_ttn_diverged = sum(r["ttn_diverged"] for r in seed_results)

    if n_gnn_diverged > 0 or n_ttn_diverged > 0:
        print(f"  AVISO: GNN divergiu em {n_gnn_diverged}/{len(seed_results)} seeds, TTN divergiu em {n_ttn_diverged}/{len(seed_results)} seeds.")

    stat_primary, p_primary = wilcoxon(ttn_mses, gnn_mses)
    gnn_lr_clean = [v for v in gnn_lr_corrs if not np.isnan(v)]
    ttn_lr_clean = [v for v in ttn_lr_corrs if not np.isnan(v)]
    if len(gnn_lr_clean) == len(ttn_lr_clean) and len(gnn_lr_clean) > 0:
        stat_secondary, p_secondary = wilcoxon(ttn_lr_clean, gnn_lr_clean)
    else:
        stat_secondary, p_secondary = float("nan"), float("nan")

    alpha = cfg["evaluation"]["significance_alpha"]
    return {
        "size": dataset.size,
        "network": dataset.network,
        "gnn_hidden_dim_used": gnn_hidden_dim,
        "ttn_bond_dim_used": bond_dim,
        "gnn_params": gnn_params_matched,
        "ttn_params": ttn_params_matched,
        "param_diff_ratio": abs(ttn_params_matched - gnn_params_matched) / max(gnn_params_matched, 1),
        "n_gnn_diverged": n_gnn_diverged,
        "n_ttn_diverged": n_ttn_diverged,
        "ttn_mse_per_param_mean": float(np.mean(ttn_mses)),
        "gnn_mse_per_param_mean": float(np.mean(gnn_mses)),
        "h1_pass": bool(p_primary < alpha and np.mean(ttn_mses) < np.mean(gnn_mses)),
        "p_primary": float(p_primary),
        "ttn_long_range_corr_mean": float(np.mean(ttn_lr_clean)) if ttn_lr_clean else float("nan"),
        "gnn_long_range_corr_mean": float(np.mean(gnn_lr_clean)) if gnn_lr_clean else float("nan"),
        "h1b_pass": bool(not np.isnan(p_secondary) and p_secondary < alpha and np.mean(ttn_lr_clean) > np.mean(gnn_lr_clean)),
        "p_secondary": float(p_secondary),
    }


def run_all(config_path="config/config.yaml"):
    cfg = load_config(config_path)
    datasets = build_dataset(config_path)

    confirmatory_networks = set(cfg["data"]["confirmatory_networks"])

    results = []
    for dataset in datasets:
        label = "CONFIRMATORIO" if dataset.network in confirmatory_networks else "EXPLORATORIO"
        print(f"\n=== Config: {dataset.size} genes / rede {dataset.network} [{label}] ===")
        result = run_single_config(dataset, cfg)
        result["label"] = label
        results.append(result)
        print(result)

    confirmatory = [r for r in results if r["label"] == "CONFIRMATORIO"]
    exploratory = [r for r in results if r["label"] == "EXPLORATORIO"]

    if confirmatory:
        n_h1_pass = sum(r["h1_pass"] for r in confirmatory)
        n_h1b_pass = sum(r["h1b_pass"] for r in confirmatory)
        print(f"\n[CONFIRMATORIO] H1 passou em {n_h1_pass}/{len(confirmatory)} configs")
        print(f"[CONFIRMATORIO] H1b passou em {n_h1b_pass}/{len(confirmatory)} configs")

    if exploratory:
        n_h1_pass_exp = sum(r["h1_pass"] for r in exploratory)
        n_h1b_pass_exp = sum(r["h1b_pass"] for r in exploratory)
        print(f"\n[EXPLORATORIO - nao conta para o pre-registro] H1 passou em {n_h1_pass_exp}/{len(exploratory)} configs adicionais")
        print(f"[EXPLORATORIO - nao conta para o pre-registro] H1b passou em {n_h1b_pass_exp}/{len(exploratory)} configs adicionais")

    return results


if __name__ == "__main__":
    run_all()
