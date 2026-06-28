"""
src/analysis/entanglement_pilot.py

Fase 2 (pre-registrada na Secao 15 do pre_registration.md): testa se a
"entropia local de bond" de uma comunidade prediz sua sensibilidade a
perturbacao (knockout simulado).

Rodar com: python -m src.analysis.entanglement_pilot
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import spearmanr

from src.data.pipeline import build_dataset, load_config, split_trajectories_by_fraction
from src.experiments.runner import match_parameter_counts, trajectories_to_stacked_arrays
from src.models.ttn_model import TTNModel, train_ttn


def bond_entropy_for_community(model: TTNModel, comm_id) -> float:
    target_leaf = ("leaf", comm_id)

    for left_ref, right_ref, module_idx in model.root_plan:
        if left_ref == target_leaf or right_ref == target_leaf:
            comm_is_left = left_ref == target_leaf
            linear = model.contractions[module_idx]
            W = linear.weight.detach().numpy()
            bond_dim = model.bond_dim
            W3 = W.reshape(bond_dim, bond_dim, bond_dim)

            if comm_is_left:
                M = W3.transpose(1, 0, 2).reshape(bond_dim, bond_dim * bond_dim)
            else:
                M = W3.transpose(2, 0, 1).reshape(bond_dim, bond_dim * bond_dim)

            sigma = np.linalg.svd(M, compute_uv=False)
            sigma_sq = sigma ** 2
            total = sigma_sq.sum()
            if total < 1e-12:
                return 0.0
            p = sigma_sq / total
            p = p[p > 1e-12]
            return float(-np.sum(p * np.log(p)))

    return float("nan")


def perturbation_sensitivity_for_community(model, comm_id, gene_names, partition, x_test):
    community_gene_idx = [i for i, g in enumerate(gene_names) if partition.get(g, -1) == comm_id]
    outside_idx = [i for i in range(len(gene_names)) if i not in community_gene_idx]

    if not community_gene_idx or not outside_idx:
        return float("nan")

    with torch.no_grad():
        pred_baseline = model(x_test)

        x_knockout = x_test.clone()
        x_knockout[:, community_gene_idx] = 0.0
        pred_knockout = model(x_knockout)

        diff = (pred_knockout[:, outside_idx] - pred_baseline[:, outside_idx]).abs()
        return float(diff.mean().item())


def run_pilot(config_path: str = "config/config.yaml", seeds: list = None) -> dict:
    if seeds is None:
        seeds = [0, 1, 2, 3, 4]

    cfg = load_config(config_path)
    datasets = build_dataset(config_path)

    all_entropies = []
    all_sensitivities = []
    records = []
    per_seed_rho = {}

    for seed in seeds:
        print(f"\n=== Seed {seed} ===")
        seed_entropies = []
        seed_sensitivities = []

        for dataset in datasets:
            print(f"  Processando: {dataset.size} genes / rede {dataset.network}...")
            gene_names = dataset.gene_names
            hierarchy = dataset.hierarchy
            partition = hierarchy["level_0"]

            train_trajs, val_trajs, test_trajs = split_trajectories_by_fraction(
                dataset.trajectories,
                cfg["data"]["train_traj_frac"],
                cfg["data"]["val_traj_frac"],
                cfg["data"]["test_traj_frac"],
            )

            x_train, x_train_next = trajectories_to_stacked_arrays(train_trajs)
            x_val, x_val_next = trajectories_to_stacked_arrays(val_trajs)
            x_test, _ = trajectories_to_stacked_arrays(test_trajs)

            _, bond_dim, _, _ = match_parameter_counts(hierarchy, gene_names, len(gene_names), cfg)

            ttn = TTNModel(hierarchy=hierarchy, gene_names=gene_names, bond_dim=bond_dim)
            train_ttn(
                ttn,
                torch.tensor(x_train), torch.tensor(x_train_next),
                torch.tensor(x_val), torch.tensor(x_val_next),
                lr=cfg["ttn"]["lr"], weight_decay=cfg["ttn"]["weight_decay"],
                epochs=cfg["ttn"]["epochs"], seed=seed,
            )
            ttn.eval()

            x_test_t = torch.tensor(x_test)

            for comm_id in ttn.community_ids:
                entropy = bond_entropy_for_community(ttn, comm_id)
                sensitivity = perturbation_sensitivity_for_community(
                    ttn, comm_id, gene_names, partition, x_test_t
                )
                if not (np.isnan(entropy) or np.isnan(sensitivity)):
                    seed_entropies.append(entropy)
                    seed_sensitivities.append(sensitivity)
                    all_entropies.append(entropy)
                    all_sensitivities.append(sensitivity)
                    records.append({
                        "seed": seed, "size": dataset.size, "network": dataset.network,
                        "community": comm_id, "entropy": entropy, "sensitivity": sensitivity,
                    })

        rho_seed, p_seed = spearmanr(seed_entropies, seed_sensitivities)
        per_seed_rho[seed] = (float(rho_seed), float(p_seed))
        print(f"  Seed {seed}: rho={rho_seed:.4f}, p={p_seed:.4e}")

    rho, p_value = spearmanr(all_entropies, all_sensitivities)
    alpha = cfg["evaluation"]["significance_alpha"]
    h2_pass = bool(rho > 0 and p_value < alpha)
    n_seeds_positive = sum(1 for r, _ in per_seed_rho.values() if r > 0)

    print(f"\n=== Resultado agregado (todas as seeds e configs) ===")
    print(f"N pares (comunidade x seed) analisados: {len(all_entropies)}")
    print(f"Spearman rho = {rho:.4f}, p = {p_value:.4e}")
    print(f"H2 (criterio pre-registrado: rho>0 e p<{alpha}): {'PASSOU' if h2_pass else 'FALHOU'}")
    print(f"Checagem secundaria: {n_seeds_positive}/{len(seeds)} seeds tiveram rho>0 individualmente")

    return {
        "records": records,
        "rho": float(rho),
        "p_value": float(p_value),
        "h2_pass": h2_pass,
        "n_pairs": len(all_entropies),
        "per_seed_rho": per_seed_rho,
        "n_seeds_positive": n_seeds_positive,
    }


if __name__ == "__main__":
    run_pilot()
