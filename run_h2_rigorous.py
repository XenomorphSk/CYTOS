"""
run_h2_rigorous.py

Revalida H2 usando a entropia RIGOROSA (canonicalizacao via QR) em vez
do proxy simplificado. Configs/comunidades que falham a canonicalizacao
sao puladas e contadas, nao silenciosamente ignoradas.
"""

import torch
from scipy.stats import spearmanr

from cytos.datasets import load_dream4
from cytos.data import split_trajectories_by_fraction, trajectories_to_stacked_arrays, detect_hierarchy
from cytos.ttn_model import TTNModel, train_ttn
from cytos.canonicalization import compute_rigorous_bond_entropy
from cytos.entanglement import perturbation_sensitivity_for_community

if __name__ == "__main__":
    configs = [(10, n) for n in range(1, 6)] + [(100, n) for n in range(1, 6)]
    seeds = [0, 1, 2, 3, 4]

    all_entropies, all_sensitivities = [], []
    n_skipped_bond_dim = 0
    records = []

    for size, network in configs:
        graph, trajectories, gene_names = load_dream4(size=size, network=network, root=".")
        hierarchy = detect_hierarchy(graph, method="louvain")

        train_trajs, val_trajs, test_trajs = split_trajectories_by_fraction(trajectories)
        x_train, x_train_next = trajectories_to_stacked_arrays(train_trajs)
        x_val, x_val_next = trajectories_to_stacked_arrays(val_trajs)
        x_test, _ = trajectories_to_stacked_arrays(test_trajs)
        x_train_t, x_train_next_t = torch.tensor(x_train), torch.tensor(x_train_next)
        x_val_t, x_val_next_t = torch.tensor(x_val), torch.tensor(x_val_next)
        x_test_t = torch.tensor(x_test)

        for seed in seeds:
            bond_dim = 3
            ttn = TTNModel(hierarchy=hierarchy, gene_names=gene_names, bond_dim=bond_dim)
            train_ttn(ttn, x_train_t, x_train_next_t, x_val_t, x_val_next_t, epochs=100, seed=seed)
            ttn.eval()

            partition = hierarchy["level_0"]
            for comm_id in ttn.community_ids:
                plan, _ = ttn.community_plans[comm_id]
                if not plan:
                    continue
                try:
                    entropy = compute_rigorous_bond_entropy(ttn, comm_id, x_test_t)
                except (ValueError, RuntimeError) as e:
                    n_skipped_bond_dim += 1
                    continue
                sensitivity = perturbation_sensitivity_for_community(
                    ttn, comm_id, gene_names, partition, x_test_t
                )
                if sensitivity == sensitivity:
                    all_entropies.append(entropy)
                    all_sensitivities.append(sensitivity)
                    records.append({
                        "size": size, "network": network, "seed": seed,
                        "community": comm_id, "entropy_rigorous": entropy,
                        "sensitivity": sensitivity,
                    })

        print(f"Config {size}/{network} processada ({len(records)} pares acumulados)")

    rho, p_value = spearmanr(all_entropies, all_sensitivities)
    print(f"\n=== H2 com entropia RIGOROSA ===")
    print(f"N pares: {len(all_entropies)}")
    print(f"N pulados (falha na canonicalizacao): {n_skipped_bond_dim}")
    print(f"Spearman rho={rho:.4f}, p={p_value:.4e}")
    print(f"H2-rigoroso: {'PASSOU' if rho > 0 and p_value < 0.05 else 'FALHOU'}")
