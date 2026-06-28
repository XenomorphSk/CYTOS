"""
cytos/entanglement.py

API para o piloto de "entropia local de bond" (proxy simplificado, NAO
entropia de emaranhamento de von Neumann rigorosa - requer
canonicalizacao completa da rede, nao implementada).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.stats import spearmanr

from cytos.ttn_model import TTNModel, train_ttn


def bond_entropy_for_community(model, comm_id):
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


@dataclass
class EntanglementPilotResult:
    rho: float
    p_value: float
    h2_pass: bool
    n_pairs: int
    per_seed_rho: dict
    n_seeds_positive: int
    n_seeds: int
    records: list

    def summary(self) -> str:
        lines = [
            "=== Piloto de Entropia de Bond (proxy, nao entropia de von Neumann rigorosa) ===",
            f"N pares (comunidade x seed): {self.n_pairs}",
            f"Spearman rho={self.rho:.4f}, p={self.p_value:.3e}",
            f"H2 (rho>0 e p<0.05): {'PASSOU' if self.h2_pass else 'FALHOU'}",
            f"Seeds com rho>0 individualmente: {self.n_seeds_positive}/{self.n_seeds}",
        ]
        return "\n".join(lines)

    def to_dataframe(self):
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas nao instalado. Use result.records (lista de dicts) em vez disso.")
        return pd.DataFrame(self.records)


class EntanglementPilot:
    def __init__(self, ttnvsgnn):
        self.experiment = ttnvsgnn

    def run(self, seeds=None, epochs=200, batch_size=16, patience=15, lr=0.001, weight_decay=1e-5, significance_alpha=0.05):
        if seeds is None:
            seeds = [0, 1, 2, 3, 4]

        exp = self.experiment
        partition = exp.hierarchy["level_0"]

        all_entropies, all_sensitivities, records = [], [], []
        per_seed_rho = {}

        x_train_t = torch.tensor(exp.x_train)
        x_train_next_t = torch.tensor(exp.x_train_next)
        x_val_t = torch.tensor(exp.x_val)
        x_val_next_t = torch.tensor(exp.x_val_next)
        x_test_t = torch.tensor(exp.x_test)

        for seed in seeds:
            seed_entropies, seed_sensitivities = [], []

            ttn = TTNModel(hierarchy=exp.hierarchy, gene_names=exp.gene_names, bond_dim=exp.bond_dim)
            train_ttn(
                ttn, x_train_t, x_train_next_t, x_val_t, x_val_next_t,
                lr=lr, weight_decay=weight_decay, epochs=epochs,
                seed=seed, patience=patience, batch_size=batch_size,
            )
            ttn.eval()

            for comm_id in ttn.community_ids:
                entropy = bond_entropy_for_community(ttn, comm_id)
                sensitivity = perturbation_sensitivity_for_community(
                    ttn, comm_id, exp.gene_names, partition, x_test_t
                )
                if not (np.isnan(entropy) or np.isnan(sensitivity)):
                    seed_entropies.append(entropy)
                    seed_sensitivities.append(sensitivity)
                    all_entropies.append(entropy)
                    all_sensitivities.append(sensitivity)
                    records.append({"seed": seed, "community": comm_id, "entropy": entropy, "sensitivity": sensitivity})

            rho_seed, p_seed = spearmanr(seed_entropies, seed_sensitivities)
            per_seed_rho[seed] = (float(rho_seed), float(p_seed))

        rho, p_value = spearmanr(all_entropies, all_sensitivities)
        h2_pass = bool(rho > 0 and p_value < significance_alpha)
        n_seeds_positive = sum(1 for r, _ in per_seed_rho.values() if r > 0)

        n_unique_entropies = len(set(round(e, 6) for e in all_entropies))
        if len(all_entropies) > 0 and n_unique_entropies < max(3, len(all_entropies) // 4):
            print(
                f"AVISO: apenas {n_unique_entropies} valores distintos de entropia "
                f"entre {len(all_entropies)} pares analisados. Isso e esperado quando "
                f"se testa um unico grafo pequeno com poucas comunidades, "
                f"especialmente combinado com bond_dim baixo (resolucao limitada da "
                f"metrica). O resultado original deste metodo foi confirmado agregando "
                f"comunidades de MULTIPLOS grafos/topologias - considere rodar o piloto "
                f"em varios grafos e agregar os resultados antes de interpretar H2 "
                f"como confirmado ou falseado com base em um unico grafo pequeno."
            )

        return EntanglementPilotResult(
            rho=float(rho), p_value=float(p_value), h2_pass=h2_pass,
            n_pairs=len(all_entropies), per_seed_rho=per_seed_rho,
            n_seeds_positive=n_seeds_positive, n_seeds=len(seeds), records=records,
        )
