"""
test_rigorous_entanglement_validation.py

Valida rigorous_bond_entropy_for_community contra uma referencia EXATA
(forca bruta): contrai a rede inteira num tensor completo via probing
com base canonica nas folhas, SVD direto, sem canonicalizacao. Se os
dois nao baterem, ha bug de integracao na canonicalizacao.
"""

import itertools
import numpy as np
import torch

from cytos.ttn_model import TTNModel
from cytos.rigorous_entanglement import rigorous_bond_entropy_for_community


def brute_force_full_tensor(model):
    n_genes = len(model.gene_names)
    bond_dim = model.bond_dim
    if n_genes > 10:
        raise ValueError("Forca bruta so e viavel para <=10 genes.")

    full_tensor = np.zeros([bond_dim] + [2] * n_genes)

    with torch.no_grad():
        for bits in itertools.product([0, 1], repeat=n_genes):
            leaf_vecs = {}
            for gi, gene in enumerate(model.gene_names):
                vec = torch.zeros(1, 2)
                vec[0, bits[gi]] = 1.0
                leaf_vecs[gene] = vec

            community_vecs = {}
            for comm_id, (plan, root_ref) in model.community_plans.items():
                values = model._execute_plan(plan, leaf_vecs)
                vec = leaf_vecs[root_ref[1]] if root_ref[0] == "leaf" else values[root_ref[1]]
                if str(comm_id) in model.community_finalize:
                    vec = model.community_finalize[str(comm_id)](vec)
                community_vecs[comm_id] = vec

            root_values = model._execute_plan(model.root_plan, community_vecs)
            global_repr = (
                community_vecs[model.root_ref[1]] if model.root_ref[0] == "leaf" else root_values[model.root_ref[1]]
            )
            full_tensor[(slice(None),) + bits] = global_repr.squeeze(0).numpy()

    return full_tensor


def brute_force_entropy_for_community(model, comm_id):
    full_tensor = brute_force_full_tensor(model)
    community_gene_idx = [i for i, g in enumerate(model.gene_names) if model.gene_to_community_slot[g] == model.community_ids.index(comm_id)]
    other_idx = [i for i in range(len(model.gene_names)) if i not in community_gene_idx]

    order = [0] + [1 + i for i in community_gene_idx] + [1 + i for i in other_idx]
    reordered = full_tensor.transpose(order)
    bond_dim = full_tensor.shape[0]
    target_size = 2 ** len(community_gene_idx)
    reordered2 = reordered.reshape(bond_dim, target_size, 2 ** len(other_idx))
    reordered2 = reordered2.transpose(1, 0, 2)
    M = reordered2.reshape(target_size, bond_dim * (2 ** len(other_idx)))

    sigma = np.linalg.svd(M, compute_uv=False)
    sigma_sq = sigma ** 2
    total = sigma_sq.sum()
    if total < 1e-12:
        return 0.0
    p = sigma_sq / total
    p = p[p > 1e-12]
    return float(-np.sum(p * np.log(p)))


if __name__ == "__main__":
    gene_names = [f"G{i}" for i in range(8)]
    hierarchy = {"level_0": {f"G{i}": i // 3 for i in range(8)}}

    model = TTNModel(hierarchy=hierarchy, gene_names=gene_names, bond_dim=3)

    for comm_id in model.community_ids:
        rigorous = rigorous_bond_entropy_for_community(model, comm_id)
        brute = brute_force_entropy_for_community(model, comm_id)
        match = abs(rigorous - brute) < 1e-6
        print(f"Comunidade {comm_id}: rigoroso={rigorous:.6f}, forca_bruta={brute:.6f}, BATEM={match}")
