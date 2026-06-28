"""
src/models/ttn_model.py

Genuine multilinear binary Tree Tensor Network - FLAT EXECUTION PLAN
(2026-06-24, fix de performance).

A arvore e percorrida (post-order) UMA UNICA VEZ em __init__, gerando um
plano de execucao plano (lista simples de operacoes, sem recursao). No
forward, esse plano e apenas iterado em loop simples - sem reconstruir
estrutura, sem recursao, sem isinstance() repetido.
"""

from __future__ import annotations

import torch
import torch.nn as nn

LEAF_DIM = 2


def leaf_embedding(x: torch.Tensor) -> torch.Tensor:
    angle = x * (torch.pi / 2)
    return torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)


def build_binary_tree(items: list):
    nodes = list(items)
    if not nodes:
        return None
    while len(nodes) > 1:
        next_level = []
        i = 0
        while i + 1 < len(nodes):
            next_level.append((nodes[i], nodes[i + 1]))
            i += 2
        if i < len(nodes):
            next_level.append(nodes[i])
        nodes = next_level
    return nodes[0]


def _build_flat_plan(tree, leaf_dim_fn, bond_dim, contractions_list):
    plan = []

    def walk(node):
        if not isinstance(node, tuple):
            return ("leaf", node)
        left, right = node
        left_ref = walk(left)
        right_ref = walk(right)
        left_dim = leaf_dim_fn(left) if left_ref[0] == "leaf" else bond_dim
        right_dim = leaf_dim_fn(right) if right_ref[0] == "leaf" else bond_dim
        module_idx = len(contractions_list)
        contractions_list.append(nn.Linear(left_dim * right_dim, bond_dim, bias=False))
        plan.append((left_ref, right_ref, module_idx))
        return ("node", len(plan) - 1)

    root_ref = walk(tree) if tree is not None else None
    return plan, root_ref


class TTNModel(nn.Module):
    def __init__(self, hierarchy: dict, gene_names: list, bond_dim: int = 4):
        super().__init__()
        self.gene_names = gene_names
        self.gene_to_idx = {g: i for i, g in enumerate(gene_names)}
        self.bond_dim = bond_dim

        partition = hierarchy["level_0"]
        communities = {}
        for gene in gene_names:
            comm_id = partition.get(gene, -1)
            communities.setdefault(comm_id, []).append(gene)
        self.community_ids = list(communities.keys())
        self.gene_to_community_slot = {
            gene: self.community_ids.index(partition.get(gene, -1))
            for gene in gene_names
        }

        self.contractions = nn.ModuleList()

        self.community_plans = {}
        community_trees = {}
        for comm_id, members in communities.items():
            tree = build_binary_tree(members)
            community_trees[comm_id] = tree
            plan, root_ref = _build_flat_plan(
                tree, leaf_dim_fn=lambda _g: LEAF_DIM, bond_dim=bond_dim,
                contractions_list=self.contractions,
            )
            self.community_plans[comm_id] = (plan, root_ref)

        self.community_finalize = nn.ModuleDict()
        for comm_id, tree in community_trees.items():
            out_dim = self.bond_dim if isinstance(tree, tuple) else LEAF_DIM
            if out_dim != self.bond_dim:
                self.community_finalize[str(comm_id)] = nn.Linear(out_dim, self.bond_dim, bias=False)

        root_tree = build_binary_tree(list(communities.keys()))
        self.root_plan, self.root_ref = _build_flat_plan(
            root_tree, leaf_dim_fn=lambda _c: bond_dim, bond_dim=bond_dim,
            contractions_list=self.contractions,
        )

        self.output_head = nn.Linear(LEAF_DIM + 2 * bond_dim, 1)

        self.register_buffer(
            "community_slot_per_gene",
            torch.tensor([self.gene_to_community_slot[g] for g in gene_names], dtype=torch.long),
            persistent=False,
        )

    def _execute_plan(self, plan, leaf_lookup):
        values = []
        for left_ref, right_ref, module_idx in plan:
            left_val = leaf_lookup[left_ref[1]] if left_ref[0] == "leaf" else values[left_ref[1]]
            right_val = leaf_lookup[right_ref[1]] if right_ref[0] == "leaf" else values[right_ref[1]]
            outer = torch.einsum("bi,bj->bij", left_val, right_val)
            outer = outer.reshape(outer.shape[0], -1)
            values.append(self.contractions[module_idx](outer))
        return values

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        single = x.dim() == 1
        if single:
            x = x.unsqueeze(0)

        embedded = leaf_embedding(x)
        leaf_vecs = {
            gene: embedded[:, self.gene_to_idx[gene], :] for gene in self.gene_names
        }

        community_vecs = {}
        for comm_id, (plan, root_ref) in self.community_plans.items():
            values = self._execute_plan(plan, leaf_vecs)
            vec = leaf_vecs[root_ref[1]] if root_ref[0] == "leaf" else values[root_ref[1]]
            if str(comm_id) in self.community_finalize:
                vec = self.community_finalize[str(comm_id)](vec)
            community_vecs[comm_id] = vec

        root_values = self._execute_plan(self.root_plan, community_vecs)
        global_repr = (
            community_vecs[self.root_ref[1]] if self.root_ref[0] == "leaf" else root_values[self.root_ref[1]]
        )

        n_genes = len(self.gene_names)
        comm_stack = torch.stack([community_vecs[cid] for cid in self.community_ids], dim=0)
        gathered_comm = comm_stack[self.community_slot_per_gene]
        leaf_stack = embedded.permute(1, 0, 2)
        global_expanded = global_repr.unsqueeze(0).expand(n_genes, -1, -1)

        combined = torch.cat([leaf_stack, gathered_comm, global_expanded], dim=-1)
        combined_flat = combined.reshape(-1, combined.shape[-1])
        out_flat = self.output_head(combined_flat)
        out = out_flat.reshape(n_genes, -1).transpose(0, 1)

        if single:
            out = out.squeeze(0)
        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_ttn(
    model,
    x_train: torch.Tensor,
    x_train_next: torch.Tensor,
    x_val: torch.Tensor,
    x_val_next: torch.Tensor,
    lr: float = 0.001,
    weight_decay: float = 1e-5,
    epochs: int = 200,
    seed: int = 0,
    patience: int = 15,
    batch_size: int = 16,
) -> dict:
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    n_train = x_train.shape[0]
    best_val_mse = float("inf")
    history = {"train_loss": [], "val_loss": []}
    diverged = False
    epochs_without_improvement = 0
    generator = torch.Generator().manual_seed(seed)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train, generator=generator)
        total_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            xb, xb_next = x_train[idx], x_train_next[idx]
            optimizer.zero_grad()
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, xb_next)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_train_loss = total_loss / max(n_batches, 1)
        history["train_loss"].append(avg_train_loss)

        if not torch.isfinite(torch.tensor(avg_train_loss)) or avg_train_loss > 1e6:
            diverged = True
            print(f"AVISO: TTN divergiu na epoch {epoch} (train_loss={avg_train_loss:.2e}).")
            break

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val)
            val_loss = nn.functional.mse_loss(val_pred, x_val_next).item()
            history["val_loss"].append(val_loss)
            if val_loss < best_val_mse - 1e-12:
                best_val_mse = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    return {"history": history, "best_val_mse": best_val_mse, "diverged": diverged}
