"""
src/models/ttn_model.py

Tree Tensor Network (TTN) para predicao de expressao genica t -> t+1.
Saida: cabeca de leitura local compartilhada entre todos os genes
(nao escala com n_genes).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TTNModel(nn.Module):
    def __init__(
        self,
        hierarchy: dict,
        gene_names: list[str],
        bond_dim: int = 4,
    ):
        super().__init__()
        self.gene_names = gene_names
        self.gene_to_idx = {g: i for i, g in enumerate(gene_names)}
        self.bond_dim = bond_dim

        partition = hierarchy["level_0"]
        self.communities: dict[int, list[str]] = {}
        for gene in gene_names:
            comm_id = partition.get(gene, -1)
            self.communities.setdefault(comm_id, []).append(gene)

        self.community_ids = list(self.communities.keys())
        self.gene_to_community_slot = {
            gene: self.community_ids.index(partition.get(gene, -1))
            for gene in gene_names
        }

        self.leaf_proj = nn.Linear(1, bond_dim)

        self.community_tensors = nn.ModuleDict()
        for comm_id, members in self.communities.items():
            in_dim = len(members) * bond_dim
            self.community_tensors[str(comm_id)] = nn.Linear(in_dim, bond_dim)

        n_communities = len(self.communities)
        self.root_tensor = nn.Linear(n_communities * bond_dim, bond_dim)

        self.output_head = nn.Linear(3 * bond_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        leaf_vecs = self.leaf_proj(x.unsqueeze(-1))

        community_vecs = []
        for comm_id in self.community_ids:
            members = self.communities[comm_id]
            idxs = [self.gene_to_idx[m] for m in members]
            member_vecs = leaf_vecs[idxs].reshape(-1)
            comm_out = torch.relu(self.community_tensors[str(comm_id)](member_vecs))
            community_vecs.append(comm_out)
        community_vecs_stacked = torch.stack(community_vecs, dim=0)

        root_input = torch.cat(community_vecs, dim=0)
        global_repr = torch.relu(self.root_tensor(root_input))

        outputs = []
        for gene in self.gene_names:
            gi = self.gene_to_idx[gene]
            ci = self.gene_to_community_slot[gene]
            local_context = torch.cat(
                [leaf_vecs[gi], community_vecs_stacked[ci], global_repr], dim=0
            )
            outputs.append(self.output_head(local_context))
        return torch.cat(outputs, dim=0)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_ttn(
    model: TTNModel,
    train_data: list[tuple[torch.Tensor, torch.Tensor]],
    val_data: list[tuple[torch.Tensor, torch.Tensor]],
    lr: float = 0.001,
    weight_decay: float = 1e-5,
    epochs: int = 200,
    seed: int = 0,
    patience: int = 15,
) -> dict:
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_mse = float("inf")
    history = {"train_loss": [], "val_loss": []}
    diverged = False
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x_t, x_next in train_data:
            optimizer.zero_grad()
            pred = model(x_t)
            loss = nn.functional.mse_loss(pred, x_next)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / max(len(train_data), 1)
        history["train_loss"].append(avg_train_loss)

        if not torch.isfinite(torch.tensor(avg_train_loss)) or avg_train_loss > 1e6:
            diverged = True
            print(f"AVISO: TTN divergiu na epoch {epoch} (train_loss={avg_train_loss:.2e}).")
            break

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for x_t, x_next in val_data:
                pred = model(x_t)
                val_loss += nn.functional.mse_loss(pred, x_next).item()
            val_loss /= max(len(val_data), 1)
            history["val_loss"].append(val_loss)
            if val_loss < best_val_mse - 1e-12:
                best_val_mse = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    return {"history": history, "best_val_mse": best_val_mse, "diverged": diverged}
