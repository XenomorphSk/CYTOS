"""
cytos/experiment.py

API principal da biblioteca: compara Tree Tensor Network (TTN) com
Graph Neural Network (GNN), em qualquer grafo + trajetorias fornecidos
pelo usuario (nao especifico ao DREAM4).

Uso basico:
    from cytos import TTNvsGNN
    result = TTNvsGNN(graph=meu_grafo, trajectories=minhas_trajetorias).run(seeds=20)
    print(result.summary())
"""

from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import torch
from scipy.stats import wilcoxon

from cytos.data import detect_hierarchy, split_trajectories_by_fraction, trajectories_to_stacked_arrays
from cytos.gnn_baseline import GNNBaseline, make_batched_edge_index, train_gnn
from cytos.ttn_model import TTNModel, train_ttn


def match_parameter_counts(hierarchy, gene_names, num_nodes, gnn_architecture="GCN", gnn_num_layers=2, tolerance=0.10):
    hidden_dim_candidates = [
        2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 144, 160, 176, 192, 224, 256,
        320, 384, 448, 512, 640, 768, 896, 1024, 1280, 1536, 2048, 2560, 3072, 4096,
    ]
    bond_dim_candidates = list(range(2, 65))

    best = None
    for hidden_dim in hidden_dim_candidates:
        gnn_probe = GNNBaseline(num_nodes=num_nodes, hidden_dim=hidden_dim, num_layers=gnn_num_layers, architecture=gnn_architecture)
        gnn_params = gnn_probe.count_parameters()

        for bond_dim in bond_dim_candidates:
            ttn_probe = TTNModel(hierarchy=hierarchy, gene_names=gene_names, bond_dim=bond_dim)
            ttn_params = ttn_probe.count_parameters()
            diff_ratio = abs(ttn_params - gnn_params) / max(gnn_params, 1)
            if best is None or diff_ratio < best[0]:
                best = (diff_ratio, hidden_dim, bond_dim, gnn_params, ttn_params)
            if diff_ratio <= tolerance:
                break
        if best is not None and best[0] <= tolerance:
            break

    diff_ratio, hidden_dim, bond_dim, gnn_params, ttn_params = best
    if diff_ratio > tolerance:
        print(f"AVISO: nao foi possivel casar parametros dentro da tolerancia ({diff_ratio:.2%} > {tolerance:.2%}). Melhor par: hidden_dim={hidden_dim} (GNN, {gnn_params} params), bond_dim={bond_dim} (TTN, {ttn_params} params).")
    return hidden_dim, bond_dim, gnn_params, ttn_params


def long_range_correlation(pred, true, hierarchy, gene_names):
    pred = np.asarray(pred).flatten()
    true = np.asarray(true).flatten()
    partition = hierarchy["level_0"]
    pairs = [(i, j) for i, gi in enumerate(gene_names) for j, gj in enumerate(gene_names) if i < j and partition.get(gi, -1) != partition.get(gj, -1)]
    if not pairs:
        return float("nan")
    pred_diffs = np.array([pred[i] - pred[j] for i, j in pairs])
    true_diffs = np.array([true[i] - true[j] for i, j in pairs])
    if np.std(pred_diffs) == 0 or np.std(true_diffs) == 0:
        return float("nan")
    return float(np.corrcoef(pred_diffs, true_diffs)[0, 1])


def _run_one_seed(args):
    torch.set_num_threads(1)

    seed = args["seed"]
    gene_names = args["gene_names"]
    hierarchy = args["hierarchy"]
    edge_index = args["edge_index"]
    gnn_hidden_dim = args["gnn_hidden_dim"]
    bond_dim = args["bond_dim"]
    train_cfg = args["train_cfg"]
    num_nodes = len(gene_names)

    x_train = torch.tensor(args["x_train"])
    x_train_next = torch.tensor(args["x_train_next"])
    x_val = torch.tensor(args["x_val"])
    x_val_next = torch.tensor(args["x_val_next"])
    x_test = torch.tensor(args["x_test"])
    x_test_next = torch.tensor(args["x_test_next"])

    gnn = GNNBaseline(num_nodes=num_nodes, hidden_dim=gnn_hidden_dim, num_layers=train_cfg["gnn_num_layers"], architecture=train_cfg["gnn_architecture"])
    gnn_result = train_gnn(
        gnn, x_train.unsqueeze(-1), x_train_next.unsqueeze(-1), x_val.unsqueeze(-1), x_val_next.unsqueeze(-1),
        edge_index=edge_index, num_nodes=num_nodes,
        lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"], epochs=train_cfg["epochs"],
        seed=seed, patience=train_cfg["patience"], batch_size=train_cfg["batch_size"],
    )
    gnn_mse_per_param = gnn_result["best_val_mse"] / gnn.count_parameters()

    ttn = TTNModel(hierarchy=hierarchy, gene_names=gene_names, bond_dim=bond_dim)
    ttn_result = train_ttn(
        ttn, x_train, x_train_next, x_val, x_val_next,
        lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"], epochs=train_cfg["epochs"],
        seed=seed, patience=train_cfg["patience"], batch_size=train_cfg["batch_size"],
    )
    ttn_mse_per_param = ttn_result["best_val_mse"] / ttn.count_parameters()

    n_test = x_test.shape[0]
    with torch.no_grad():
        gnn.eval()
        ttn.eval()
        batched_edge_test = make_batched_edge_index(edge_index, num_nodes, n_test)
        gnn_preds = gnn(x_test.unsqueeze(-1).reshape(-1, 1), batched_edge_test).reshape(n_test, num_nodes).numpy()
        ttn_preds = ttn(x_test).numpy()
        trues = x_test_next.numpy()

        gnn_lr_corrs = [long_range_correlation(gnn_preds[i], trues[i], hierarchy, gene_names) for i in range(n_test)]
        ttn_lr_corrs = [long_range_correlation(ttn_preds[i], trues[i], hierarchy, gene_names) for i in range(n_test)]

    return {
        "seed": seed,
        "gnn_mse_per_param": gnn_mse_per_param, "ttn_mse_per_param": ttn_mse_per_param,
        "gnn_lr_corrs": gnn_lr_corrs, "ttn_lr_corrs": ttn_lr_corrs,
        "gnn_diverged": gnn_result["diverged"], "ttn_diverged": ttn_result["diverged"],
    }


@dataclass
class TTNvsGNNResult:
    h1_pass: bool
    h1b_pass: bool
    p_primary: float
    p_secondary: float
    ttn_mse_per_param_mean: float
    gnn_mse_per_param_mean: float
    ttn_long_range_corr_mean: float
    gnn_long_range_corr_mean: float
    gnn_params: int
    ttn_params: int
    param_diff_ratio: float
    n_seeds: int
    n_gnn_diverged: int
    n_ttn_diverged: int
    _experiment: "TTNvsGNN" = field(repr=False)

    def summary(self) -> str:
        lines = [
            "=== Resultado: TTN vs GNN ===",
            f"Parametros: GNN={self.gnn_params}, TTN={self.ttn_params} (diff={self.param_diff_ratio:.1%})",
            f"H1 (eficiencia parametrica): {'PASSOU' if self.h1_pass else 'FALHOU'} "
            f"(MSE/param TTN={self.ttn_mse_per_param_mean:.3e}, GNN={self.gnn_mse_per_param_mean:.3e}, p={self.p_primary:.3e})",
            f"H1b (correlacao de longo alcance): {'PASSOU' if self.h1b_pass else 'FALHOU'} "
            f"(corr TTN={self.ttn_long_range_corr_mean:.3f}, GNN={self.gnn_long_range_corr_mean:.3f}, p={self.p_secondary:.3e})",
        ]
        if self.n_gnn_diverged or self.n_ttn_diverged:
            lines.append(f"AVISO: GNN divergiu em {self.n_gnn_diverged}/{self.n_seeds} seeds, TTN em {self.n_ttn_diverged}/{self.n_seeds} seeds.")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}

    def to_dataframe(self):
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas nao instalado. Use result.to_dict() em vez disso, ou: pip install pandas")
        return pd.DataFrame([self.to_dict()])

    def entanglement_pilot(self, seeds: list = None):
        from cytos.entanglement import EntanglementPilot
        return EntanglementPilot(self._experiment).run(seeds=seeds)


class TTNvsGNN:
    """
    Compara TTN com GNN em um grafo regulatorio e trajetorias fornecidos
    pelo usuario. Nao especifico ao DREAM4 - qualquer grafo dirigido
    (networkx.DiGraph) e trajetorias (lista de arrays numpy, cada um de
    shape (n_timepoints, n_nodes)) servem como entrada.
    """

    def __init__(
        self,
        graph: nx.DiGraph,
        trajectories: list,
        gene_names: list = None,
        train_frac: float = 0.6,
        val_frac: float = 0.2,
        test_frac: float = 0.2,
        clustering_method: str = "louvain",
        gnn_architecture: str = "GCN",
        gnn_num_layers: int = 2,
        param_match_tolerance: float = 0.10,
        significance_alpha: float = 0.05,
    ):
        self.graph = graph
        self.gene_names = gene_names if gene_names is not None else list(graph.nodes())
        self.hierarchy = detect_hierarchy(graph, method=clustering_method)
        self.gnn_architecture = gnn_architecture
        self.gnn_num_layers = gnn_num_layers
        self.param_match_tolerance = param_match_tolerance
        self.significance_alpha = significance_alpha

        train_trajs, val_trajs, test_trajs = split_trajectories_by_fraction(trajectories, train_frac, val_frac, test_frac)
        self.x_train, self.x_train_next = trajectories_to_stacked_arrays(train_trajs)
        self.x_val, self.x_val_next = trajectories_to_stacked_arrays(val_trajs)
        self.x_test, self.x_test_next = trajectories_to_stacked_arrays(test_trajs)

        node_to_idx = {n: i for i, n in enumerate(self.gene_names)}
        edges = [(node_to_idx[u], node_to_idx[v]) for u, v in graph.edges() if u in node_to_idx and v in node_to_idx]
        self.edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        self.gnn_hidden_dim, self.bond_dim, self.gnn_params, self.ttn_params = match_parameter_counts(
            self.hierarchy, self.gene_names, num_nodes=len(self.gene_names),
            gnn_architecture=gnn_architecture, gnn_num_layers=gnn_num_layers, tolerance=param_match_tolerance,
        )

    def run(self, seeds=20, n_workers=None, epochs=200, batch_size=16, patience=15, lr=0.001, weight_decay=1e-5, show_progress=True):
        seed_list = list(range(seeds)) if isinstance(seeds, int) else list(seeds)
        n_workers = n_workers or min(len(seed_list), os.cpu_count() or 4)

        train_cfg = {
            "gnn_architecture": self.gnn_architecture, "gnn_num_layers": self.gnn_num_layers,
            "lr": lr, "weight_decay": weight_decay, "epochs": epochs,
            "patience": patience, "batch_size": batch_size,
        }

        seed_args = [
            {
                "seed": seed, "gene_names": self.gene_names, "hierarchy": self.hierarchy,
                "edge_index": self.edge_index, "gnn_hidden_dim": self.gnn_hidden_dim,
                "bond_dim": self.bond_dim, "train_cfg": train_cfg,
                "x_train": self.x_train, "x_train_next": self.x_train_next,
                "x_val": self.x_val, "x_val_next": self.x_val_next,
                "x_test": self.x_test, "x_test_next": self.x_test_next,
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
                    sys.stdout.write(f"\r[{bar}] {n_done}/{n_total} seeds | {elapsed:.1f}s | ETA {eta:.1f}s   ")
                    sys.stdout.flush()
        if show_progress:
            print()

        gnn_mses = [r["gnn_mse_per_param"] for r in seed_results]
        ttn_mses = [r["ttn_mse_per_param"] for r in seed_results]
        gnn_lr_corrs = [c for r in seed_results for c in r["gnn_lr_corrs"]]
        ttn_lr_corrs = [c for r in seed_results for c in r["ttn_lr_corrs"]]
        n_gnn_diverged = sum(r["gnn_diverged"] for r in seed_results)
        n_ttn_diverged = sum(r["ttn_diverged"] for r in seed_results)

        stat_p, p_primary = wilcoxon(ttn_mses, gnn_mses)
        gnn_lr_clean = [v for v in gnn_lr_corrs if not np.isnan(v)]
        ttn_lr_clean = [v for v in ttn_lr_corrs if not np.isnan(v)]
        if len(gnn_lr_clean) == len(ttn_lr_clean) and gnn_lr_clean:
            stat_s, p_secondary = wilcoxon(ttn_lr_clean, gnn_lr_clean)
        else:
            p_secondary = float("nan")

        alpha = self.significance_alpha
        return TTNvsGNNResult(
            h1_pass=bool(p_primary < alpha and np.mean(ttn_mses) < np.mean(gnn_mses)),
            h1b_pass=bool(not np.isnan(p_secondary) and p_secondary < alpha and np.mean(ttn_lr_clean) > np.mean(gnn_lr_clean)),
            p_primary=float(p_primary), p_secondary=float(p_secondary),
            ttn_mse_per_param_mean=float(np.mean(ttn_mses)), gnn_mse_per_param_mean=float(np.mean(gnn_mses)),
            ttn_long_range_corr_mean=float(np.mean(ttn_lr_clean)) if ttn_lr_clean else float("nan"),
            gnn_long_range_corr_mean=float(np.mean(gnn_lr_clean)) if gnn_lr_clean else float("nan"),
            gnn_params=self.gnn_params, ttn_params=self.ttn_params,
            param_diff_ratio=abs(self.ttn_params - self.gnn_params) / max(self.gnn_params, 1),
            n_seeds=len(seed_results), n_gnn_diverged=n_gnn_diverged, n_ttn_diverged=n_ttn_diverged,
            _experiment=self,
        )
