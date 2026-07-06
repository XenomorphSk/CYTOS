"""
run_h10_sos_temporal.py

H10 (Secao 41): TTN com hierarquia biologica SOS vs GNN parametro-casada,
em series temporais reais (Ronen 2002, 4 experimentos x 50 timepoints).
Primeira vez com dados temporais genuinos no sistema SOS.
"""

import torch
import numpy as np
import networkx as nx
from scipy.stats import wilcoxon
from concurrent.futures import ProcessPoolExecutor, as_completed
import os, sys, time

from cytos.experiment import match_parameter_counts, _run_one_seed


def load_sos_trajectories(exp_dir="sosdata/SOSData"):
    gene_names = ["uvrD", "lexA", "umuD", "recA", "uvrA", "uvrY", "ruvA", "polB"]
    trajectories = []
    for exp_file in ["Exp1.txt", "Exp2.txt", "Exp3.txt", "Exp4.txt"]:
        path = os.path.join(exp_dir, exp_file)
        with open(path) as f:
            lines = f.readlines()
        expr = []
        for line in lines[1:]:
            parts = line.strip().split("\t")
            values = [float(v) for v in parts[1:]]
            expr.append(values)
        traj = np.array(expr, dtype=np.float32).T  # (50, 8)
        trajectories.append(traj)
    return trajectories, gene_names


def normalize_trajectory(traj):
    """Z-score por gene (pre-registrado, Secao 41)."""
    mean = traj.mean(axis=0, keepdims=True)
    std  = traj.std(axis=0, keepdims=True) + 1e-8
    return (traj - mean) / std


def traj_to_pairs(traj):
    x      = torch.tensor(traj[:-1])
    x_next = torch.tensor(traj[1:])
    return x, x_next


if __name__ == "__main__":
    trajectories, gene_names = load_sos_trajectories()
    n_genes = len(gene_names)
    print(f"Dados: {len(trajectories)} trajetorias x {trajectories[0].shape[0]} timepoints x {n_genes} genes")

    # normalizacao z-score por gene por trajetoria (pre-registrado)
    trajectories = [normalize_trajectory(t) for t in trajectories]

    # split cross-UV (pre-registrado): Exp1+2 treino, Exp3 val, Exp4 teste
    x_train = np.concatenate([trajectories[0][:-1], trajectories[1][:-1]], axis=0)
    x_train_next = np.concatenate([trajectories[0][1:], trajectories[1][1:]], axis=0)
    x_val,   x_val_next   = trajectories[2][:-1], trajectories[2][1:]
    x_test,  x_test_next  = trajectories[3][:-1], trajectories[3][1:]
    print(f"Pares: treino={len(x_train)}, val={len(x_val)}, teste={len(x_test)}")

    # hierarquia biologica SOS (pre-registrada, Secao 41)
    partition = {
        "uvrD": 0, "uvrA": 0, "ruvA": 0,  # reparacao NER
        "recA": 1, "lexA": 1,               # reguladores centrais
        "umuD": 2, "polB": 2,               # mutagenese/tolerancia
        "uvrY": 3,                           # singleton regulador geral
    }
    hierarchy = {"level_0": partition}

    # grafo de interacoes conhecidas (gold standard SOS)
    known_edges = [
        ("lexA","uvrD"),("lexA","umuD"),("lexA","recA"),("lexA","uvrA"),
        ("lexA","uvrY"),("lexA","ruvA"),("lexA","polB"),("lexA","lexA"),
        ("recA","lexA"),("recA","umuD"),("recA","recA"),
        ("umuD","recA"),("uvrD","lexA"),
    ]
    graph = nx.DiGraph()
    graph.add_nodes_from(gene_names)
    for u, v in known_edges:
        if u in gene_names and v in gene_names:
            graph.add_edge(u, v)
    node_to_idx = {g: i for i, g in enumerate(gene_names)}
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in graph.edges()]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    print(f"Grafo SOS: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} arestas")

    gnn_hidden_dim, bond_dim, gnn_params, ttn_params = match_parameter_counts(
        hierarchy, gene_names, num_nodes=n_genes)
    diff = abs(ttn_params - gnn_params) / max(gnn_params, 1)
    print(f"GNN={gnn_params} (hidden={gnn_hidden_dim}), TTN={ttn_params} (bond={bond_dim}), diff={diff:.1%}")

    train_cfg = {"gnn_architecture":"GCN","gnn_num_layers":2,"lr":0.001,
                 "weight_decay":1e-5,"epochs":200,"patience":15,"batch_size":16}
    seed_list = list(range(20))
    n_workers = min(len(seed_list), os.cpu_count() or 4)

    seed_args = [
        {"seed":s,"gene_names":gene_names,"hierarchy":hierarchy,
         "edge_index":edge_index,"gnn_hidden_dim":gnn_hidden_dim,
         "bond_dim":bond_dim,"train_cfg":train_cfg,
         "x_train":x_train,"x_train_next":x_train_next,
         "x_val":x_val,"x_val_next":x_val_next,
         "x_test":x_test,"x_test_next":x_test_next}
        for s in seed_list
    ]

    seed_results = []
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=n_workers, max_tasks_per_child=4) as executor:
        futures = {executor.submit(_run_one_seed, a): a["seed"] for a in seed_args}
        for future in as_completed(futures):
            seed = futures[future]
            try:
                seed_results.append(future.result())
            except Exception as e:
                print(f"\nAVISO seed {seed}: {e}")
                continue
            n_done = len(seed_results)
            elapsed = time.time() - start_time
            eta = (elapsed/n_done)*(len(seed_list)-n_done) if n_done>0 else 0
            bar = "#"*int(30*n_done/len(seed_list))+"-"*(30-int(30*n_done/len(seed_list)))
            sys.stdout.write(f"\r[{bar}] {n_done}/{len(seed_list)} | {elapsed:.1f}s | ETA {eta:.1f}s   ")
            sys.stdout.flush()
    print()

    ttn_mses = [r["ttn_mse_per_param"] for r in seed_results]
    gnn_mses = [r["gnn_mse_per_param"] for r in seed_results]
    ttn_lrs  = [c for r in seed_results for c in r["ttn_lr_corrs"] if not np.isnan(c)]
    gnn_lrs  = [c for r in seed_results for c in r["gnn_lr_corrs"] if not np.isnan(c)]

    _, p_h1  = wilcoxon(ttn_mses, gnn_mses)
    n = min(len(ttn_lrs), len(gnn_lrs))
    _, p_h1b = wilcoxon(ttn_lrs[:n], gnn_lrs[:n]) if n > 0 else (None, float("nan"))

    h1_pass  = bool(p_h1  < 0.05 and np.mean(ttn_mses) < np.mean(gnn_mses))
    h1b_pass = bool(p_h1b < 0.05 and np.mean(ttn_lrs)  > np.mean(gnn_lrs))

    print(f"\n=== Resultado H10 (SOS Ronen 2002, serie temporal genuina) ===")
    print(f"H1:  {'PASSOU' if h1_pass  else 'FALHOU'} | TTN={np.mean(ttn_mses):.3e}, GNN={np.mean(gnn_mses):.3e}, p={p_h1:.3e}")
    print(f"H1b: {'PASSOU' if h1b_pass else 'FALHOU'} | TTN lr={np.mean(ttn_lrs):.4f}, GNN lr={np.mean(gnn_lrs):.4f}, p={p_h1b:.3e}")
