"""
run_h9_sos.py

H9 (Secao 39): TTN com hierarquia biologica (modulos SOS experimentalmente
estabelecidos) vs GNN parametro-casada, em dados reais de E. coli SOS2.
"""

import torch
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import wilcoxon
from concurrent.futures import ProcessPoolExecutor, as_completed
import os, sys, time

from cytos.experiment import match_parameter_counts, _run_one_seed

if __name__ == "__main__":
    df = pd.read_csv("sos2_expression.csv", index_col=0)
    gene_names = list(df.columns)
    expr = df.values.astype(np.float32)
    n_genes = len(gene_names)
    print(f"SOS2: {expr.shape} ({n_genes} genes)")

    # hierarquia pre-registrada: dois modulos funcionais SOS
    partition = {
        "recA": 0, "lexA": 0, "recF": 0, "dinI": 0, "umuDC": 0,  # modulo reparacao
        "rpoD": 1, "rpoH": 1, "rpoS": 1,  # modulo fatores sigma
        "ssb": 2,   # conector singleton
    }
    hierarchy = {"level_0": partition}
    print(f"Comunidades: {set(partition.values())}")

    # grafo de interacoes conhecidas (gold standard SOS, 43 arestas da literatura)
    known_edges = [
        ("recA","lexA"),("lexA","recA"),("recA","recF"),("recA","dinI"),
        ("recA","umuDC"),("lexA","recF"),("lexA","dinI"),("lexA","umuDC"),
        ("lexA","ssb"),("ssb","recA"),("recF","recA"),("dinI","recA"),
        ("umuDC","recA"),("rpoD","recA"),("rpoD","lexA"),("rpoD","ssb"),
        ("rpoD","recF"),("rpoD","dinI"),("rpoD","umuDC"),("rpoH","recA"),
        ("rpoH","ssb"),("rpoS","recA"),("rpoS","ssb"),
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

    # split 60/20/20 por PERFIL (pre-registrado, seed=0)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(expr))
    n_train = int(len(expr) * 0.6)
    n_val   = int(len(expr) * 0.2)
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train+n_val]
    test_idx  = idx[n_train+n_val:]

    x_train = expr[train_idx]
    x_val   = expr[val_idx]
    x_test  = expr[test_idx]

    # pares cross-condicao: embaralha e pareia (i, i+1)
    def make_pairs(X, seed):
        rng2 = np.random.default_rng(seed)
        perm = rng2.permutation(len(X))
        X_s = X[perm]
        n = (len(X_s) // 2) * 2
        return X_s[:n//2], X_s[n//2:n]

    x_tr, x_tr_next = make_pairs(x_train, seed=1)
    x_va, x_va_next = make_pairs(x_val,   seed=2)
    x_te, x_te_next = make_pairs(x_test,  seed=3)
    print(f"Pares: treino={len(x_tr)}, val={len(x_va)}, teste={len(x_te)}")

    gnn_hidden_dim, bond_dim, gnn_params, ttn_params = match_parameter_counts(
        hierarchy, gene_names, num_nodes=n_genes)
    diff = abs(ttn_params-gnn_params)/max(gnn_params,1)
    print(f"GNN={gnn_params} (hidden={gnn_hidden_dim}), TTN={ttn_params} (bond={bond_dim}), diff={diff:.1%}")

    train_cfg = {"gnn_architecture":"GCN","gnn_num_layers":2,"lr":0.001,
                 "weight_decay":1e-5,"epochs":200,"patience":15,"batch_size":16}
    seed_list = list(range(20))
    n_workers = min(len(seed_list), os.cpu_count() or 4)

    seed_args = [
        {"seed":s,"gene_names":gene_names,"hierarchy":hierarchy,
         "edge_index":edge_index,"gnn_hidden_dim":gnn_hidden_dim,
         "bond_dim":bond_dim,"train_cfg":train_cfg,
         "x_train":x_tr,"x_train_next":x_tr_next,
         "x_val":x_va,"x_val_next":x_va_next,
         "x_test":x_te,"x_test_next":x_te_next}
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
            elapsed = time.time()-start_time
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
    _, p_h1b = wilcoxon(ttn_lrs[:n], gnn_lrs[:n]) if n>0 else (None, float("nan"))

    h1_pass  = bool(p_h1  < 0.05 and np.mean(ttn_mses) < np.mean(gnn_mses))
    h1b_pass = bool(p_h1b < 0.05 and np.mean(ttn_lrs)  > np.mean(gnn_lrs))

    print(f"\n=== Resultado H9 (SOS E. coli, hierarquia biologica) ===")
    print(f"H1:  {'PASSOU' if h1_pass  else 'FALHOU'} | TTN={np.mean(ttn_mses):.3e}, GNN={np.mean(gnn_mses):.3e}, p={p_h1:.3e}")
    print(f"H1b: {'PASSOU' if h1b_pass else 'FALHOU'} | TTN lr={np.mean(ttn_lrs):.4f}, GNN lr={np.mean(gnn_lrs):.4f}, p={p_h1b:.3e}")
