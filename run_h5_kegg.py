"""
run_h5_kegg.py

H5 (Secao 31 do pre-registro): TTN com hierarquia biologica (KEGG
pathways) vs GNN parametro-casada, em dados reais (levedura, 223 genes
com pathway conhecido).
"""

import torch
import numpy as np
from scipy.stats import wilcoxon

from cytos.kegg_hierarchy import load_kegg_gene_pathway_map, build_kegg_hierarchy
from cytos.datasets import load_alias_to_preferred_name, load_string_network
from cytos.data import split_trajectories_by_fraction, trajectories_to_stacked_arrays
from cytos.experiment import match_parameter_counts, _run_one_seed
from concurrent.futures import ProcessPoolExecutor, as_completed
import os, sys, time
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("spellman_kegg_restricted.csv")
    gene_names = list(df.columns)
    expr = df.values.astype(np.float32)
    n_genes = len(gene_names)
    print(f"Genes: {n_genes}")

    gene_to_pathways_systematic = load_kegg_gene_pathway_map("sce_gene_pathway.txt")
    alias_map = load_alias_to_preferred_name("4932.protein.info.v12.0.txt", "4932.protein.aliases.v12.0.txt")
    gene_to_pathways = {alias_map.get(k, k): v for k, v in gene_to_pathways_systematic.items()}
    hierarchy = build_kegg_hierarchy(gene_names, gene_to_pathways)

    graph_full = load_string_network("4932.protein.links.v12.0.txt", "4932.protein.info.v12.0.txt", confidence_threshold=700)
    subgraph = graph_full.subgraph(gene_names).copy()
    print(f"Subgrafo: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")

    node_to_idx = {n: i for i, n in enumerate(gene_names)}
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in subgraph.edges() if u in node_to_idx and v in node_to_idx]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    gnn_hidden_dim, bond_dim, gnn_params, ttn_params = match_parameter_counts(hierarchy, gene_names, num_nodes=n_genes)
    print(f"GNN params={gnn_params} (hidden={gnn_hidden_dim}), TTN params={ttn_params} (bond={bond_dim})")

    # split temporal contiguo (mesma decisao da Fase I original, Secao 22-23)
    x_train, x_val, x_test = expr[:11], expr[11:15], expr[14:]
    def pairs(s): return torch.tensor(s[:-1]), torch.tensor(s[1:])
    x_train_t, x_train_next_t = pairs(x_train)
    x_val_t,   x_val_next_t   = pairs(x_val)
    x_test_t,  x_test_next_t  = pairs(x_test)

    train_cfg = {"gnn_architecture":"GCN","gnn_num_layers":2,"lr":0.001,
                 "weight_decay":1e-5,"epochs":200,"patience":15,"batch_size":16}
    seed_list = list(range(20))
    n_workers = min(len(seed_list), os.cpu_count() or 4)

    seed_args = [{"seed":s,"gene_names":gene_names,"hierarchy":hierarchy,
                  "edge_index":edge_index,"gnn_hidden_dim":gnn_hidden_dim,
                  "bond_dim":bond_dim,"train_cfg":train_cfg,
                  "x_train":x_train_t.numpy(),"x_train_next":x_train_next_t.numpy(),
                  "x_val":x_val_t.numpy(),"x_val_next":x_val_next_t.numpy(),
                  "x_test":x_test_t.numpy(),"x_test_next":x_test_next_t.numpy()}
                 for s in seed_list]

    seed_results = []
    start_time = time.time()
    with ProcessPoolExecutor(max_workers=n_workers, max_tasks_per_child=4) as executor:
        futures = {executor.submit(_run_one_seed, args): args["seed"] for args in seed_args}
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
            bar = "#"*int(30*n_done/len(seed_list)) + "-"*(30-int(30*n_done/len(seed_list)))
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

    print(f"\n=== Resultado H5 (hierarquia KEGG, 223 genes) ===")
    print(f"GNN params={gnn_params}, TTN params={ttn_params} (diff={abs(ttn_params-gnn_params)/gnn_params:.1%})")
    print(f"H1 (KEGG):  {'PASSOU' if h1_pass  else 'FALHOU'} | TTN={np.mean(ttn_mses):.3e}, GNN={np.mean(gnn_mses):.3e}, p={p_h1:.3e}")
    print(f"H1b (KEGG): {'PASSOU' if h1b_pass else 'FALHOU'} | TTN lr={np.mean(ttn_lrs):.4f}, GNN lr={np.mean(gnn_lrs):.4f}, p={p_h1b:.3e}")
