import torch
import numpy as np
import networkx as nx
import pandas as pd
from cytos import TTNvsGNN

if __name__ == "__main__":
    # carrega expressao alinhada
    df = pd.read_csv("spellman_cycling_string_aligned.csv")
    gene_names = list(df.columns)
    expr = df.values.astype(np.float32)  # (18, 451)

    print(f"Expressao: {expr.shape} (timepoints x genes)")
    print(f"Genes: {len(gene_names)}, primeiros: {gene_names[:5]}")

    # split temporal contíguo (pre-registrado): 0-10 treino, 11-14 val, 15-17 teste
    splits = [expr[:11], expr[11:15], expr[14:]]  # 11, 4, 4 timepoints

    # cada "trajetoria" e um segmento contíguo - formato: lista de arrays (n_time, n_genes)
    trajectories = [expr]  # uma unica serie - split feito manualmente abaixo

    # split manual por indice temporal (nao por trajetoria inteira)
    x_train = expr[:11]
    x_val = expr[11:15]
    x_test = expr[14:]  # 4 timepoints de teste

    # par (t, t+1) para cada segmento
    def make_pairs(segment):
        xs = segment[:-1]
        x_nexts = segment[1:]
        return xs, x_nexts

    x_train_t, x_train_next_t = [torch.tensor(a) for a in make_pairs(x_train)]
    x_val_t, x_val_next_t = [torch.tensor(a) for a in make_pairs(x_val)]
    x_test_t, x_test_next_t = [torch.tensor(a) for a in make_pairs(x_test)]

    print(f"\nPares de treino: {x_train_t.shape[0]}, val: {x_val_t.shape[0]}, teste: {x_test_t.shape[0]}")

    # carrega subgrafo STRING (reconstruido dos genes alinhados)
    from cytos.datasets import load_string_network, load_alias_to_preferred_name, rename_genes_via_aliases
    graph_full = load_string_network(
        links_path="4932.protein.links.v12.0.txt",
        info_path="4932.protein.info.v12.0.txt",
        confidence_threshold=900,
    )
    subgraph = graph_full.subgraph(gene_names).copy()
    print(f"Subgrafo: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")

    # instancia TTNvsGNN diretamente com os dados pre-splitados
    from cytos.experiment import TTNvsGNN as _TTNvsGNN, match_parameter_counts, _run_one_seed
    from cytos.data import detect_hierarchy
    import os, sys, time
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from scipy.stats import wilcoxon

    hierarchy = detect_hierarchy(subgraph, method="louvain")
    node_to_idx = {n: i for i, n in enumerate(gene_names)}
    edges = [(node_to_idx[u], node_to_idx[v]) for u, v in subgraph.edges() if u in node_to_idx and v in node_to_idx]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous() if edges else torch.zeros((2,0), dtype=torch.long)

    gnn_hidden_dim, bond_dim, gnn_params, ttn_params = match_parameter_counts(
        hierarchy, gene_names, num_nodes=len(gene_names)
    )
    print(f"\nParametros: GNN={gnn_params} (hidden_dim={gnn_hidden_dim}), TTN={ttn_params} (bond_dim={bond_dim})")

    seed_list = list(range(20))
    n_workers = min(len(seed_list), os.cpu_count() or 4)

    train_cfg = {
        "gnn_architecture": "GCN", "gnn_num_layers": 2,
        "lr": 0.001, "weight_decay": 1e-5, "epochs": 200,
        "patience": 15, "batch_size": 16,
    }

    seed_args = [
        {
            "seed": seed, "gene_names": gene_names, "hierarchy": hierarchy,
            "edge_index": edge_index, "gnn_hidden_dim": gnn_hidden_dim,
            "bond_dim": bond_dim, "train_cfg": train_cfg,
            "x_train": x_train_t.numpy(), "x_train_next": x_train_next_t.numpy(),
            "x_val": x_val_t.numpy(), "x_val_next": x_val_next_t.numpy(),
            "x_test": x_test_t.numpy(), "x_test_next": x_test_next_t.numpy(),
        }
        for seed in seed_list
    ]

    print(f"\nRodando {len(seed_list)} seeds em paralelo ({n_workers} workers)...")
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
                print(f"\nAVISO seed {seed}: {e}")
                continue
            n_done = len(seed_results)
            elapsed = time.time() - start_time
            eta = (elapsed / n_done) * (len(seed_list) - n_done) if n_done > 0 else 0
            bar = "#" * int(30 * n_done / len(seed_list)) + "-" * (30 - int(30 * n_done / len(seed_list)))
            sys.stdout.write(f"\r[{bar}] {n_done}/{len(seed_list)} seeds | {elapsed:.1f}s | ETA {eta:.1f}s   ")
            sys.stdout.flush()
    print()

    ttn_mses = [r["ttn_mse_per_param"] for r in seed_results]
    gnn_mses = [r["gnn_mse_per_param"] for r in seed_results]
    ttn_lrs = [c for r in seed_results for c in r["ttn_lr_corrs"] if not np.isnan(c)]
    gnn_lrs = [c for r in seed_results for c in r["gnn_lr_corrs"] if not np.isnan(c)]

    _, p_h1 = wilcoxon(ttn_mses, gnn_mses)
    _, p_h1b = wilcoxon(ttn_lrs[:len(gnn_lrs)], gnn_lrs)

    h1_pass = bool(p_h1 < 0.05 and np.mean(ttn_mses) < np.mean(gnn_mses))
    h1b_pass = bool(p_h1b < 0.05 and np.mean(ttn_lrs) > np.mean(gnn_lrs))

    print(f"\n=== Resultado Fase I (dados reais) ===")
    print(f"TTN MSE/param: {np.mean(ttn_mses):.3e} | GNN MSE/param: {np.mean(gnn_mses):.3e}")
    print(f"H1-real: {'PASSOU' if h1_pass else 'FALHOU'} (p={p_h1:.3e})")
    print(f"TTN long-range: {np.mean(ttn_lrs):.4f} | GNN long-range: {np.mean(gnn_lrs):.4f}")
    print(f"H1b-real: {'PASSOU' if h1b_pass else 'FALHOU'} (p={p_h1b:.3e})")
