"""
debug_canonicalization.py

Diagnostico passo-a-passo.
"""

import copy
import torch

from cytos.datasets import load_dream4
from cytos.data import split_trajectories_by_fraction, trajectories_to_stacked_arrays, detect_hierarchy
from cytos.ttn_model import TTNModel, train_ttn
from cytos.canonicalization import _build_consumer_map, _ref_dim
from cytos.ttn_model import LEAF_DIM


def diagnose(model, comm_id, x_test):
    model = copy.deepcopy(model)
    model.eval()

    with torch.no_grad():
        output_before_anything = model(x_test).clone()

    plan, root_ref = model.community_plans[comm_id]
    consumer_of = _build_consumer_map(plan, comm_id, model.root_plan)

    gene_leaf_dim = lambda _g: LEAF_DIM
    root_leaf_dim = lambda _c: model.bond_dim

    print(f"Comunidade {comm_id}: plan tem {len(plan)} nos")
    print(f"consumer_of = {consumer_of}")

    for i, (left_ref, right_ref, module_idx) in enumerate(plan):
        linear = model.contractions[module_idx]
        W = linear.weight.data.clone()
        Wt = W.t()
        Q, R = torch.linalg.qr(Wt, mode="reduced")

        print(f"\n--- Processando no {i} (module_idx={module_idx}) ---")
        print(f"  W.shape={W.shape}, Q.shape={Q.shape}, R.shape={R.shape}")
        print(f"  Q@R == Wt? {torch.allclose(Q @ R, Wt, atol=1e-5)}")
        print(f"  R.t()@Q.t() == W? {torch.allclose(R.t() @ Q.t(), W, atol=1e-5)}")

        linear.weight.data = Q.t().contiguous()

        with torch.no_grad():
            output_after_qr_only = model(x_test).clone()
        diff_qr_only = (output_before_anything - output_after_qr_only).abs().max().item()
        print(f"  Apos so trocar W por Q (SEM absorver R ainda): diff_max={diff_qr_only:.6f}")

        if i not in consumer_of:
            print(f"  No {i} nao tem consumidor mapeado - pulando absorcao")
            continue

        location, j, side = consumer_of[i]
        print(f"  Consumidor: location={location}, j={j}, side={side}")

        if location == "community":
            c_left_ref, c_right_ref, c_module_idx = plan[j]
        else:
            c_left_ref, c_right_ref, c_module_idx = model.root_plan[j]

        print(f"  c_left_ref={c_left_ref}, c_right_ref={c_right_ref}, no_atual=('node',{i})")
        side_correto = "left" if c_left_ref == ("node", i) else ("right" if c_right_ref == ("node", i) else "NENHUM - BUG!")
        print(f"  side calculado='{side}', side correto pelos refs='{side_correto}'")

        c_linear = model.contractions[c_module_idx]
        c_out_dim = c_linear.weight.shape[0]
        leaf_dim_fn = gene_leaf_dim if location == "community" else root_leaf_dim
        left_dim = _ref_dim(c_left_ref, leaf_dim_fn, model.bond_dim)
        right_dim = _ref_dim(c_right_ref, leaf_dim_fn, model.bond_dim)
        print(f"  Consumidor module_idx={c_module_idx}, c_out_dim={c_out_dim}, left_dim={left_dim}, right_dim={right_dim}")
        print(f"  c_linear.weight.shape={c_linear.weight.shape} (esperado: ({c_out_dim}, {left_dim*right_dim}))")

        W_consumer = c_linear.weight.data.reshape(c_out_dim, left_dim, right_dim)
        if side == "left":
            W_consumer_new = torch.einsum("oij,ki->okj", W_consumer, R)
        else:
            W_consumer_new = torch.einsum("oij,kj->oik", W_consumer, R)
        c_linear.weight.data = W_consumer_new.reshape(c_out_dim, left_dim * right_dim).contiguous()

        with torch.no_grad():
            output_after_absorb = model(x_test).clone()
        diff_after_absorb = (output_before_anything - output_after_absorb).abs().max().item()
        print(f"  Apos absorver R no consumidor: diff_max={diff_after_absorb:.6f}")


if __name__ == "__main__":
    graph, trajectories, gene_names = load_dream4(size=10, network=1, root=".")

    train_trajs, val_trajs, test_trajs = split_trajectories_by_fraction(trajectories)
    x_train, x_train_next = trajectories_to_stacked_arrays(train_trajs)
    x_val, x_val_next = trajectories_to_stacked_arrays(val_trajs)
    x_test, _ = trajectories_to_stacked_arrays(test_trajs)

    x_train_t, x_train_next_t = torch.tensor(x_train), torch.tensor(x_train_next)
    x_val_t, x_val_next_t = torch.tensor(x_val), torch.tensor(x_val_next)
    x_test_t = torch.tensor(x_test)

    hierarchy = detect_hierarchy(graph, method="louvain")
    ttn = TTNModel(hierarchy=hierarchy, gene_names=gene_names, bond_dim=3)
    train_ttn(ttn, x_train_t, x_train_next_t, x_val_t, x_val_next_t, epochs=50, seed=0)
    ttn.eval()

    print(f"community_ids = {ttn.community_ids}")
    target = ttn.community_ids[0]
    diagnose(ttn, target, x_test_t)
