import torch
from cytos.datasets import load_dream4
from cytos.data import split_trajectories_by_fraction, trajectories_to_stacked_arrays
from cytos.ttn_model import TTNModel, train_ttn
from cytos.entanglement import bond_entropy_for_community
from cytos.canonicalization import compute_rigorous_bond_entropy

if __name__ == "__main__":
    graph, trajectories, gene_names = load_dream4(size=10, network=1, root=".")

    train_trajs, val_trajs, test_trajs = split_trajectories_by_fraction(trajectories)
    x_train, x_train_next = trajectories_to_stacked_arrays(train_trajs)
    x_val, x_val_next = trajectories_to_stacked_arrays(val_trajs)
    x_test, _ = trajectories_to_stacked_arrays(test_trajs)

    x_train_t, x_train_next_t = torch.tensor(x_train), torch.tensor(x_train_next)
    x_val_t, x_val_next_t = torch.tensor(x_val), torch.tensor(x_val_next)
    x_test_t = torch.tensor(x_test)

    from cytos.data import detect_hierarchy
    hierarchy = detect_hierarchy(graph, method="louvain")

    ttn = TTNModel(hierarchy=hierarchy, gene_names=gene_names, bond_dim=3)
    train_ttn(ttn, x_train_t, x_train_next_t, x_val_t, x_val_next_t, epochs=200, seed=0)
    ttn.eval()

    print(f"bond_dim={ttn.bond_dim} (precisa ser <= 4 para a versao rigorosa funcionar)")
    print(f"comunidades: {ttn.community_ids}")

    for comm_id in ttn.community_ids:
        proxy = bond_entropy_for_community(ttn, comm_id)
        try:
            rigorous = compute_rigorous_bond_entropy(ttn, comm_id, x_test_t)
            print(f"Comunidade {comm_id}: proxy={proxy:.4f}, rigorosa={rigorous:.4f}")
        except (ValueError, RuntimeError) as e:
            print(f"Comunidade {comm_id}: proxy={proxy:.4f}, rigorosa FALHOU: {e}")
