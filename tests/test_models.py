"""
tests/test_models.py

Testes de sanidade — não validam ciência, só garantem que os modelos rodam,
produzem shapes corretas e que a contagem de parâmetros é o que esperamos.
Rodar com: pytest tests/test_models.py -v
"""

import torch

from src.models.gnn_baseline import GNNBaseline
from src.models.ttn_model import TTNModel


def _toy_hierarchy():
    # 6 genes, 2 comunidades de 3
    return {
        "level_0": {
            "G1": 0, "G2": 0, "G3": 0,
            "G4": 1, "G5": 1, "G6": 1,
        }
    }


def test_gnn_forward_shape():
    num_nodes = 6
    model = GNNBaseline(num_nodes=num_nodes, hidden_dim=8, num_layers=2)
    x = torch.randn(num_nodes, 1)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    out = model(x, edge_index)
    assert out.shape == (num_nodes, 1)


def test_gnn_param_count_positive():
    model = GNNBaseline(num_nodes=6, hidden_dim=8, num_layers=2)
    assert model.count_parameters() > 0


def test_ttn_forward_shape():
    gene_names = ["G1", "G2", "G3", "G4", "G5", "G6"]
    hierarchy = _toy_hierarchy()
    model = TTNModel(hierarchy=hierarchy, gene_names=gene_names, bond_dim=4)
    x = torch.randn(len(gene_names))
    out = model(x)
    assert out.shape == (len(gene_names),)


def test_ttn_param_count_scales_with_bond_dim():
    gene_names = ["G1", "G2", "G3", "G4", "G5", "G6"]
    hierarchy = _toy_hierarchy()
    small = TTNModel(hierarchy=hierarchy, gene_names=gene_names, bond_dim=2)
    large = TTNModel(hierarchy=hierarchy, gene_names=gene_names, bond_dim=8)
    assert large.count_parameters() > small.count_parameters()


def test_ttn_backward_pass_runs():
    gene_names = ["G1", "G2", "G3", "G4", "G5", "G6"]
    hierarchy = _toy_hierarchy()
    model = TTNModel(hierarchy=hierarchy, gene_names=gene_names, bond_dim=4)
    x = torch.randn(len(gene_names))
    target = torch.randn(len(gene_names))
    out = model(x)
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()
    # garante que gradientes foram calculados em pelo menos um parâmetro
    assert any(p.grad is not None for p in model.parameters())

def test_ttn_handles_singleton_community():
    gene_names = ["G1", "G2", "G3"]
    hierarchy = {"level_0": {"G1": 0, "G2": 0, "G3": 1}}
    model = TTNModel(hierarchy=hierarchy, gene_names=gene_names, bond_dim=4)
    x = torch.randn(len(gene_names))
    out = model(x)
    assert out.shape == (len(gene_names),)
    assert "1" in model.community_finalize


def test_ttn_genuine_contraction_uses_outer_product():
    gene_names = ["G1", "G2"]
    hierarchy = {"level_0": {"G1": 0, "G2": 0}}
    model = TTNModel(hierarchy=hierarchy, gene_names=gene_names, bond_dim=4)
    x1 = torch.tensor([0.3, 0.3])
    x2 = torch.tensor([0.3, 0.9])
    out1 = model(x1)
    out2 = model(x2)
    assert not torch.allclose(out1, out2)
