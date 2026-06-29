"""
cytos/rigorous_entanglement.py

Entropia de emaranhamento de von Neumann RIGOROSA (nao o proxy
simplificado de cytos.entanglement), via canonicalizacao da TTN
treinada (sweep de decomposicoes QR).

AVISO: a matematica do algoritmo foi validada contra referencia exata
em casos de teste pequenos (numpy puro), mas a INTEGRACAO com a
estrutura real do TTNModel NAO foi testada com torch ao vivo. Rode
test_rigorous_entanglement_validation.py (forca bruta numa rede
pequena) ANTES de confiar em qualquer numero desta implementacao.
"""

from __future__ import annotations

import numpy as np

from cytos.ttn_model import LEAF_DIM


def _shannon_entropy_from_singular_values(sigma):
    sigma_sq = sigma ** 2
    total = sigma_sq.sum()
    if total < 1e-12:
        return 0.0
    p = sigma_sq / total
    p = p[p > 1e-12]
    return float(-np.sum(p * np.log(p)))


def _weight_matrix(model, namespace, node):
    key = model._node_key(namespace, node)
    return model.contractions[key].weight.detach().numpy()


def _full_canon_residual(model, node, namespace, leaf_dim_fn):
    if not isinstance(node, tuple):
        return None, leaf_dim_fn(node)

    left, right = node
    l_res, l_dim = _full_canon_residual(model, left, namespace, leaf_dim_fn)
    r_res, r_dim = _full_canon_residual(model, right, namespace, leaf_dim_fn)

    orig_left_dim = model.bond_dim if isinstance(left, tuple) else leaf_dim_fn(left)
    orig_right_dim = model.bond_dim if isinstance(right, tuple) else leaf_dim_fn(right)

    W = _weight_matrix(model, namespace, node)
    bond_out = W.shape[0]
    W3 = W.reshape(bond_out, orig_left_dim, orig_right_dim)

    if l_res is not None:
        W3 = np.einsum("blr,kl->bkr", W3, l_res)
    if r_res is not None:
        W3 = np.einsum("blr,kr->blk", W3, r_res)

    Wmat = W3.reshape(bond_out, l_dim * r_dim).T
    Q, R = np.linalg.qr(Wmat)
    return R, R.shape[0]


def _sweep_with_target(model, node, target_leaf, namespace, leaf_dim_fn, leaf_residual_fn):
    if node == target_leaf:
        return True, leaf_residual_fn(node)

    if not isinstance(node, tuple):
        return False, (None, leaf_dim_fn(node))

    left, right = node
    left_has, left_payload = _sweep_with_target(model, left, target_leaf, namespace, leaf_dim_fn, leaf_residual_fn)
    right_has, right_payload = _sweep_with_target(model, right, target_leaf, namespace, leaf_dim_fn, leaf_residual_fn)

    orig_left_dim = model.bond_dim if isinstance(left, tuple) else leaf_dim_fn(left)
    orig_right_dim = model.bond_dim if isinstance(right, tuple) else leaf_dim_fn(right)

    W = _weight_matrix(model, namespace, node)
    bond_out = W.shape[0]
    W3 = W.reshape(bond_out, orig_left_dim, orig_right_dim)

    if not left_has and not right_has:
        l_res, l_dim = left_payload
        r_res, r_dim = right_payload
        if l_res is not None:
            W3 = np.einsum("blr,kl->bkr", W3, l_res)
        if r_res is not None:
            W3 = np.einsum("blr,kr->blk", W3, r_res)
        Wmat = W3.reshape(bond_out, l_dim * r_dim).T
        Q, R = np.linalg.qr(Wmat)
        return False, (R, R.shape[0])

    if left_has:
        target_payload, other_payload = left_payload, right_payload
        target_is_left = True
    else:
        target_payload, other_payload = right_payload, left_payload
        target_is_left = False

    other_res, other_dim = other_payload
    if target_is_left and other_res is not None:
        W3 = np.einsum("blr,kr->blk", W3, other_res)
    elif not target_is_left and other_res is not None:
        W3 = np.einsum("blr,kl->bkr", W3, other_res)

    if len(target_payload) == 2:
        t_res, t_dim = target_payload
        if t_res is not None:
            if target_is_left:
                W3 = np.einsum("blr,kl->bkr", W3, t_res)
            else:
                W3 = np.einsum("blr,kr->blk", W3, t_res)
        tensor3 = W3
        if not target_is_left:
            tensor3 = tensor3.transpose(0, 2, 1)
        return True, (t_dim, tensor3.shape[2], tensor3)
    else:
        t_dim, other_dim_inner, tensor3_inner = target_payload
        if target_is_left:
            combined = np.einsum("blr,lto->btor", W3, tensor3_inner)
        else:
            combined = np.einsum("blr,rto->btol", W3, tensor3_inner)
            combined = combined.transpose(0, 1, 3, 2)
        b_shape, t_dim_out, o_inner, o_outer = combined.shape
        tensor3 = combined.reshape(b_shape, t_dim_out, o_inner * o_outer)
        return True, (t_dim_out, tensor3.shape[2], tensor3)


def rigorous_bond_entropy_for_community(model, comm_id):
    residual_comm = {}
    eff_dim_comm = {}
    for cid, tree in model.community_trees.items():
        gene_leaf_dim = lambda _g: LEAF_DIM
        res, dim = _full_canon_residual(model, tree, namespace=f"comm{cid}", leaf_dim_fn=gene_leaf_dim)

        if str(cid) in model.community_finalize:
            finalize_W = model.community_finalize[str(cid)].weight.detach().numpy()
            combined_W = finalize_W
            Q, R = np.linalg.qr(combined_W.T)
            residual_comm[cid] = R
            eff_dim_comm[cid] = R.shape[0]
        else:
            residual_comm[cid] = res
            eff_dim_comm[cid] = dim

    leaf_dim_fn = lambda cid: eff_dim_comm[cid]
    leaf_residual_fn = lambda cid: (residual_comm[cid], eff_dim_comm[cid])

    has_target, payload = _sweep_with_target(
        model, model.root_tree, comm_id, namespace="root",
        leaf_dim_fn=leaf_dim_fn, leaf_residual_fn=leaf_residual_fn,
    )

    if not has_target:
        return float("nan")

    target_dim, other_dim, tensor3 = payload
    bond_out = tensor3.shape[0]
    M = tensor3.transpose(1, 0, 2).reshape(target_dim, bond_out * other_dim)
    sigma = np.linalg.svd(M, compute_uv=False)
    return _shannon_entropy_from_singular_values(sigma)
