"""
cytos/canonicalization.py

Canonicalizacao de tensor network (sweep de decomposicoes QR), gerando
entropia de emaranhamento de von Neumann RIGOROSA para uma comunidade
especifica - em vez do proxy simplificado usado em cytos.entanglement.

ESCOPO E LIMITACOES:

1. Apenas o lado da COMUNIDADE e canonicalizado (toda a subarvore interna
   da comunidade e transformada em isometrias puras, com o conteudo
   nao-isometrico absorvido no no do root_plan onde a comunidade se
   conecta ao resto). O lado "resto da rede" NAO e canonicalizado.

2. Restricao de seguranca: funciona apenas quando bond_dim <= LEAF_DIM**2
   (= 4). Fora disso, levanta erro explicito em vez de prosseguir.

3. VERIFICACAO OBRIGATORIA: apos canonicalizar, a saida do modelo nos
   dados de teste e comparada a saida ANTES da canonicalizacao. Se
   diferentes, levanta erro - canonicalizacao correta nunca muda a
   funcao computada (transformacao de gauge exata).
"""

from __future__ import annotations

import copy

import numpy as np
import torch

from cytos.ttn_model import LEAF_DIM, TTNModel


def _ref_dim(ref, leaf_dim_fn, bond_dim):
    return bond_dim if ref[0] == "node" else leaf_dim_fn(ref[1])


def _build_consumer_map(plan, comm_id, root_plan):
    consumer_of = {}
    for i in range(len(plan)):
        for j, (left_ref, right_ref, _) in enumerate(plan):
            if left_ref == ("node", i):
                consumer_of[i] = ("community", j, "left")
            if right_ref == ("node", i):
                consumer_of[i] = ("community", j, "right")

    last_idx = len(plan) - 1
    target_leaf = ("leaf", comm_id)
    for j, (left_ref, right_ref, _) in enumerate(root_plan):
        if left_ref == target_leaf:
            consumer_of[last_idx] = ("root", j, "left")
        if right_ref == target_leaf:
            consumer_of[last_idx] = ("root", j, "right")

    return consumer_of


def canonicalize_community_subtree(model, comm_id):
    if model.bond_dim > LEAF_DIM ** 2:
        raise ValueError(
            f"bond_dim={model.bond_dim} > LEAF_DIM**2={LEAF_DIM**2}. "
            "Canonicalizacao rigorosa nao implementada com seguranca "
            "para esse caso. Use o proxy simplificado (cytos.entanglement)."
        )

    plan, root_ref = model.community_plans[comm_id]
    if not plan:
        raise ValueError(f"Comunidade {comm_id} e singleton - nada a canonicalizar.")

    consumer_of = _build_consumer_map(plan, comm_id, model.root_plan)

    gene_leaf_dim = lambda _g: LEAF_DIM
    root_leaf_dim = lambda _c: model.bond_dim

    for i, (left_ref, right_ref, module_idx) in enumerate(plan):
        linear = model.contractions[module_idx]
        W = linear.weight.data.clone()
        Wt = W.t()
        Q, R = torch.linalg.qr(Wt, mode="reduced")

        if Q.shape[1] != W.shape[0]:
            raise ValueError(
                f"Dimensao efetiva encolheu no no {i} da comunidade {comm_id}. "
                "Abortando em vez de prosseguir com resultado potencialmente errado."
            )

        linear.weight.data = Q.t().contiguous()

        if i not in consumer_of:
            continue
        location, j, side = consumer_of[i]
        if location == "community":
            c_left_ref, c_right_ref, c_module_idx = plan[j]
        else:
            c_left_ref, c_right_ref, c_module_idx = model.root_plan[j]

        c_linear = model.contractions[c_module_idx]
        c_out_dim = c_linear.weight.shape[0]
        leaf_dim_fn = gene_leaf_dim if location == "community" else root_leaf_dim
        left_dim = _ref_dim(c_left_ref, leaf_dim_fn, model.bond_dim)
        right_dim = _ref_dim(c_right_ref, leaf_dim_fn, model.bond_dim)

        W_consumer = c_linear.weight.data.reshape(c_out_dim, left_dim, right_dim)
        if side == "left":
            W_consumer_new = torch.einsum("oij,ki->okj", W_consumer, R)
        else:
            W_consumer_new = torch.einsum("oij,kj->oik", W_consumer, R)
        c_linear.weight.data = W_consumer_new.reshape(c_out_dim, left_dim * right_dim).contiguous()

    target_leaf = ("leaf", comm_id)
    for j, (left_ref, right_ref, module_idx) in enumerate(model.root_plan):
        if left_ref == target_leaf or right_ref == target_leaf:
            comm_is_left = left_ref == target_leaf
            return {"module_idx": module_idx, "comm_is_left": comm_is_left}

    raise ValueError(f"Comunidade {comm_id} nao encontrada como folha do root_plan.")


def compute_rigorous_bond_entropy(model, comm_id, x_test):
    """
    LIMITACAO ESTRUTURAL DESCOBERTA (2026-06-24): comm_vec e usado em
    DOIS lugares: (1) entrada pro root_plan (corrigido pela absorcao de
    R), e (2) DIRETAMENTE na predicao de cada gene da propria comunidade
    (nao so via global_repr). A transformacao de gauge so corrige (1).
    A verificacao de invariancia e por isso restrita aos genes DE FORA
    da comunidade-alvo - para esses, a saida deve ser EXATAMENTE
    preservada (e e exatamente essa relacao - "esta comunidade" vs "o
    resto" - que a entropia de bond mede).
    """
    model_copy = copy.deepcopy(model)
    model_copy.eval()

    comm_slot = model.community_ids.index(comm_id)
    outside_idx = [i for i, g in enumerate(model.gene_names) if model.gene_to_community_slot[g] != comm_slot]

    with torch.no_grad():
        output_before = model_copy(x_test).clone()

    info = canonicalize_community_subtree(model_copy, comm_id)

    with torch.no_grad():
        output_after = model_copy(x_test)

    if outside_idx:
        out_before_ext = output_before[:, outside_idx]
        out_after_ext = output_after[:, outside_idx]
        if not torch.allclose(out_before_ext, out_after_ext, atol=1e-4, rtol=1e-3):
            max_diff = (out_before_ext - out_after_ext).abs().max().item()
            raise RuntimeError(
                f"VERIFICACAO FALHOU (genes de fora da comunidade): a saida do "
                f"modelo mudou apos canonicalizacao (diferenca maxima: {max_diff:.6f}). "
                f"Bug real - resultado de entropia NAO confiavel."
            )

    module_idx = info["module_idx"]
    comm_is_left = info["comm_is_left"]
    linear = model_copy.contractions[module_idx]
    W = linear.weight.detach().numpy()
    bond_dim = model_copy.bond_dim
    W3 = W.reshape(bond_dim, bond_dim, bond_dim)

    if comm_is_left:
        M = W3.transpose(1, 0, 2).reshape(bond_dim, bond_dim * bond_dim)
    else:
        M = W3.transpose(2, 0, 1).reshape(bond_dim, bond_dim * bond_dim)

    sigma = np.linalg.svd(M, compute_uv=False)
    sigma_sq = sigma ** 2
    total = sigma_sq.sum()
    if total < 1e-12:
        return 0.0
    p = sigma_sq / total
    p = p[p > 1e-12]
    return float(-np.sum(p * np.log(p)))
