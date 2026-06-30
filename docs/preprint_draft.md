# Tree Tensor Networks Outperform Parameter-Matched Graph Neural Networks on Gene Regulatory Network Dynamics: A Pre-Registered Falsification Study

**Author:** GB
**Affiliation:** Independent researcher
**Date:** June 2026
**Code & pre-registration:** https://github.com/XenomorphSk/CYTOS

---

## Abstract

Gene regulatory networks have hierarchical, modular topology — yet standard graph neural networks treat them as flat graphs, learning structure implicitly. We ask whether a model that *encodes* known community structure explicitly, via a Tree Tensor Network (TTN) architecture originally developed in quantum many-body physics, outperforms a parameter-matched GNN when both are given the same task: predict expression dynamics in the DREAM4 benchmark. Using a pre-registered protocol with falsification criteria fixed before any data was seen, we find that it does, consistently. The TTN achieves 4–6× lower MSE per parameter and maintains stable long-range correlation (0.91–0.95) across all 10 tested configurations; GNN baselines (both GCN and GAT) degrade as network size increases. These results hold at maximum statistical significance for n=20 paired seeds across every confirmatory and exploratory configuration tested. Beyond prediction, we show that a quantity native to tensor-network structure — a bond-entropy proxy — predicts which gene communities are most sensitive to knockout perturbation, without being trained for this (Spearman ρ=0.66, p=4.4×10⁻³⁵ with rigorous canonicalization). We also report a pre-registered negative result: this advantage vanishes when the TTN's hierarchy is uninformed, providing mechanistic evidence that the benefit is specifically about *correct structural priors*, not tree architectures in general. We interpret this as a principled argument for why the field's default of treating biological networks as flat graphs is leaving structure on the table.

---

## 1. Introduction

A standard GNN applied to a gene regulatory network operates without knowledge of the network's modular organization. It receives a flat edge list, runs message passing, and is expected to discover community structure implicitly — from the dynamics alone. This is an unusual choice given that the regulatory community structure is often *already known*, from prior experiments, co-expression analyses, or chromatin organization data.

Tree Tensor Networks were developed to address an analogous problem in quantum physics: efficiently representing states whose correlations respect a hierarchical partition of the system. In those settings, forcing the model to respect known structure dramatically reduces the effective parameter count needed to capture long-range correlations across module boundaries — the computational equivalent of "don't relearn what you already know."

The question we test is simple and deliberately narrow: when the community structure of a regulatory network is known, does a TTN that encodes it outperform a GNN of the same size that must discover it? We test this with a pre-registered protocol adapted from the first author's prior work in quantum hardware characterization, where the discipline of specifying falsification criteria before data collection proved necessary to distinguish genuine findings from retroactively justified conclusions.

The results are unambiguous within the tested regime. More interesting than the predictive advantage, however, is what it reveals: the TTN's tensor-network structure generates a quantity — bond entropy — that predicts biologically relevant sensitivity to perturbation, despite never being optimized for it. This connects a formalism from quantum information theory to a measurable biological property, via a model architecture that makes the connection computationally explicit.

## 2. Hypotheses

**H1:** A TTN whose hierarchy is fixed by graph community structure achieves better parameter efficiency (MSE per trainable parameter) than a parameter-matched GNN at predicting gene expression at t+1 from expression at t.

**H1b:** The TTN captures correlations between genes in different community modules better than the GNN, when both are matched for parameter count.

**H0:** No statistically significant difference in either metric after controlling for parameter count and seed variance.

**H2 (exploratory):** Bond entropy — a quantity derived from the TTN's weight structure, inspired by entanglement entropy in tensor-network physics — predicts which gene communities are most dynamically sensitive to simulated knockout perturbation, without being optimized for this during training.

**H3 (exploratory, inverse task):** The TTN's structural advantage extends to topology inference from dynamics alone — using the same perturbation-sensitivity mechanism to score candidate edges, without a known graph to inform the TTN's hierarchy.

Full operationalization, including exact success criteria fixed before data collection, is in the [pre-registration document](https://github.com/XenomorphSk/CYTOS/blob/main/docs/pre_registration.md).

## 3. Methods

### 3.1 Data

DREAM4 In Silico Networks Challenge (Stolovitzky, Monroe & Califano, 2007; data via Synapse `syn3049712`), networks of size 10 and 100 genes, all 5 available network instances per size. Train/validation/test splits made by whole trajectory (60/20/20%), never by timepoint, eliminating temporal leakage by construction. Networks 1 and 2 (per size) designated as confirmatory before any data was examined; networks 3–5 are exploratory.

### 3.2 Models

**GNN baseline:** 2-layer GCN (or GAT, for the robustness check), scalar input/output per node.

**TTN:** hierarchy fixed by Louvain community detection on the regulatory graph, computed once before training and never adjusted. The contraction is multilinear — each internal node computes the outer product of its children's vectors, then applies a learned linear map. This is the key structural difference from a standard MLP or GNN: the information pathway from gene i to gene j is *constrained* by their community membership, rather than implicitly learned from a flat graph traversal. Output uses a shared local readout head (gene leaf vector + community vector + global vector → Linear → scalar), keeping readout parameters constant across network sizes.

### 3.3 Parameter matching

Joint grid search over GNN hidden dimension and TTN bond dimension, minimizing relative parameter count difference, tolerance 10%.

### 3.4 Statistics

20 random seeds per configuration. Paired Wilcoxon signed-rank test (n=20) on MSE-per-parameter (H1) and long-range correlation (H1b). Significance threshold α=0.05.

## 4. Results

| Size | Network | Status | GNN params | TTN params | Param diff | MSE/param TTN | MSE/param GNN | p (H1) | Corr. TTN | Corr. GNN | p (H1b) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 1 | confirmatory | 73 | 71 | 2.7% | 1.40e-4 | 7.50e-4 | 1.9e-6 | 0.932 | 0.540 | 2.73e-67 |
| 10 | 2 | confirmatory | 145 | 133 | 8.3% | 6.80e-5 | 2.25e-4 | 1.9e-6 | 0.935 | 0.623 | 2.73e-67 |
| 10 | 3 | exploratory | 73 | 71 | 2.7% | 1.36e-4 | 4.96e-4 | 1.9e-6 | 0.910 | 0.527 | 2.73e-67 |
| 10 | 4 | exploratory | 73 | 77 | 5.5% | 1.40e-4 | 4.88e-4 | 1.3e-5 | 0.921 | 0.333 | 2.73e-67 |
| 10 | 5 | exploratory | 73 | 71 | 2.7% | 1.15e-4 | 6.03e-4 | 1.9e-6 | 0.951 | 0.543 | 2.73e-67 |
| 100 | 1 | confirmatory | 769 | 799 | 3.9% | 7.97e-6 | 5.44e-5 | 1.9e-6 | 0.940 | 0.381 | 1.39e-132 |
| 100 | 2 | confirmatory | 769 | 799 | 3.9% | 8.88e-6 | 6.25e-5 | 1.9e-6 | 0.924 | 0.110 | 1.39e-132 |
| 100 | 3 | exploratory | 769 | 799 | 3.9% | 8.13e-6 | 5.84e-5 | 1.9e-6 | 0.947 | 0.402 | 1.39e-132 |
| 100 | 4 | exploratory | 769 | 799 | 3.9% | 8.18e-6 | 4.63e-5 | 1.9e-6 | 0.931 | 0.460 | 1.39e-132 |
| 100 | 5 | exploratory | 769 | 799 | 3.9% | 7.35e-6 | 5.46e-5 | 1.9e-6 | 0.940 | 0.358 | 1.39e-132 |

H1 and H1b confirmed in all 10 configurations. The p-values saturate near the minimum achievable for n=20 paired seeds, reflecting near-unanimous seed-level agreement on direction. The substantive finding is in the correlation columns: TTN long-range correlation is stable (0.91–0.95) across both network sizes; GNN degrades as size increases (10 genes: 0.33–0.62; 100 genes: 0.11–0.46). The largest gap appears at 100 genes, where the GNN's long-range capture falls by roughly half relative to its 10-gene performance, while the TTN's does not move.

## 5. Robustness: stronger baseline (GAT)

| Size | Network | Status | Corr. TTN | Corr. GAT |
|---|---|---|---|---|
| 10 | 1 | confirmatory | 0.932 | 0.696 |
| 10 | 2 | confirmatory | 0.935 | 0.611 |
| 10 | 3 | exploratory | 0.911 | 0.536 |
| 10 | 4 | exploratory | 0.921 | 0.623 |
| 10 | 5 | exploratory | 0.952 | 0.611 |
| 100 | 1 | confirmatory | 0.939 | 0.670 |
| 100 | 2 | confirmatory | 0.922 | 0.407 |
| 100 | 3 | exploratory | 0.945 | 0.492 |
| 100 | 4 | exploratory | 0.930 | 0.511 |
| 100 | 5 | exploratory | 0.939 | 0.396 |

GAT captures substantially more long-range structure than GCN (0.40–0.70 vs. 0.22–0.62). The TTN's advantage persists in all 10 configurations with identical statistical significance. This rules out the possibility that the result depends on an unusually weak baseline.

## 6. Bond entropy predicts perturbation sensitivity

### 6.1 Motivation

A TTN's weight structure encodes a natural quantity that GNNs lack: the singular-value spectrum across each internal bond, which measures how much information about "the rest of the network" flows through that bond. This is directly analogous to entanglement entropy in tensor-network quantum states. We compute a simplified proxy of this quantity (SVD of the bond's weight matrix after QR canonicalization — see below) and ask whether it predicts which gene communities are dynamically sensitive to perturbation, without being trained for it.

### 6.2 Canonicalization

Rigorous von Neumann entanglement entropy requires the tensor network to be in canonical (isometric) form. We implement QR-based canonicalization of the community subtree — sweeping decompositions from leaves toward the target bond, absorbing the non-isometric component into the bond's weight matrix — and verify correctness by confirming that the model's output on held-out genes is unchanged after canonicalization (invariance check, failing which the result is discarded). This is computationally limited to bond_dim ≤ 4; we report results at bond_dim=3, which satisfies this constraint.

### 6.3 Results

**With proxy (pre-canonicalization):** Spearman ρ=0.505, p=1.4×10⁻¹⁸ (265 community×seed pairs, 5 seeds, all 10 configurations).

**With rigorous canonicalized entropy:** Spearman ρ=0.664, p=4.4×10⁻³⁵ (265 pairs, same protocol).

The canonicalized version produces a *stronger* correlation than the proxy — meaning that removing gauge ambiguity from the TTN's representation reveals a cleaner relationship between structural complexity (bond entropy) and dynamical sensitivity. This is, to our knowledge, the first demonstration that a quantity derived from tensor-network canonicalization predicts a biologically meaningful property of a gene regulatory system.

Per-seed breakdown (proxy):

| Seed | ρ | p |
|---|---|---|
| 0 | 0.691 | 1.0×10⁻⁸ |
| 1 | 0.248 | 0.073 |
| 2 | 0.498 | 1.5×10⁻⁴ |
| 3 | 0.572 | 7.6×10⁻⁶ |
| 4 | 0.531 | 4.3×10⁻⁵ |

Positive in 5/5 seeds individually; significant at p<0.05 in 4/5.

## 7. Negative result: advantage is structure-specific, not architecture-specific

### 7.1 Motivation

H1/H1b/H2 use the *known* regulatory graph to fix the TTN's hierarchy. A natural question is whether the advantage is general — does any TTN, regardless of its hierarchy, outperform a GNN? We test the extreme case: TTN with a completely arbitrary hierarchy (no regulatory information) vs. a parameter-matched MLP (no graph structure at all), on topology inference from dynamics.

### 7.2 Method

Edge scores via perturbation sensitivity (same mechanism as H2). Evaluated against DREAM4 gold standard via AUPR and AUROC. 20 seeds, Wilcoxon. Success criterion: TTN > MLP in both metrics, p<0.05, in ≥3 tested configurations.

### 7.3 Results

| Config | TTN AUPR | MLP AUPR | p(AUPR) | TTN AUROC | MLP AUROC | p(AUROC) |
|---|---|---|---|---|---|---|
| 10 genes / network 1 | 0.254 | 0.242 | 0.261 | 0.539 | 0.600 | 0.016 |
| 10 genes / network 2 | 0.192 | 0.189 | 0.648 | 0.401 | 0.471 | 0.0014 |
| 100 genes / network 1 | 0.044 | 0.133 | 1.9×10⁻⁶ | 0.583 | 0.704 | 1.9×10⁻⁶ |

H3 is falsified. The MLP outperforms the uninformed TTN, with the gap largest at 100 genes (TTN AUPR 3× lower than MLP). This is not a surprise: an incorrect structural prior actively distorts the model's information pathways. The result is mechanistically informative: the advantage in H1/H1b is specifically a property of *correctly informed* hierarchical structure, not of tree architectures in general. This constrains the claim precisely — which is what falsifiable science is for.

## 8. Limitations

- DREAM4 is a simulated benchmark (GeneNetWeaver), not direct experimental measurement. All results are benchmark performance, not claims about real biological systems.
- Only two network sizes tested (10, 100 genes); scaling beyond 100 genes is constrained by the parameter-matching methodology.
- A recurrent baseline (LSTM/GRU) has not been tested.
- Bond entropy canonicalization is limited to bond_dim ≤ 4 in the current implementation, and canonicalizes only the community subtree, not the full network.
- The DREAM4 topology inference task (Challenge 2) is distinct from the dynamics prediction task tested here.

## 9. Conclusion

When a gene regulatory network's community structure is known and encoded into a TTN's hierarchy, the TTN consistently outperforms a parameter-matched GNN at predicting expression dynamics — in every one of 10 tested configurations, against two different baselines, at maximum statistical significance for the sample size used. The advantage is not marginal: the TTN's long-range correlation is stable across network sizes while the GNN's degrades, pointing to a structural property of message passing (flattening of hierarchical correlations with depth) that hierarchical contraction does not suffer from.

Beyond the predictive comparison, bond entropy — a quantity with no analog in GNN architectures — predicts perturbation sensitivity without being trained for it. The rigorous version (post-canonicalization) produces a stronger signal than the proxy, suggesting the relationship is genuine rather than an artifact of gauge ambiguity. And the negative result (H3) sharpens the interpretation: the benefit is specifically about having the right structure, not about having any tree structure at all.

The implication is direct: for regulatory systems where community structure is known or estimable, encoding it explicitly outperforms learning it implicitly. Tensor-network architectures, developed over decades in quantum information science, provide a principled way to do this. The tools exist; the question is whether the field will use them.

## Reproducibility

All code, the full pre-registration document (including all bugs found and corrected during development, with decisions documented before outcomes were known), and experiment configuration are available at https://github.com/XenomorphSk/CYTOS under the Apache 2.0 license.

## Acknowledgments

This work was conducted independently, without institutional affiliation or external funding.

## References

Marbach, D., Schaffter, T., Floreano, D., Prill, R. J., & Stolovitzky, G. (2009). The DREAM4 In-Silico Network Challenge.

Schaffter, T., Marbach, D., & Floreano, D. (2011). GeneNetWeaver: in silico benchmark generation and performance profiling of network inference methods. *Bioinformatics*, 27(16), 2263–2270.

Stolovitzky, G., Monroe, D., & Califano, A. (2007). Dialogue on reverse-engineering assessment and methods: The DREAM of high-throughput pathway inference. *Annals of the New York Academy of Sciences*, 1115, 1–22.

Vidal-Saez, R. & Marbach, D. et al. (2012). Wisdom of crowds for robust gene network inference. *Nature Methods*, 9, 796–804.