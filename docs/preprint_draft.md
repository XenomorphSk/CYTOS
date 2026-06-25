# Tree Tensor Networks Outperform Parameter-Matched Graph Neural Networks on Gene Regulatory Network Dynamics: A Pre-Registered Comparison on the DREAM4 Benchmark

**Author:** GB
**Affiliation:** Independent researcher
**Date:** June 2026
**Code & full pre-registration document:** https://github.com/XenomorphSk/CYTOS

---

## Abstract

Gene regulatory networks (GRNs) exhibit modular, hierarchical topology, yet standard graph neural networks (GNNs) treat them as flat graphs and must learn this structure implicitly through message passing. We test whether a Tree Tensor Network (TTN) - a model class originally developed in quantum many-body physics to efficiently represent systems with hierarchical entanglement structure - outperforms a parameter-matched GNN baseline at predicting GRN expression dynamics, when the TTN's hierarchy is fixed by community detection on the regulatory graph rather than learned. Using a pre-registered protocol (hypotheses, success criteria, and statistical thresholds fixed before running on real data), we test both models on the DREAM4 In Silico Networks benchmark across all 5 available networks at two sizes (10 and 100 genes), with model parameter counts matched within ~10%. The TTN outperforms the GNN baseline on parameter efficiency (4-6x lower MSE per parameter) and on capturing long-range correlations between genes in different community modules (TTN: 0.91-0.95 across all configurations; GCN baseline: 0.22-0.62, degrading further as network size increases). The result replicates in 10/10 tested configurations and is robust to baseline strength: repeating the full experiment with a Graph Attention Network (GAT) baseline - which captures substantially more long-range structure than GCN (0.40-0.70) - the TTN still wins in 10/10 configurations. We report this as a pilot-scale, falsification-driven result on a simulated benchmark, not a claim about real biological systems, and document every methodological correction made during development.

---

## 1. Introduction

Target-based and graph-based machine learning approaches to gene regulatory network modeling typically encode the regulatory graph directly into a GNN, relying on the model's message-passing mechanism to discover whatever hierarchical or modular structure is present in the data. This is a reasonable default, but it does not exploit a structural prior that is already known about most biological regulatory networks: they are organized into communities of co-regulated genes, often with sparse hub connectivity between modules (scale-free-like topology).

Tree Tensor Networks (TTNs) were developed in condensed matter physics to compress quantum many-body states whose entanglement entropy scales with the boundary of a subregion rather than its volume - a property closely associated with hierarchical, modular structure. This motivates a direct question: if a TTN's hierarchy is fixed to match the *known* community structure of a regulatory graph, does it outperform a same-size GNN that must discover structure on its own?

This is the question this work tests, with a pre-registration protocol adapted from the author's prior work in experimental quantum information research, where pre-specifying falsification criteria proved necessary to avoid retroactively reinterpreting ambiguous results as confirmatory.

## 2. Hypotheses

**H1:** A TTN whose hierarchy is fixed by graph community structure achieves equal or better parameter efficiency (MSE per trainable parameter) than a parameter-matched GNN, when predicting gene expression at t+1 from expression at t.

**H1b:** The TTN captures correlations between genes in different (hierarchically distant) community modules better than the GNN, when both are matched for parameter count.

**H0 (null):** No statistically significant difference in either metric after controlling for parameter count and seed variance.

Full operationalization, including exact success criteria fixed before data collection, is available in the project's [pre-registration document](https://github.com/XenomorphSk/CYTOS/blob/main/docs/pre_registration.md).

## 3. Methods

### 3.1 Data

DREAM4 In Silico Networks Challenge (Marbach et al., 2009; data via Synapse `syn3049712`), networks of size 10 and 100 genes, all 5 available network instances per size. Each network provides multiple independent trajectories (5 for size 10, 10 for size 100); train/validation/test splits are made by whole trajectory (60/20/20%), never by timepoint, eliminating temporal leakage by construction.

### 3.2 Models

**GNN baseline:** 2-layer GCN, scalar input/output per node (current expression -> next-timestep expression).

**TTN:** hierarchy fixed by Louvain community detection on the (undirected projection of the) regulatory graph, computed once before training and never adjusted post-hoc. Each gene is a leaf tensor; each community has a tensor contracting its members; a root tensor contracts community representations into a single global vector. Output is computed via a *shared* local readout head: each gene's prediction comes from concatenating its own leaf vector, its community's vector, and the global vector, passed through one small `Linear` layer shared across all genes - keeping the readout parameter count constant rather than scaling with the number of genes (an earlier design with a per-gene-scaling output layer made parameter matching infeasible at 100 genes; the architecture was revised before any run on real data, and this revision is documented in the pre-registration changelog).

### 3.3 Parameter matching

Model capacities are matched via joint grid search over GNN hidden dimension and TTN bond dimension, minimizing relative parameter count difference, with a target tolerance of 10%.

### 3.4 Statistics

20 random seeds per configuration. Primary metric (H1): paired Wilcoxon signed-rank test on per-seed MSE-per-parameter (TTN vs. GNN). Secondary metric (H1b): paired Wilcoxon test on per-sample long-range correlation (Pearson correlation between predicted and true pairwise differences for genes in different communities). Significance threshold alpha=0.05, pre-registered success criterion of >=3/4 confirmatory configurations passing.

### 3.5 Confirmatory vs. exploratory testing

Networks 1 and 2 (of 5 available per size) were designated as the confirmatory test before any data was examined. Networks 3-5 were run afterward as an explicitly labeled exploratory replication - not used to rescue H1/H1b had the confirmatory test failed, but to assess generalization across the full available benchmark.

## 4. Results

| Size | Network | Status | GNN params | TTN params | Param diff | MSE/param TTN | MSE/param GNN | p (H1) | Corr. TTN | Corr. GNN | p (H1b) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 1 | confirmatory | 73 | 71 | 2.7% | 1.40e-4 | 7.50e-4 | 1.9e-6 | 0.932 | 0.540 | 2.73e-67 |
| 10 | 2 | confirmatory | 145 | 133 | 8.3% | 6.80e-5 | 2.25e-4 | 1.9e-6 | 0.935 | 0.623 | 2.73e-67 |
| 10 | 3 | exploratory | 73 | 71 | 2.7% | 1.36e-4 | 4.96e-4 | 1.9e-6 | 0.910 | 0.527 | 2.73e-67 |
| 10 | 4 | exploratory | 73 | 77 | 5.5% | 1.40e-4 | 4.88e-4 | 1.9e-6 | 0.921 | 0.333 | 2.73e-67 |
| 10 | 5 | exploratory | 73 | 71 | 2.7% | 1.15e-4 | 6.03e-4 | 1.9e-6 | 0.951 | 0.543 | 2.73e-67 |
| 100 | 1 | confirmatory | 433 | 461 | 6.5% | 1.28e-5 | 9.55e-5 | 1.9e-6 | 0.939 | 0.389 | 1.39e-132 |
| 100 | 2 | confirmatory | 433 | 443 | 2.3% | 1.38e-5 | 1.04e-4 | 1.9e-6 | 0.922 | 0.247 | 1.39e-132 |
| 100 | 3 | exploratory | 433 | 467 | 7.9% | 1.29e-5 | 1.03e-4 | 1.9e-6 | 0.945 | 0.409 | 1.39e-132 |
| 100 | 4 | exploratory | 433 | 461 | 6.5% | 1.24e-5 | 8.14e-5 | 1.9e-6 | 0.930 | 0.458 | 1.39e-132 |
| 100 | 5 | exploratory | 433 | 461 | 6.5% | 1.14e-5 | 9.65e-5 | 1.9e-6 | 0.939 | 0.365 | 1.39e-132 |

H1 and H1b are each confirmed in 10/10 configurations. The p-values for both tests saturate near the minimum achievable value given n=20 paired seeds (Wilcoxon) - this reflects near-unanimous seed-level agreement on direction, not a fine-grained measure of effect size. Effect size should be read from the MSE-per-parameter and correlation columns directly: the TTN's long-range correlation is stable (0.91-0.95) across all networks and both sizes, while the GNN's degrades as network size grows (10 genes: 0.33-0.62; 100 genes: 0.25-0.46).

## 5. Robustness check: stronger baseline (GAT)

To rule out the possibility that the result depends on comparing against a weak baseline, the full experiment (all 5 networks, both sizes, 20 seeds) was repeated with a Graph Attention Network (GAT) in place of the GCN, with parameter matching redone from scratch for the new architecture (GAT carries more parameters per layer due to attention weights).

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

GAT is a substantially stronger baseline than GCN - its attention mechanism captures meaningfully more long-range structure (0.40-0.70, vs. 0.22-0.62 for GCN). However, H1 and H1b are confirmed in 10/10 configurations again: the TTN remains stable at 0.91-0.95 and outperforms GAT in every tested configuration, with the same maximal statistical significance (p=1.9e-6 for H1) observed with the GCN baseline. This indicates the result is not an artifact of comparing against a particularly weak baseline architecture.

## 6. Limitations

- **DREAM4 is itself a simulated benchmark** (generated via GeneNetWeaver, parameterized ODEs), not direct experimental measurement. Results should be read as benchmark performance, not a claim about real biological systems.
- **Small test sets**: one held-out trajectory per network/size, per the pre-registered split.
- **Only two network sizes tested** (10, 100 genes); scaling behavior beyond this range is unknown.
- **The original DREAM4 Challenge 2 task (network topology inference) was not addressed** - this work tests dynamics prediction given a known topology, which is a different (and easier) problem.
- p-values for the secondary metric (H1b) are identical across nearly all configurations, reflecting a ceiling effect of the statistical test given sample size, not a precise measure of effect magnitude.
- A recurrent baseline (e.g., per-gene LSTM/GRU) has not yet been tested; GCN and GAT were the two baselines evaluated.

## 7. Conclusion

Under the criteria fixed before observing the data, both H1 and H1b are supported by the DREAM4 benchmark across all 5 available networks and both tested sizes, against two different GNN baselines (GCN and GAT). The most substantive finding - that GNN baselines' ability to capture long-range, cross-community correlation degrades as network size grows while the TTN's does not, and that this gap persists even against a stronger attention-based baseline - is consistent with the theoretical motivation (modular structure becomes more pronounced, not less, as networks grow, and the TTN's fixed hierarchy exploits this directly while flat or attention-based message-passing does not). This is a positive pilot-scale result, not a final claim; the limitations above outline the next steps required before treating it as established.

## Reproducibility

All code, the full pre-registration document (including every infrastructure bug found and corrected during development), and exact experiment configuration are available at https://github.com/XenomorphSk/CYTOS under the Apache 2.0 license.

## Acknowledgments

This work was conducted independently, without institutional affiliation or external funding.

## References

Marbach, D., Schaffter, T., Floreano, D., Prill, R. J., & Stolovitzky, G. (2009). The DREAM4 In-Silico Network Challenge.

Schaffter, T., Marbach, D., & Floreano, D. (2011). GeneNetWeaver: in silico benchmark generation and performance profiling of network inference methods. *Bioinformatics*, 27(16), 2263-2270.

Stolovitzky, G., Monroe, D., & Califano, A. (2007). Dialogue on reverse-engineering assessment and methods: The DREAM of high-throughput pathway inference. *Annals of the New York Academy of Sciences*, 1115, 1-22.
