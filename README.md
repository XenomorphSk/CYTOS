# CYTOS

**Tree Tensor Networks vs. Graph Neural Networks for Gene Regulatory Network Dynamics**

A pre-registered, falsification-driven comparison between hierarchical Tree Tensor Networks (TTN) and parameter-matched Graph Neural Networks (GNN) on the DREAM4 In Silico Networks benchmark.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

---

## TL;DR

A TTN whose hierarchy is fixed by community structure beats a parameter-matched GNN baseline on:
1. **Parameter efficiency** (MSE per parameter) — 4 to 6× better across all tested configurations.
2. **Long-range correlation capture** — TTN stays at ~0.91–0.95 regardless of network size; the GNN baseline degrades from ~0.5 (10 genes) to ~0.2–0.4 (100 genes).

Confirmed in **10/10 tested configurations** (all 5 networks in the DREAM4 benchmark, sizes 10 and 100 genes). Full methodology, every hypothesis, and every bug found during development are documented in [`docs/pre_registration.md`](docs/pre_registration.md) — including the parts that didn't work on the first try.

## Why this might matter

Gene regulatory networks exhibit modular, hierarchical topology (communities of genes, sparse hubs). Standard GNNs treat this as a flat graph and learn structure implicitly through message passing. Tree Tensor Networks were originally developed in quantum many-body physics to efficiently compress systems whose entanglement entropy scales with the *boundary* of a region rather than its *volume* — a property that maps naturally onto modular biological networks. This project tests whether that structural prior gives a TTN a real, measurable advantage over a GNN with an equivalent parameter budget — not just "TTN is a different parametrization that happens to do okay."

## Methodology (short version)

- **Task:** predict gene expression at t+1 given expression at t.
- **Models:** GCN baseline vs. TTN whose hierarchy is fixed by Louvain community detection on the regulatory graph — decided *before* training, not tuned to the result.
- **Fairness control:** model sizes are matched within 10% of each other's parameter count via joint grid search.
- **Data:** DREAM4 In Silico Networks Challenge (Stolovitzky et al., via Synapse `syn3049712`). Each network provides multiple independent trajectories (replicates); splits are done by whole trajectory, never by timepoint, to eliminate temporal leakage by construction.
- **Statistics:** paired Wilcoxon signed-rank test across 20 random seeds per configuration.
- **Pre-registration:** hypotheses, success criteria, and statistical thresholds were written down *before* running on real data. See [`docs/pre_registration.md`](docs/pre_registration.md) for the full document, including a transparent changelog of every infrastructure bug found and fixed, and an explicit amendment process for design decisions made after discovering the real data didn't match initial assumptions.

## Important caveat

DREAM4, despite being a real community benchmark (not a dataset generated for this project), is itself **simulated** — produced via GeneNetWeaver (parameterized ODEs), not direct experimental measurement. Results here should be read as "TTN beats GNN on the DREAM4 in-silico benchmark," not as a claim about real biological systems in general. See Section 13 of the pre-registration document for the full scope statement.

## Repository structure

```
config/             # experiment configuration (config.yaml)
docs/               # pre-registration document (methodology, results, changelog)
src/
  data/             # DREAM4 loading, trajectory parsing, community detection
  models/           # GNN baseline and TTN implementations
  experiments/       # experiment runner (parallelized across seeds)
tests/              # sanity tests for both models
```

## Reproducing this

1. Get access to the DREAM4 In Silico Networks Challenge data via [Synapse](https://www.synapse.org/#!Synapse:syn3049712) (free registration, accept data use terms). Download:
   - `DREAM4_InSilico_Size10.zip` and `DREAM4_InSilico_Size100.zip` (training data)
   - `DREAM4_InSilicoNetworks_GoldStandard.zip` (gold standard regulatory networks)
2. Extract into the repository root (the paths in `config/config.yaml` expect `DREAM4_InSilico_Size10/`, `DREAM4_InSilico_Size100/`, and `DREAM4_Challenge2_GoldStandards/` at the top level — see `.gitignore`, this data is intentionally not redistributed in this repo due to Synapse's terms of use).
3. Install dependencies:
   ```bash
   pip install torch torch_geometric networkx pandas numpy scipy python-louvain scikit-learn pyyaml --break-system-packages
   ```
4. Run sanity tests:
   ```bash
   pytest tests/test_models.py -v
   ```
5. Run the full experiment:
   ```bash
   python -m src.experiments.runner
   ```

## Status

Pilot-scale result, replicated across all available DREAM4 networks at two sizes (10, 100 genes). Not yet tested: larger networks, real (non-simulated) expression data, or the original DREAM4 Challenge 2 task (topology inference, as opposed to dynamics prediction). See the pre-registration document's limitations section for the full honest accounting.

## Citation

If you use this code or build on this result, please cite/link back to this repository. A preprint write-up is planned — this README will be updated with a citation block once available.

## License

Apache License 2.0 — see [LICENSE](LICENSE).