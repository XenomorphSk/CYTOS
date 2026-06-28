# cytos

**Tree Tensor Networks vs. Graph Neural Networks for graph-structured temporal dynamics - bring your own graph and data.**

A reusable, installable Python library extracted from the CYTOS research project (see `../docs/pre_registration.md` for the full pre-registered methodology and original DREAM4 results). Test whether a Tree Tensor Network whose hierarchy is fixed by community detection outperforms a parameter-matched Graph Neural Network on *your own* graph and time-series data - not just the DREAM4 benchmark this method was originally validated on.

## Install

```bash
pip install -e .
```

## Quick start

```python
import networkx as nx
from cytos import TTNvsGNN

graph = nx.DiGraph()
graph.add_edges_from([("A", "B"), ("B", "C"), ("A", "C")])

# trajectories: list of np.ndarray, each shape (n_timepoints, n_nodes)
# (at least 3 trajectories needed; if you only have one long series,
# split it into sub-trajectories yourself first)
trajectories = [...]

result = TTNvsGNN(graph=graph, trajectories=trajectories).run(seeds=20)
print(result.summary())
```

## Entanglement-inspired interpretability pilot

```python
ent = result.entanglement_pilot(seeds=[0, 1, 2, 3, 4])
print(ent.summary())
```

Tests whether a "local bond entropy" (simplified proxy, NOT rigorous von Neumann entanglement entropy) predicts which graph communities are most sensitive to simulated node knockout.

## Convenience loader for DREAM4

```python
from cytos.datasets import load_dream4

graph, trajectories, gene_names = load_dream4(size=100, network=1, root=".")
result = TTNvsGNN(graph=graph, trajectories=trajectories, gene_names=gene_names).run(seeds=20)
```

Requires the DREAM4 data obtained separately via [Synapse](https://www.synapse.org/#!Synapse:syn3049712) (not redistributed with this package).

## API

- `TTNvsGNN(graph, trajectories, gene_names=None, **kwargs)` - key kwargs: `train_frac`/`val_frac`/`test_frac` (0.6/0.2/0.2), `clustering_method` ("louvain"/"spectral"), `gnn_architecture` ("GCN"/"GAT"), `param_match_tolerance` (0.10), `significance_alpha` (0.05).
- `.run(seeds=20, **train_kwargs)` -> `TTNvsGNNResult` with `.summary()`, `.to_dict()`, `.to_dataframe()`, `.entanglement_pilot(seeds=...)`.
- `cytos.datasets.load_dream4(size, network, root=".")`.

## Important notes

- **Multiprocessing requirement:** `.run()` uses `ProcessPoolExecutor`. Guard your entry point with `if __name__ == "__main__":`. Does not work from interactive heredoc/stdin execution - save to a `.py` file.
- **Minimum 3 trajectories required** (split by whole trajectory, never by timepoint).
- Pilot-scale research tool (see `../docs/pre_registration.md`); results on DREAM4 should not be assumed to transfer to arbitrary graphs without independent validation.

## License

Apache License 2.0 - see [LICENSE](../LICENSE).
