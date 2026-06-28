"""
CYTOS - Tree Tensor Networks vs Graph Neural Networks para dinamica de
redes regulatorias genicas (ou qualquer grafo com dinamica temporal).

Uso basico:
    from cytos import TTNvsGNN
    result = TTNvsGNN(graph=meu_grafo, trajectories=minhas_trajetorias).run(seeds=20)
    print(result.summary())
"""

from cytos.experiment import TTNvsGNN, TTNvsGNNResult
from cytos.entanglement import EntanglementPilot, EntanglementPilotResult

__all__ = ["TTNvsGNN", "TTNvsGNNResult", "EntanglementPilot", "EntanglementPilotResult"]
__version__ = "0.1.0"
