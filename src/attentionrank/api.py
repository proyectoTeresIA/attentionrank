


from pathlib import Path

from typing import List, Any, Union

from .attentionrank import AttentionRankModel
from .config import AttentionRankConfig


class AttentionRank:
    """
    API pública de la librería: inicializa la configuración,
    crea el modelo y permite ejecutar extracción sobre múltiples datasets.
    """

    def __init__(self, config: AttentionRankConfig):
        self.config = config
        self.model = AttentionRankModel(config)

    def extract(self, dataset: Union[str, Path, List[str]], k_values):

        results= self.model.extract_terms(dataset, "exec", k_values)
        return results

    def evaluate(self, dataset: Union[str, Path, List[str]], eval_dataset: Union[str, Path, List[str]], k_values):

        results= self.model.extract_terms(dataset,"eval",k_values)
        return results



    def __prune_values(self,data: List[List[Any]], n: int) -> List[List[Any]]:
        """
        Dada una lista de listas, devuelve cada sublista truncada
        a los n primeros elementos.

        Ejemplo:
            data = [[1,2,3], [4,5], [6,7,8,9]]
            truncate_sublists(data, 2)
            → [[1,2], [4,5], [6,7]]
        """
        return [sublist[:n] for sublist in data]






