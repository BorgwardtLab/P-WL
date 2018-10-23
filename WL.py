from sklearn.base import TransformerMixin
from igraph import *

from typing import List

class WL(TransformerMixin):
	
    def fit_transform(self, X: Graph, num_iterations: int=3):
        for it in range(i):
            multi_set = self._get_multiset(X, iteration=it)
            sorted_multiset = self._sort_multiset(multi_set)
            # Add l_{it-1} as prefix
            label_dict = self._compress_labels(sorted_multiset)
            relabeled_graph = self._relabel(X, label_dict) 

    def _get_multiset(self, X: Graph, iteration: int):
        if iteration == 0:
            pass
        else:
            pass

    def _sort_multi_set(self, multiset: List[list]) -> List[str]:
        pass
    
    def _compress_labels(self, List[str]) -> dict:
        pass

    def _relabel(self, X: Graph, label_dict: dict) -> Graph:
        pass
