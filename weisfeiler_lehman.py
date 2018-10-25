import copy
import numpy as np

from sklearn.base import TransformerMixin
from igraph import *
from LabelGenerator import LabelGenerator

from typing import List

import numpy as np

class LabelGenerator():

    def __init__(self, start, length):
        self._current_label_index = 0
        self._start = start
        self._length = length
        self._label_generator = np.nditer(np.arange(start, start+length, 1))

    def get_next_label(self):
        if self._current_label_index < self._length:
            self._current_label_index += 1
            return str(next(self._label_generator))
        else:
            print("reset")
            self._label_generator = np.nditer(np.arange(self._start, self._start+self._length, 1)) 
            self._current_label_index = 1
            return str(next(self._label_generator))
            
class WL(TransformerMixin):
    """
    Class that implements the Weisefeiler-Lehman transform 

    Attributes:
        _multisets (dict): key: iteration, value: graph multiset at itertation i
    """
    
    def __init__(self):
        self._multisets = {}
        self._graphs = {}

    def fit(self):
        pass

    def fit_transform(self, X: Graph, num_iterations: int=5):
        self._graphs[0] = X.copy()
        X = X.copy()
        for it in np.arange(1, num_iterations+1, 1):
            # Create a LabelGenerator object that can #nodes different labels
            self.label_generator = LabelGenerator(len(X.vs), len(X.vs))
            
            # Get labels of current interation
            current_labels = X.vs['label']
           
            # Get for each vertex the labels of its neighbors
            neighbor_labels = self._get_neighbor_labels(X, sort=True)
            
            # Prepend the vertex label to the list of labels of its neighbors
            merged_labels = [[b]+a for a,b in zip(neighbor_labels, current_labels)]
            self._multisets[it] = merged_labels
            
            # Generate a label dictionary based on the merged labels
            label_dict = self._generate_label_dict(merged_labels)

            # Relabel the graph
            self._relabel_graph(X, merged_labels, label_dict)
            self._graphs[it] = X.copy()
        return X

    def _relabel_graph(self, X: Graph, merged_labels: list, label_dict: dict):
        new_labels = [label_dict[''.join(merged)] for merged in merged_labels]
        X.vs['label'] = new_labels

    def _generate_label_dict(self, merged_labels: List[list]):
        label_dict = {}
        for merged_label in merged_labels:
            label_dict[ ''.join(merged_label) ] = self.label_generator.get_next_label()
        
        return label_dict

    def _get_neighbor_labels(self, X: Graph, sort: bool=True):
            neighbor_indices = [[n_v.index for n_v in X.vs[X.neighbors(v.index)]] for v in X.vs]
            neighbor_labels = []
            for n_indices in neighbor_indices:
                if sort:
                    neighbor_labels.append( sorted(X.vs[n_indices]['label']) )
                else:
                    neighbor_labels.append( X.vs[n_indices]['label'] )
            return neighbor_labels
