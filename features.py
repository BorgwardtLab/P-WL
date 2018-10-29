'''
Contains methods and classes for generating features of unlabelled
graphs.
'''


import collections

import igraph as ig
import numpy as np


from topology import PersistenceDiagram
from topology import PersistenceDiagramCalculator


class WeightAssigner:
    '''
    Given a labelled graph, this class assigns weights based on
    a distance metric and returns the weighted graph.
    '''

    def __init__(self, ignore_label_order=False, metric='hamming', base_weight=1.0):
        self._ignore_label_order = ignore_label_order
        self._similarity = None
        self._base_weight = base_weight

        # Select metric to use in the `fit_transform()` function later
        # on. All of these metrics need to support multi-sets.
        metric_map = {
            'angular':   self._angular,
            'canberra':  self._canberra,
            'hamming':   self._hamming,
            'jaccard':   self._jaccard,
            'minkowski': self._minkowski,
            'sorensen':  self._sorensen,
        }

        if metric not in metric_map:
            raise RuntimeError('Unknown metric \"{}\" requested'.format(metric))

        self._metric = metric_map[metric]

    def fit_transform(self, graph):

        for edge in graph.es:
            source, target = edge.tuple

            source_labels = self._ensure_list(graph.vs[source]['label'])
            target_labels = self._ensure_list(graph.vs[target]['label'])

            source_label = source_labels[0]
            target_label = target_labels[0]

            weight = self._metric(source_labels[1:], target_labels[1:])
            weight = weight + (source_label != target_label)
            weight = weight + self._base_weight
            edge['weight'] = weight

        return graph

    def _ensure_list(self, l):
        '''
        Ensures that the input data is a list. Thus, if the input data
        is a single element, it will be converted to a list containing
        a single element.
        '''

        if type(l) is not list:
            return [l]
        else:
            return l

    def _angular(self, A, B):
        a, b = self._to_vectors(A, B)

        denominator = np.linalg.norm(a) * np.linalg.norm(b)

        # This should not happen for normal graphs, but let's be
        # prepared for it nonetheless.
        if denominator == 0.0:
            return 0.0

        cosine_similarity = np.clip(np.dot(a, b) / denominator, -1, 1)
        return 2 * np.arccos(cosine_similarity) / np.pi

    def _canberra(self, A, B):
        a, b = self._to_vectors(A, B)
        return np.sum(np.abs(a - b) / (a + b))

    def _hamming(self, A, B):
        '''
        Computes the (normalized) Hamming distance between two sets of
        labels A and B. This amounts to counting how many overlaps are
        present in the sequences.
        '''

        n = len(A) + len(B)

        # Empty lists are always treated as being equal
        if n == 0:
            return 0.0

        counter = collections.Counter(A)
        counter.subtract(B)

        num_missing = sum([abs(c) for _, c in counter.most_common()])
        return num_missing / n

    def _jaccard(self, A, B):
        a, b = self._to_vectors(A, B)
        n = len(a)

        denominator = (n - np.sum(np.multiply(1 - a, 1 - b)))

        # This follows the standard definition of multi-set distances
        if denominator == 0.0:
            return 0.0

        return np.sum(np.abs(a - b)) / denominator

    def _minkowski(self, A, B):
        # TODO: make configurable
        self._p = 2

        a, b = self._to_vectors(A, B)
        return np.linalg.norm(a - b, ord=self._p)

    def _sorensen(self, A, B):
        a, b = self._to_vectors(A, B)

        denominator = np.sum(a + b)

        # This follows the standard definition of multi-set distances;
        # it should never happen for this distance but let's be on the
        # safe side for once.
        if denominator == 0.0:
            return 0.0

        return np.sum(np.abs(a - b)) / denominator

    @staticmethod
    def _to_vectors(A, B):
        '''
        Transforms two sets of labels to their corresponding
        high-dimensional vectors. For example, a sequence of
        `{a, a, b}` and `{a, c, c}` will be transformed to a
        vector `(2, 1, 0)` and `(1, 0, 2)`, respectively.

        This function does not have to care about the global
        alphabet of labels because they will only yield zero
        values.

        :param A: First label sequence
        :param B: Second label sequence

        :return: Two transformed vectors
        '''

        label_to_index = dict()
        index = 0
        for label in A + B:
            if label not in label_to_index:
                label_to_index[label] = index
                index += 1

        a = np.zeros(len(label_to_index))
        b = np.zeros(len(label_to_index))

        for label in A:
            a[label_to_index[label]] += 1

        for label in B:
            b[label_to_index[label]] += 1

        return a, b


class PersistenceFeaturesGenerator:
    '''
    Creates persistence-based features of a sequence of weighted graphs.
    '''

    def __init__(self,
                 use_infinity_norm=False,
                 use_total_persistence=True,
                 use_label_persistence=False,
                 p=1.0):
        self._use_infinity_norm = use_infinity_norm
        self._use_total_persistence = use_total_persistence
        self._use_label_persistence = use_label_persistence
        self._p = 1.0

        if p <= 0.0:
            raise RuntimeError('Power parameter must be non-negative')

    def fit_transform(self, graphs):
        '''
        Calculates the feature vector of a sequence of graphs. The
        graphs are assumed to be weighted such that persistence is
        a suitable invariant.
        '''

        num_labels = 0

        # Calculating label persistence requires us to know the number
        # of distinct labels in the set of graphs as it determines the
        # length of the created feature vector.
        if self._use_label_persistence:
            labels = set()

            for graph in graphs:
                labels.update(graph.vs['compressed_label'])

            num_labels = len(labels)

            # Ensures that the labels form a contiguous sequence of
            # indices so that they can be easily mapped.
            assert min(labels) == 0
            assert max(labels) == num_labels - 1

        num_rows = len(graphs)
        num_columns = self._use_infinity_norm   \
            + self._use_total_persistence       \
            + self._use_label_persistence * num_labels

        X = np.zeros((num_rows, num_columns))

        for index, graph in enumerate(graphs):

            x_infinity_norm = []      # Optionally contains the infinity norm of the diagram
            x_total_persistence = []  # Optionally contains the total persistence of the diagram
            x_label_persistence = []  # Optionally contains the label persistence as a vector

            pdc = PersistenceDiagramCalculator()
            persistence_diagram = pdc.fit_transform(graph)

            if self._use_infinity_norm:
                x_infinity_norm = [persistence_diagram.infinity_norm(self._p)]

            if self._use_total_persistence:
                x_total_persistence = [persistence_diagram.total_persistence(self._p)]

            if self._use_label_persistence:
                x_label_persistence = np.zeros(num_labels)

                for x, y, c in persistence_diagram:
                    label = graph.vs[c]['compressed_label']
                    persistence = abs(x - y)**self._p
                    x_label_persistence[label] += persistence

            X[index, :] = np.concatenate((x_infinity_norm,
                                          x_total_persistence,
                                          x_label_persistence))

        return X
