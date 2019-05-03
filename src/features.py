'''
Contains methods and classes for generating features of unlabelled
graphs.
'''


import collections

import igraph as ig
import numpy as np

from topology import PersistenceDiagram
from topology import PersistenceDiagramCalculator

from weisfeiler_lehman import WeisfeilerLehman

from scipy.stats import entropy
from sklearn.base import TransformerMixin


class WeightAssigner:
    '''
    Given a labelled graph, this class assigns weights based on
    a distance metric and returns the weighted graph.
    '''

    def __init__(self, metric='minkowski', p=1.0, tau=1.0, smooth=False):
        self._p = p
        self._tau = tau

        # Select metric to use in the `fit_transform()` function later
        # on. All of these metrics need to support multi-sets.
        metric_map = {
            'angular': self._angular,
            'canberra': self._canberra,
            'jaccard': self._jaccard,
            'jensen_shannon': self._jensen_shannon,
            'kullback_leibler': self._kullback_leibler,
            'minkowski': self._minkowski,
            'uniform': self._uniform,  # Not a 'real' metric
            'sorensen': self._sorensen,
        }

        if metric not in metric_map:
            raise RuntimeError(
                '''
                Unknown metric \"{}\" requested
                '''.format(metric)
            )

        self._metric = metric_map[metric]
        self._smooth = smooth

    def fit_transform(self, graph):

        for edge in graph.es:
            source, target = edge.tuple

            source_labels = self._ensure_list(graph.vs[source]['label'])
            target_labels = self._ensure_list(graph.vs[target]['label'])

            source_label = source_labels[0]
            target_label = target_labels[0]

            weight = self._metric(source_labels[1:], target_labels[1:])

            # For all non-uniform metrics, we want to take into account
            # the differences between the source and target label of an
            # edge.
            if self._metric != self._uniform:
                weight = weight + (source_label != target_label) + self._tau

            # Update the edge weight if smoothing is required for the
            # distances.
            if self._smooth:
                edge['weight'] += weight
            else:
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

    def _jaccard(self, A, B):
        a, b = self._to_vectors(A, B)
        n = len(a)

        denominator = (n - np.sum(np.multiply(1 - a, 1 - b)))

        # This follows the standard definition of multi-set distances
        if denominator == 0.0:
            return 0.0

        return np.sum(np.abs(a - b)) / denominator

    def _jensen_shannon(self, A, B):
        return self._kullback_leibler(A, B, lambda_=0.5)

    def _kullback_leibler(self, A, B, lambda_=1.0):
        a, b = self._to_vectors(A, B)

        # Ensures that entries are valid
        a += 1
        b += 1

        # Make them into valid probability distributions; `scipy` does
        # that for us as well but explicit is better than implicit.
        a /= np.sum(a)
        b /= np.sum(b)

        # Ensures that at least symmetry holds
        return lambda_ * entropy(a, b) + lambda_ * entropy(b, a)

    def _minkowski(self, A, B):
        a, b = self._to_vectors(A, B)
        return np.linalg.norm(a - b, ord=self._p)

    def _uniform(self, A, B):
        return 1.0

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
    This requires a choice of metric, as well as an exponent, which will
    be used to perform persistent homology calculations.
    '''

    def __init__(self,
                 use_infinity_norm,
                 use_total_persistence,
                 use_label_persistence,
                 use_cycle_persistence,
                 use_original_features,
                 store_persistence_diagrams,
                 p):
        self._use_infinity_norm = use_infinity_norm
        self._use_total_persistence = use_total_persistence
        self._use_label_persistence = use_label_persistence
        self._use_cycle_persistence = use_cycle_persistence
        self._use_original_features = use_original_features
        self._p = p
        self._store_persistence_diagrams = store_persistence_diagrams

        if p <= 0.0:
            raise RuntimeError('Power parameter must be non-negative')

    def fit_transform(self, graphs):
        '''
        Calculates the feature vector of a sequence of graphs. The
        graphs are assumed to be weighted such that persistence is
        a suitable invariant.

        :param graphs: Sequence of weighted graphs

        :return:  Feature matrix
        '''

        num_labels = 0

        if self._store_persistence_diagrams:
            self._persistence_diagrams = []

        # Calculating label persistence requires us to know the number
        # of distinct labels in the set of graphs as it determines the
        # length of the created feature vector.
        if self._use_label_persistence or \
           self._use_original_features or \
           self._use_cycle_persistence:
            labels = set()

            for graph in graphs:
                labels.update(graph.vs['compressed_label'])

            num_labels = len(labels)

            # Ensures that the labels form a contiguous sequence of
            # indices so that they can be easily mapped.
            assert min(labels) == 0
            assert max(labels) == num_labels - 1

        # Calculate the space required for the feature vector matrix.
        # This depends on the attributes selected by the client.

        num_rows = len(graphs)
        num_columns = self._use_infinity_norm          \
            + self._use_total_persistence              \
            + self._use_label_persistence * num_labels \
            + self._use_original_features * num_labels \
            + self._use_cycle_persistence * num_labels

        X = np.zeros((num_rows, num_columns))

        # Fill the feature matrix by calculating persistence-based
        # features for each of the graphs, using the initial
        # configuration.

        for index, graph in enumerate(graphs):

            # Initially, all of these vectors are empty and will only be
            # filled depending on the client configuration.
            x_infinity_norm = []
            x_total_persistence = []
            x_label_persistence = []
            x_cycle_persistence = []
            x_original_features = []

            pdc = PersistenceDiagramCalculator()
            persistence_diagram, edge_indices_cycles = pdc.fit_transform(graph)

            if self._use_infinity_norm:
                x_infinity_norm = \
                    [persistence_diagram.infinity_norm(self._p)]

            if self._use_total_persistence:
                x_total_persistence = \
                    [persistence_diagram.total_persistence(self._p)]

            if self._use_label_persistence:
                x_label_persistence = np.zeros(num_labels)

                for x, y, c in persistence_diagram:
                    label = graph.vs[c]['compressed_label']
                    persistence = abs(x - y)**self._p
                    x_label_persistence[label] += persistence

                if self._store_persistence_diagrams:
                    self._persistence_diagrams.append(persistence_diagram)

            # Add the original features of the Weisfeiler--Lehman
            # iteration to the feature matrix. This can be easily
            # done by just counting '1' for each point in the PD.
            if self._use_original_features:
                x_original_features = np.zeros(num_labels)

                for _, _, c in persistence_diagram:
                    label = graph.vs[c]['compressed_label']
                    x_original_features[label] += 1

            # Cycle persistence: use the edge information, i.e. the
            # classification gained from the persistence diagram
            # calculation above, in order to assign weights.
            if self._use_cycle_persistence:
                n = len(persistence_diagram)
                m = graph.ecount()
                k = persistence_diagram.betti
                num_cycles = m - n + k

                # If this assertion fails, there's something seriously
                # wrong with our understanding of cycles.
                assert num_cycles == len(edge_indices_cycles)

                x_cycle_persistence = np.zeros(num_labels)
                total_cycle_persistence = 0.0

                for edge_index in edge_indices_cycles:
                    edge = graph.es[edge_index]
                    weight = edge['weight']

                    total_cycle_persistence += weight**self._p

                    source_label = graph.vs[edge.source]['compressed_label']
                    target_label = graph.vs[edge.target]['compressed_label']

                    x_cycle_persistence[source_label] += weight**self._p
                    x_cycle_persistence[target_label] += weight**self._p

            X[index, :] = np.concatenate((x_infinity_norm,
                                          x_total_persistence,
                                          x_label_persistence,
                                          x_original_features,
                                          x_cycle_persistence))

        return X


class PersistentWeisfeilerLehman:
    '''
    Main class for generating persistence-based Weisfeiler--Lehman
    features. Actually, this class should contain 'subtree' in its
    name because it *only* handles subtree features for now.
    '''

    def __init__(self,
                 use_infinity_norm=False,
                 use_total_persistence=False,
                 use_label_persistence=False,
                 use_cycle_persistence=False,
                 use_original_features=False,
                 store_persistence_diagrams=False,
                 metric='minkowski',
                 p=2.0,
                 smooth=False):
        '''
        TODO: describe parameters
        '''

        self._use_infinity_norm = use_infinity_norm
        self._use_total_persistence = use_total_persistence
        self._use_label_persistence = use_label_persistence
        self._use_cycle_persistence = use_cycle_persistence
        self._use_original_features = use_original_features
        self._store_persistence_diagrams = store_persistence_diagrams
        self._original_labels = None
        self._metric = metric
        self._p = p
        self._smooth = smooth

    def transform(self, graphs, num_iterations):
        wl = WeisfeilerLehman()

        # If the *uniform* metric was selected, the transformation
        # degenerates into the regular Weisfeiler--Lehman subtree,
        # also known as WL-subtree, computation. The twist is that
        # we can *also* add cycles.
        wa = WeightAssigner(
            metric=self._metric,
            p=self._p,
            smooth=self._smooth,
        )

        pfg = PersistenceFeaturesGenerator(
                use_infinity_norm=self._use_infinity_norm,
                use_total_persistence=self._use_total_persistence,
                use_label_persistence=self._use_label_persistence,
                use_cycle_persistence=self._use_cycle_persistence,
                use_original_features=self._use_original_features,
                store_persistence_diagrams=self._store_persistence_diagrams,
                p=self._p)

        # Performs *all* steps of Weisfeiler--Lehman for the pre-defined
        # number of iterations.
        label_dicts = wl.fit_transform(graphs, num_iterations)

        X_per_iteration = []
        num_columns_per_iteration = {}

        if self._store_persistence_diagrams:
            self._persistence_diagrams = collections.defaultdict(list)

        # Stores the *original* labels in the original graph for
        # subsequent forward propagation.
        original_labels = collections.defaultdict(list)

        # It is sufficient to copy the graphs *once*; their edge weights
        # will be reset anyway (unless 'smoothed' distances are selected
        # by the client).
        weighted_graphs = [graph.copy() for graph in graphs]

        for iteration in sorted(label_dicts.keys()):

            for graph_index in sorted(label_dicts[iteration].keys()):
                labels_raw, labels_compressed = \
                    label_dicts[iteration][graph_index]

                weighted_graphs[graph_index].vs['label'] = \
                    labels_raw

                weighted_graphs[graph_index].vs['compressed_label'] = \
                    labels_compressed

                # Assign the *compressed* labels as the *original*
                # labels of the graph in order to ensure that they
                # are zero-indexed.
                if iteration == 0:
                    original_labels[graph_index] = labels_compressed

                # Use labels from the *previous* iteration to assign the
                # *original* label.
                else:
                    labels = original_labels[graph_index]
                    weighted_graphs[graph_index]['original_label'] = labels

                weighted_graphs[graph_index] = \
                    wa.fit_transform(weighted_graphs[graph_index])

            X_per_iteration.append(pfg.fit_transform(weighted_graphs))

            if self._store_persistence_diagrams:
                self._persistence_diagrams[iteration] = \
                    pfg._persistence_diagrams

            if iteration not in num_columns_per_iteration:
                num_columns_per_iteration[iteration] = \
                    X_per_iteration[-1].shape[1]

            # Make sure that we did not remove the *wrong* number of
            # columns.
            assert num_columns_per_iteration[iteration] == \
                X_per_iteration[-1].shape[1]

        # Store original labels only if there is something to store.
        # Notice that these labels are *standardized*, i.e. they are
        # zero-indexed.
        if original_labels:
            self._original_labels = original_labels

        return np.concatenate(X_per_iteration, axis=1), \
            num_columns_per_iteration


class WeisfeilerLehmanSubtree:
    '''
    Class for generating Weisfeiler--Lehman subtree features, following
    the original paper on graph kernels by Shervashidze et al.; one may
    also rephrase this in terms of a graph with _uniform_ weights. Yet,
    in the interest of readability, we provide a separate class.
    '''

    def __init__(self):
        pass

    def transform(self, graphs, num_iterations):

        # Performs *all* steps of Weisfeiler--Lehman for the pre-defined
        # number of iterations.
        wl = WeisfeilerLehman()
        label_dicts = wl.fit_transform(graphs, num_iterations)

        X_per_iteration = []
        num_columns_per_iteration = {}

        for iteration in sorted(label_dicts.keys()):

            # Make the feature vector assignment easier for each of the
            # 'subtree graphs'.
            wl_graphs = [graph.copy() for graph in graphs]

            for graph_index in sorted(label_dicts[iteration].keys()):
                labels_raw, labels_compressed = \
                    label_dicts[iteration][graph_index]

                # Assign the compressed label (an integer) to the
                # 'subtree graph' in order to generate features.
                wl_graphs[graph_index].vs['label'] = labels_compressed

            X_per_iteration.append(
                self.get_subtree_feature_vectors(wl_graphs)
            )

            if iteration not in num_columns_per_iteration:
                num_columns_per_iteration[iteration] = \
                    X_per_iteration[-1].shape[1]

        return np.concatenate(X_per_iteration, axis=1), \
            num_columns_per_iteration

    def get_subtree_feature_vectors(self, graphs):
        '''
        Calculates the feature vectors of a sequence of graphs. The
        `label` attribute is used to calculate features.
        '''

        num_labels = 0
        labels = set()

        for graph in graphs:
            labels.update(graph.vs['label'])

        num_labels = len(labels)

        # Ensures that the labels form a contiguous sequence of
        # indices so that they can be easily mapped.
        assert min(labels) == 0
        assert max(labels) == num_labels - 1

        # Increases readability and follows the 'persistent' feature
        # generation method.
        num_rows = len(graphs)
        num_columns = num_labels

        X = np.zeros((num_rows, num_columns))

        for index, graph in enumerate(graphs):

            # Features, i.e. label counts, for the current graph
            x = np.zeros(num_columns)

            for label in graph.vs['label']:
                x[label] += 1

            X[index, :] = x

        return X


class WeisfeilerLehmanAttributePropagation:
    '''
    Simple Weisfeiler--Lehman feature propagation class. Given a set of
    graphs and an attribute name, this class progressively smooths them
    by replacing attributes by their *average* in a neighbourhood.
    '''

    def __init__(self):
        pass

    def transform(self, graphs, attribute, num_iterations):

        # Will contain the respective attribute values for all graphs,
        # indexed by the iteration.
        attributes = collections.defaultdict(list)

        for graph in graphs:
            attributes[0].append(np.array(graph.vs[attribute]))

        for iteration in range(1, num_iterations + 1):
            for index, graph in enumerate(graphs):
                attributes_per_vertex = np.array(graph.vs[attribute])

                for edge in graph.es:
                    source = edge.source
                    target = edge.target

                    source_attribute = graph.vs[attribute][source]
                    target_attribute = graph.vs[attribute][target]

                    # Switch the calculation here: the *source* vertex
                    # contributes once to the *target* attribute while
                    # the *target* vertex contributes to *source*.
                    attributes_per_vertex[source] += target_attribute
                    attributes_per_vertex[target] += source_attribute

                # Normalize the number of attributes again using the
                # number of neighbours of the node.
                attributes_per_vertex = np.divide(
                    attributes_per_vertex, np.array(graph.vs.degree()) + 1
                )

                # Store the new attributes.
                attributes[iteration].append(
                    attributes_per_vertex
                )

                # Store the new attributes in the graph because the
                # smoothing process needs to continue in subsequent
                # iterations.
                graph.vs[attribute] = attributes_per_vertex

        return attributes


class FeatureSelector(TransformerMixin):
    def __init__(self, num_columns_per_iteration):
        self._num_columns_per_iteration = num_columns_per_iteration

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)

    def fit(self, X, y=None, **params):
        return self

    def transform(self, X):
        return self.fit_transform(X)

    def fit_transform(self, X, y=None, **params):

        # Determine the number of columns to select for the desired
        # number of iterations.

        last_column = 0
        for iteration in range(0, self.num_iterations + 1):
            last_column += self._num_columns_per_iteration[iteration]

        return X[:, :last_column]
