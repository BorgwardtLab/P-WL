#!/usr/bin/env python3
#
# p-wl_unlabelled.py: script for calculating persistent Weisfeiler--Lehman
# features for graphs that have no attributes and no (node|edge) labels.


import igraph as ig
import numpy as np

import argparse
import collections
import logging
import itertools

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from features import WeisfeilerLehmanAttributePropagation

from kernels import PersistenceScaleSpaceKernel

from utilities import read_labels

from topology import assign_filtration_values
from topology import multiscale_persistence_diagram_kernel
from topology import PersistenceDiagramCalculator


def main(args, logger):

    graphs = [ig.read(filename) for filename in args.FILES]
    labels = read_labels(args.labels)

    for graph in graphs:

        # Make sure that no label information exists as a graph
        # attribute already.
        assert 'label' not in graph.vs.attributes()

        graph.vs['degree'] = graph.vs.degree()

    logger.info('Read {} graphs and {} labels'.format(len(graphs), len(labels)))

    assert len(graphs) == len(labels)

    prop = WeisfeilerLehmanAttributePropagation()
    attributes_per_iteration = prop.transform(
        graphs,
        'degree',
        args.num_iterations
    )

    # TODO: make configurable
    use_vertex_weights = False

    if use_vertex_weights:
        pdc = PersistenceDiagramCalculator(vertex_attribute='degree')
    else:
        pdc = PersistenceDiagramCalculator()

    # Stores *all* persistence diagrams because they will be used to
    # represent the data set later on.
    persistence_diagrams_per_iteration = collections.defaultdict(list)

    for iteration in sorted(attributes_per_iteration.keys()):
        for index, graph in enumerate(graphs):
            attributes = attributes_per_iteration[iteration][index]
            graph.vs['degree'] = attributes
            weighted_graph = assign_filtration_values(graph, attributes)

            pd, edge_indices_cycles = pdc.fit_transform(graph)

            # Store the persistence diagram as a 2D array in order to
            # facilitate the subsequent kernel calculations.
            persistence_diagrams_per_iteration[iteration].append(
                np.array([(c, d) for c, d, _ in pd])
            )

    # Will contain the full kernel matrix over all iterations; it is
    # composed of sums of kernel matrices for individual iterations.
    K = np.zeros((len(graphs), len(graphs)))

    # Use this as the kernel for evaluating individual persistence
    # diagrams
    #pss = PersistenceScaleSpaceKernel(sigma=1.0)

    # Prepare kernel matrix _per iteration_; since this is a kernel, we
    # can just sum over individual iterations
    for iteration in sorted(persistence_diagrams_per_iteration.keys()):
        persistence_diagrams = persistence_diagrams_per_iteration[iteration]
        n = len(persistence_diagrams)
        K_iteration = np.zeros((n, n))

        # We need to use `combinations_with_replacement` because the
        # diagonal elements of the kernel are relevant as well. This
        # is *not* a metric, after all.
        for i, j in itertools.combinations_with_replacement(range(n), 2):
            K_iteration[i, j] = multiscale_persistence_diagram_kernel(
                persistence_diagrams[i],
                persistence_diagrams[j],
                sigma=1  # TODO: make configurable
            )

            K_iteration[j, i] = K_iteration[i, j]

        # TODO: make this configurable?
        K += 1 / (iteration + 1)**2 * K_iteration

    y = LabelEncoder().fit_transform(labels)
    cv = StratifiedKFold(
            n_splits=10,
            shuffle=True,
    )
    mean_scores = []

    np.random.seed(42)

    for i in range(10):
        scores = []
        for train, test in cv.split(np.zeros(len(y)), y):
            K_train = K[train][:, train]
            y_train = y[train]

            K_test = K[test][:, train]
            y_test = y[test]

            clf = SVC(kernel='precomputed', C=1)
            clf.fit(K_train, y_train)

            y_pred = clf.predict(K_test)
            scores.append(accuracy_score(y_test, y_pred))

        mean_scores.append(np.mean(scores))

    print('Accuracy: {:2.2f} +- {:2.2f}'.format(
            np.mean(mean_scores) * 100.0,
            np.std(mean_scores) * 100.0
        )
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('FILES', nargs='+', help='Input graphs (in some supported format)')
    parser.add_argument('-b', '--balanced', action='store_true', help='Make random forest classifier balanced')
    parser.add_argument('-d', '--dataset', help='Name of data set')
    parser.add_argument('-l', '--labels', type=str, help='Labels file', required=True)
    parser.add_argument('-n', '--num-iterations', default=3, type=int, help='Number of Weisfeiler-Lehman iterations')
    parser.add_argument('-f', '--filtration', type=str, default='sublevel', help='Filtration type')
    parser.add_argument('-g', '--grid-search', action='store_true', default=False, help='Whether to do hyperparameter grid search')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG,
        filename='{}_{:02d}.log'.format(args.dataset, args.num_iterations)
    )

    logger = logging.getLogger('P-WL [unlabelled]')

    # Create a second stream handler for logging to `stderr`, but set
    # its log level to be a little bit smaller such that we only have
    # informative messages
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    # Use the default format; since we do not adjust the logger before,
    # this is all right.
    stream_handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
    logger.addHandler(stream_handler)

    main(args, logger)
