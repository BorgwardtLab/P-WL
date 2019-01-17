#!/usr/bin/env python3
#
# label_sequence_distance.py: calculates a Weisfeiler--Lehman label
# sequence and uses it to assign distances between the vertices. As
# a result, a distance matrix will be stored.

import igraph as ig
import numpy as np

import argparse
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from features import PersistenceFeaturesGenerator
from features import WeisfeilerLehman

from utilities import read_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('FILES', nargs='+', help='Input graphs (in some supported format)')
    parser.add_argument('-b', '--balanced', action='store_true', help='Make random forest classifier balanced')
    parser.add_argument('-l', '--labels', type=str, help='Labels file', required=True)
    parser.add_argument('-n', '--num-iterations', default=3, type=int, help='Number of Weisfeiler-Lehman iterations')

    args = parser.parse_args()
    graphs = [ig.read(filename) for filename in args.FILES]
    y = np.array(read_labels(args.labels))

    wl = WeisfeilerLehman()
    label_dicts = wl.fit_transform(graphs, args.num_iterations)

    # Each entry in the list represents the label sequence of a single
    # graph. The label sequence contains the vertices in its rows, and
    # the individual iterations in its columns.
    #
    # Hence, (i, j) will contain the label of vertex i at iteration j.
    label_sequences = [
        np.full((len(graph.vs), args.num_iterations + 1), np.nan) for graph in graphs
    ]

    for iteration in sorted(label_dicts.keys()):
        for graph_index, graph in enumerate(graphs):
            labels_raw, labels_compressed = label_dicts[iteration][graph_index]

            # Store label sequence of the current iteration, i.e. *all*
            # of the compressed labels.
            label_sequences[graph_index][:, iteration] = labels_compressed

    weighted_graphs = []

    # Transform the label sequence into a matrix of distances by
    # calculating the Hamming distance. Since we are technically
    # in a discrete space, this is the *only* suitable distance.
    for graph_index, graph in enumerate(graphs):
        labels = label_sequences[graph_index]
        distances = (labels[:, None, :] != labels).sum(2)

        # Normalize distances to [0, 1] because they depend on the
        # number of labelling iterations.
        distances = distances / (args.num_iterations + 1)

        # Get the compressed labels from the first iteration as this
        # ensures that they are contiguous.
        _, labels_compressed = label_dicts[0][graph_index]

        weighted_graph = graph.copy()
        weighted_graph.vs['compressed_label'] = labels_compressed

        for e in weighted_graph.es:
            e['weight'] = distances[e.tuple]

        weighted_graphs.append(weighted_graph)

    pfg = PersistenceFeaturesGenerator(
        use_infinity_norm=False,
        use_total_persistence=False,
        use_cycle_persistence=False,
        use_label_persistence=True,
        use_original_features=False,
        store_persistence_diagrams=False,
        p=2.0
    )

    X = pfg.fit_transform(weighted_graphs)

    np.random.seed(42)
    cv = StratifiedKFold(n_splits=10, shuffle=True)
    mean_accuracies = []

    for i in range(10):

        # Contains accuracy scores for each cross validation step; the
        # means of this list will be used later on.
        accuracy_scores = []

        for train_index, test_index in cv.split(X, y):
            clf = RandomForestClassifier(
                n_estimators=50,
                class_weight='balanced' if args.balanced else None
            )

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            accuracy_scores.append(accuracy_score(y_test, y_pred))

        mean_accuracies.append(np.mean(accuracy_scores))

        print('  - Mean 10-fold accuracy: {:2.2f} [running mean over all folds: {:2.2f}]'.format(mean_accuracies[-1] * 100, np.mean(mean_accuracies) * 100))

    print('Accuracy: {:2.2f} +- {:2.2f}'.format(np.mean(mean_accuracies) * 100, np.std(mean_accuracies) * 100))
