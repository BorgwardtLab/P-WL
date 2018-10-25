#!/usr/bin/env python3
#
# main.py: main script for testing Persistent Weisfeiler--Lehman graph
# kernels.


import igraph as ig
import numpy as np

import argparse
import collections
import logging

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm

from features import PersistenceFeaturesGenerator
from features import WeightAssigner
from topology import PersistenceDiagramCalculator
from weisfeiler_lehman import WL


def read_labels(filename):
    labels = []
    with open(filename) as f:
        labels = f.readlines()
        labels = [label.strip() for label in labels]

    return labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('FILES', nargs='+', help='Input graphs (in some supported format)')
    parser.add_argument('-l', '--labels', type=str, help='Labels file', required=True)
    parser.add_argument('-n', '--num-iterations', default=3, type=int, help='Number of Weisfeiler-Lehman iterations')
    parser.add_argument('-f', '--filtration', type=str, default='sublevel', help='Filtration type')

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('P-WL')

    args = parser.parse_args()
    graphs = [ig.read(filename) for filename in args.FILES]
    labels = read_labels(args.labels)

    logger.debug('Read {} graphs and {} labels'.format(len(graphs), len(labels)))

    assert len(graphs) == len(labels)

    wl = WL()
    wa = WeightAssigner()
    pdc = PersistenceDiagramCalculator()  # FIXME: need to add order/filtration
    pfg = PersistenceFeaturesGenerator(use_infinity_norm=False,
                                       use_total_persistence=False,
                                       use_label_persistence=True)

    X = np.zeros((len(graphs), args.num_iterations + 1))
    y = np.array(labels)

    label_dicts = wl.fit_transform(graphs, args.num_iterations)

    X_per_iteration = []
    for iteration in tqdm(sorted(label_dicts.keys())):

        weighted_graphs = [graph.copy() for graph in graphs]

        for graph_index in sorted(label_dicts[iteration].keys()):
            labels_raw, labels_compressed = label_dicts[iteration][graph_index]

            weighted_graphs[graph_index].vs['label'] = labels_raw
            weighted_graphs[graph_index].vs['compressed_label'] = labels_compressed

            # TODO: rewrite weight assigner class to support lists
            weighted_graphs[graph_index] = wa.fit_transform(weighted_graphs[graph_index])

        X_per_iteration.append(pfg.fit_transform(weighted_graphs))

    X = np.concatenate(X_per_iteration, axis=1)

    cv = StratifiedKFold(n_splits=10)
    accuracy_scores = []

    np.savetxt('/tmp/X.txt', X)

    for train_index, test_index in cv.split(X, y):
        clf = SVC(kernel='precomputed')
        scaler = StandardScaler()

        K = pairwise_kernels(X, X, metric='linear')
        K_train = K[train_index][:, train_index]
        K_test = K[test_index][:, train_index]

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf.fit(K_train, y_train)
        y_pred = clf.predict(K_test)

        accuracy_scores.append(accuracy_score(y_test, y_pred))

    print('Accuracy: {:2.2f} +- {:2.2f}'.format(np.mean(accuracy_scores) * 100, np.std(accuracy_scores) * 100 * 2))
