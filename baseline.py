#!/usr/bin/env python3
#
# baseline.py: script for evaluating the baseline Weisfeiler--Lehman
# graph kernel on an input data set.


import graphkernels as gk
import igraph as ig
import numpy as np

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

import argparse
import logging


# TODO: shamelessly copied from `main.py`; should become a separate
# function somewhere
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

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('Baseline')

    graphs = [ig.read(filename) for filename in args.FILES]
    labels = read_labels(args.labels)

    # Set the label to be uniform over all graphs in case no labels are
    # available. This essentially changes our iteration to degree-based
    # checks.
    for graph in graphs:
        if 'label' not in graph.vs.attributes():
            graph.vs['label'] = [0] * len(graph.vs)
 
    assert len(graphs) == len(labels)

    y = np.array(labels)
    K = gk.CalculateWLKernel(graphs, args.num_iterations)
    n, _ = K.shape

    np.random.seed(42)
    cv = StratifiedKFold(n_splits=10, shuffle=True)
    mean_accuracies = []

    for i in range(10):

        # Contains accuracy scores for each cross validation step; the
        # means of this list will be used later on.
        accuracy_scores = []

        for train_index, test_index in cv.split(np.arange(n), y):
            # TODO: offer grid search capabilities here to be really
            # fair and all...
            clf = SVC(kernel='precomputed', C=1)

            y_train, y_test = y[train_index], y[test_index]

            K_train = K[train_index][:, train_index]
            K_test = K[test_index][:, train_index]

            clf.fit(K_train, y_train)
            y_pred = clf.predict(K_test)

            accuracy_scores.append(accuracy_score(y_test, y_pred))

        mean_accuracies.append(np.mean(accuracy_scores))
        print('  - Mean 10-fold accuracy: {:2.2f} [running mean over all folds: {:2.2f}]'.format(mean_accuracies[-1] * 100, np.mean(mean_accuracies) * 100))

    print('Accuracy: {:2.2f} +- {:2.2f}'.format(np.mean(mean_accuracies) * 100, np.std(mean_accuracies) * 100))
