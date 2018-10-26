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

    assert len(graphs) == len(labels)

    y = np.array(labels)
    K = gk.CalculateWLKernel(graphs, args.num_iterations)
    n, _ = K.shape

    cv = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
    accuracy_scores = []

    for train_index, test_index in cv.split(np.arange(n), y):
        clf = SVC(kernel='precomputed', C=1)

        y_train = y[train_index]
        y_test = y[test_index]

        K_train = K[train_index][:, train_index]
        K_test = K[test_index][:, train_index]

        clf.fit(K_train, y_train)
        y_pred = clf.predict(K_test)

        accuracy_scores.append(accuracy_score(y_test, y_pred))

    print('Accuracy: {:2.2f} +- {:2.2f}'.format(np.mean(accuracy_scores) * 100, np.std(accuracy_scores) * 100 * 2))
