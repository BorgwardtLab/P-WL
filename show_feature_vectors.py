#!/usr/bin/env python3
#
# show_feature_vectors.py: script for displaying the feature vectors of
# the Persistent Weisfeiler--Lehman kernel.


import igraph as ig
import numpy as np

import matplotlib.pyplot as plt

import argparse
import collections
import logging


from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from features import FeatureSelector
from features import PersistentWeisfeilerLehman


from utilities import read_labels


def main(args, logger):

    graphs = [ig.read(filename) for filename in args.FILES]
    labels = read_labels(args.labels)

    # Set the label to be uniform over all graphs in case no labels are
    # available. This essentially changes our iteration to degree-based
    # checks.
    for graph in graphs:
        if 'label' not in graph.vs.attributes():
            graph.vs['label'] = [0] * len(graph.vs)

    logger.info('Read {} graphs and {} labels'.format(len(graphs), len(labels)))

    assert len(graphs) == len(labels)

    pwl = PersistentWeisfeilerLehman(
            use_cycle_persistence=args.use_cycle_persistence,
            use_original_features=args.use_original_features,
            use_label_persistence=args.use_persistence_features,
    )

    if args.use_cycle_persistence:
        logger.info('Using cycle persistence')

    y = LabelEncoder().fit_transform(labels)
    X, num_columns_per_iteration = pwl.transform(graphs, args.num_iterations)

    X = StandardScaler().fit_transform(X)
    X = MinMaxScaler().fit_transform(X)

    logger.info('Finished persistent Weisfeiler-Lehman transformation')
    logger.info('Obtained ({} x {}) feature matrix'.format(X.shape[0], X.shape[1]))

    num_classes = len(np.bincount(y))

    fig, ax = plt.subplots(
        nrows=num_classes,
        ncols=2,
        sharex=True,
        sharey=False,
        squeeze=False
    )

    for index in range(num_classes):
        ax[index][0].matshow(X[y == index], aspect='auto')
        ax[index][0].set_title(f'Class {index} (features)')

        ax[index][1].matshow(np.mean(X[y == index], axis=0).reshape(1, -1), aspect='auto')
        ax[index][1].set_title(f'Class {index} (mean)')

    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('FILES', nargs='+', help='Input graphs (in some supported format)')
    parser.add_argument('-d', '--dataset', help='Name of data set')
    parser.add_argument('-l', '--labels', type=str, help='Labels file', required=True)
    parser.add_argument('-n', '--num-iterations', default=3, type=int, help='Number of Weisfeiler-Lehman iterations')
    parser.add_argument('-f', '--filtration', type=str, default='sublevel', help='Filtration type')
    parser.add_argument('-g', '--grid-search', action='store_true', default=False, help='Whether to do hyperparameter grid search')
    parser.add_argument('-c', '--use-cycle-persistence', action='store_true', default=False, help='Indicates whether cycle persistence should be calculated or not')
    parser.add_argument('-o', '--use-original-features', action='store_true', default=False, help='Indicates that original features should be used as well')
    parser.add_argument('-p', '--use-persistence-features', action='store_true', help='Indicates that standard persistence-based features should be used')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG,
        filename='{}_{:02d}.log'.format(args.dataset, args.num_iterations)
    )

    logger = logging.getLogger('P-WL')

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
