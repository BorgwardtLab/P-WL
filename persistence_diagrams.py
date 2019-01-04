#!/usr/bin/env python3
#
# persistence_diagrams.py: creates persistence diagrams for Persistent
# Weisfeiler--Lehman graph kernel features.

import igraph as ig
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import argparse
import collections
import logging
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from distances import jensen_shannon
from distances import kullback_leibler

from features import FeatureSelector
from features import PersistentWeisfeilerLehman

from utilities import read_labels
from utilities import to_probability_distribution


def make_kernel_matrices(persistence_diagrams, l, L):
    '''
    Converts a set of persistence diagrams into a set of kernel
    matrices, both for the KL and the JS divergence.

    :param persistence_diagrams: List of persistence diagrams
    :param l: Label lookup (maps graph indices to a set of vertex labels)
    :param L: Maximum number of labels

    :return: Probability distribution matrix, KL matrix, JS matrix
    '''

    # Number of graphs/objects/what-have-you...
    n = len(persistence_diagrams)

    # Will store *all* persistence diagrams in the form of a probability
    # distribution.
    M = np.zeros((n, L))

    for index, pd in enumerate(persistence_diagrams):
        P = to_probability_distribution(pd, l[index], L)
        M[index, :] = P

    D_KL = np.zeros((n, n))
    D_JS = np.zeros((n, n))

    # TODO: rewrite this into a broadcasted version of the same loop
    # because it is probably a lot faster.
    for i in range(len(persistence_diagrams)):
        p = M[i, :]
        for j in range(i, len(persistence_diagrams)):
            q = M[j, :]

            D_KL[i, j] = kullback_leibler(p, q)
            D_KL[j, i] = D_KL[i, j]
            D_JS[i, j] = jensen_shannon(p, q)
            D_JS[j, i] = D_JS[i, j]

    return M, D_KL, D_JS


def main(args, logger):

    graphs = [ig.read(filename) for filename in args.FILES]
    labels = read_labels(args.labels)

    # Stores *all* vertex labels of the given graph in order to
    # determine the conversion factor for persistence diagrams.
    vertex_labels = set()

    # Set the label to be uniform over all graphs in case no labels are
    # available. This essentially changes our iteration to degree-based
    # checks.
    for graph in graphs:
        if 'label' not in graph.vs.attributes():
            graph.vs['label'] = [0] * len(graph.vs)

        vertex_labels.update(graph.vs['label'])

    logger.info('Read {} graphs and {} labels'.format(len(graphs), len(labels)))

    assert len(graphs) == len(labels)

    pwl = PersistentWeisfeilerLehman(
            use_cycle_persistence=args.use_cycle_persistence,
            use_original_features=args.use_original_features,
            use_label_persistence=True,
            store_persistence_diagrams=True,
    )

    if args.use_cycle_persistence:
        logger.info('Using cycle persistence')

    y = LabelEncoder().fit_transform(labels)
    X, num_columns_per_iteration = pwl.transform(graphs, args.num_iterations)

    persistence_diagrams = pwl._persistence_diagrams

    fig, ax = plt.subplots(args.num_iterations + 1)

    for iteration in persistence_diagrams.keys():
        M = collections.defaultdict(list)

        for index, pd in enumerate(persistence_diagrams[iteration]):
            label = y[index]
            for _, d, _ in pd:
                M[label].append(d)

        d_min = sys.float_info.max
        d_max = -d_min

        for hist in M.values():
            d_min = min(d_min, min(hist))
            d_max = max(d_max, max(hist))

        bins = np.linspace(d_min, d_max, 10)

        for label, hist in M.items():
            sns.distplot(hist,
                bins=bins,
                rug=True,
                kde=True,
                hist=False,
                ax=ax[iteration])

    plt.show()

    L = len(vertex_labels)
    assert L > 0

    original_labels = pwl._original_labels

    # Will store *all* persistence diagrams in the form of a probability
    # distribution.
    M = np.zeros((len(graphs), (args.num_iterations + 1) * L))

    # Will store *all* pairwise distances according to the
    # Jensen--Shannon divergence (JS),  or, alternatively,
    # the Kullback--Leibler divergence (KL).
    D_KL = np.zeros((len(graphs), len(graphs)))
    D_JS = np.zeros((len(graphs), len(graphs)))

    D = np.zeros((len(graphs), len(graphs)))

    for iteration in persistence_diagrams.keys():

        M, D_KL, D_JS = make_kernel_matrices(
            persistence_diagrams[iteration],
            original_labels,  # notice that they do *not* change
            L
        )

        D += D_JS

    D = -D

    fig, ax = plt.subplots(len(set(y)))

    for label in sorted(set(y)):
        ax[label].matshow(M[y == label], aspect='auto', vmin=0, vmax=1)

    plt.show()

    logger.info('Finished persistent Weisfeiler-Lehman transformation')
    logger.info('Obtained ({} x {}) feature matrix'.format(X.shape[0], X.shape[1]))

    np.random.seed(42)
    cv = StratifiedKFold(n_splits=10, shuffle=True)
    mean_accuracies = []

    for i in range(10):

        # Contains accuracy scores for each cross validation step; the
        # means of this list will be used later on.
        accuracy_scores = []

        for train_index, test_index in cv.split(X, y):
            rf_clf = RandomForestClassifier(
                n_estimators=50,
                class_weight='balanced' if args.balanced else None
            )

            if args.grid_search:
                pipeline = Pipeline(
                    [
                        ('fs', FeatureSelector(num_columns_per_iteration)),
                        ('clf', rf_clf)
                    ],
                )

                grid_params = {
                    'fs__num_iterations': np.arange(0, args.num_iterations + 1),
                    'clf__n_estimators': [10, 20, 50, 100, 150, 200],
                }

                clf = GridSearchCV(
                        pipeline,
                        grid_params,
                        cv=StratifiedKFold(n_splits=10, shuffle=True),
                        iid=False,
                        scoring='accuracy',
                        n_jobs=4)
            else:
                clf = rf_clf

            clf = SVC(kernel='precomputed')
            clf.fit(D, y)
            y_test = y
            y_pred = clf.predict(D)

            #X_train, X_test = X[train_index], X[test_index]
            #y_train, y_test = y[train_index], y[test_index]

            ## TODO: need to discuss whether this is 'allowed' or smart
            ## to do; this assumes normality of the attributes.
            #scaler = StandardScaler()
            #X_train = scaler.fit_transform(X_train)
            #X_test = scaler.transform(X_test)

            #scaler = MinMaxScaler()
            #X_train = scaler.fit_transform(X_train)
            #X_test = scaler.transform(X_test)

            #clf.fit(X_train, y_train)
            #y_pred = clf.predict(X_test)

            accuracy_scores.append(accuracy_score(y_test, y_pred))

            logger.debug('Best classifier for this fold: {}'.format(clf))

            if args.grid_search:
                logger.debug('Best parameters for this fold: {}'.format(clf.best_params_))
            else:
                logger.debug('Best parameters for this fold: {}'.format(clf.get_params()))

        mean_accuracies.append(np.mean(accuracy_scores))
        logger.info('  - Mean 10-fold accuracy: {:2.2f} [running mean over all folds: {:2.2f}]'.format(mean_accuracies[-1] * 100, np.mean(mean_accuracies) * 100))

    logger.info('Accuracy: {:2.2f} +- {:2.2f}'.format(np.mean(mean_accuracies) * 100, np.std(mean_accuracies) * 100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('FILES', nargs='+', help='Input graphs (in some supported format)')
    parser.add_argument('-b', '--balanced', action='store_true', help='Make random forest classifier balanced')
    parser.add_argument('-d', '--dataset', help='Name of data set')
    parser.add_argument('-l', '--labels', type=str, help='Labels file', required=True)
    parser.add_argument('-n', '--num-iterations', default=3, type=int, help='Number of Weisfeiler-Lehman iterations')
    parser.add_argument('-f', '--filtration', type=str, default='sublevel', help='Filtration type')
    parser.add_argument('-g', '--grid-search', action='store_true', default=False, help='Whether to do hyperparameter grid search')
    parser.add_argument('-c', '--use-cycle-persistence', action='store_true', default=False, help='Indicates whether cycle persistence should be calculated or not')
    parser.add_argument('-o', '--use-original-features', action='store_true', default=False, help='Indicates that original features should be used as well')

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
