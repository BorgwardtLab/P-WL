#!/usr/bin/env python3
#
# persistence_distributions.py: Persistent Weisfeiler--Lehman graph
# kernel based on comparing persistence distributions by divergence
# measures such as the Kullback--Leibler divergence.


import igraph as ig
import numpy as np

import argparse
import collections
import itertools
import logging


from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from distances import jensen_shannon
from distances import kullback_leibler

from features import PersistentWeisfeilerLehman

from utilities import read_labels


def to_probability_distribution(X, num_columns_per_iteration):
    '''
    Given a matrix and an index array containing the number of columns
    per iteration, converts each row to a probability distribution, by
    normalizing its sum accordingly.
    '''

    start_index = 0
    for iteration in sorted(num_columns_per_iteration.keys()):
        end_index = num_columns_per_iteration[iteration]

        row_sums = np.sum(X[:, start_index:end_index], axis=1)

        X[:, start_index:end_index] /= row_sums[:, None]

        start_index += end_index

    np.nan_to_num(X, copy=False)
    return X


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
        use_label_persistence=True,
        store_persistence_diagrams=False,  # TODO: might need this later on?
    )

    y = LabelEncoder().fit_transform(labels)
    X, num_columns_per_iteration = pwl.transform(graphs, args.num_iterations)

    logger.info('Finished persistent Weisfeiler-Lehman transformation')
    logger.info('Obtained ({} x {}) feature matrix'.format(X.shape[0], X.shape[1]))

    X = to_probability_distribution(X, num_columns_per_iteration)

    np.random.seed(42)
    cv = StratifiedKFold(n_splits=3, shuffle=True)
    mean_accuracies = []

    def product_kernel(X, Y, k):

        # Index for the current iteration; indicates which column is
        # used as the *first* one.
        start_index = 0
        K = np.zeros((X.shape[0], Y.shape[0]))

        for iteration in sorted(num_columns_per_iteration.keys()):
            end_index = num_columns_per_iteration[iteration]

            P = X[:, start_index:end_index]
            Q = Y[:, start_index:end_index]

            # TODO: can this be made more efficient?
            K_iteration = np.array(
                [k(p, q) for p, q in itertools.product(P, Q)]).reshape(
                X.shape[0], Y.shape[0]
            )

            K += K_iteration

            start_index += end_index

        return K

    def jensen_shannon_kernel(X, Y):
        return product_kernel(X, Y, jensen_shannon)

    def kullback_leibler_kernel(X, Y):
        return -product_kernel(X, Y, kullback_leibler)

    for i in range(3):

        # Contains accuracy scores for each cross validation step; the
        # means of this list will be used later on.
        accuracy_scores = []

        for train_index, test_index in cv.split(X, y):
            clf = SVC(
                kernel=jensen_shannon_kernel,
            )

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # TODO: need to discuss whether this is 'allowed' or smart
            # to do; this assumes normality of the attributes.
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            accuracy_scores.append(accuracy_score(y_test, y_pred))

            logger.debug('Best classifier for this fold: {}'.format(clf))
            logger.debug('Best parameters for this fold: {}'.format(clf.get_params()))

        mean_accuracies.append(np.mean(accuracy_scores))
        logger.info('  - Mean 10-fold accuracy: {:2.2f} [running mean over all folds: {:2.2f}]'.format(mean_accuracies[-1] * 100, np.mean(mean_accuracies) * 100))

    logger.info('Accuracy: {:2.2f} +- {:2.2f}'.format(np.mean(mean_accuracies) * 100, np.std(mean_accuracies) * 100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('FILES', nargs='+', help='Input graphs (in some supported format)')
    parser.add_argument('-l', '--labels', type=str, help='Labels file', required=True)
    parser.add_argument('-n', '--num-iterations', default=3, type=int, help='Number of Weisfeiler-Lehman iterations')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
    )

    logger = logging.getLogger('P-WL [distribution]')

    main(args, logger)
