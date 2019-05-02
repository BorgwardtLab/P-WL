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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from tqdm import tqdm

from features import FeatureSelector
from features import PersistentWeisfeilerLehman
from features import WeisfeilerLehmanSubtree

from utilities import read_labels


def main(args, logger):

    # Read all graphs and labels; there is no direct way of checking
    # that the labels are 'correct' for the graphs, but at least the
    # code will check that they have the same cardinality.
    graphs = [ig.read(filename) for filename in args.FILES]
    labels = read_labels(args.labels)

    # Set the label to be uniform over all graphs in case no labels are
    # available. This essentially changes our iteration to degree-based
    # checks.
    for graph in graphs:
        if 'label' not in graph.vs.attributes():
            graph.vs['label'] = [0] * len(graph.vs)

    logger.info(
        'Read {} graphs and {} labels'.format(len(graphs), len(labels))
    )

    assert len(graphs) == len(labels)

    # Replace selected metric if necessary; this only applies to the
    # uniform metric shortcut.
    if args.use_uniform_metric:
        args.metric = 'uniform'

    pwl = PersistentWeisfeilerLehman(
            use_cycle_persistence=args.use_cycle_persistence,
            use_original_features=args.use_original_features,
            use_label_persistence=True,
            metric=args.metric,
            p=args.power
    )

    if args.use_cycle_persistence:
        logger.info('Using cycle persistence')

    # Ensures that labels are encoded correctly, regardless of whether
    # they are numerical or not.
    y = LabelEncoder().fit_transform(labels)

    # This ignores *all* other feature generation methods and falls back
    # to the original Weisfeiler--Lehman subtree kernel.
    if args.use_subtree_features:

        logger.info('Using original subtree features')

        wl_subtree = WeisfeilerLehmanSubtree()
        X, num_columns_per_iteration = \
            wl_subtree.transform(graphs, args.num_iterations)
    else:
        X, num_columns_per_iteration = \
            pwl.transform(graphs, args.num_iterations)

    logger.info('Finished persistent Weisfeiler-Lehman transformation')
    logger.info(
        'Obtained ({} x {}) feature matrix'.format(X.shape[0], X.shape[1])
    )

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
                class_weight='balanced' if args.balanced else None,
                random_state=42
            )

            if args.grid_search:
                pipeline = Pipeline(
                    [
                        ('fs', FeatureSelector(num_columns_per_iteration)),
                        ('clf', rf_clf)
                    ],
                )

                grid_params = {
                    'fs__num_iterations':
                        np.arange(0, args.num_iterations + 1),

                    'clf__n_estimators':
                        [10, 20, 50, 100, 150, 200],
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

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

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

            if args.grid_search:
                logger.debug(
                    'Best parameters for this fold: {}'.format(
                        clf.best_params_)
                    )
            else:
                logger.debug(
                    'Best parameters for this fold: {}'.format(
                        clf.get_params())
                )

        mean_accuracies.append(np.mean(accuracy_scores))
        logger.info(
            '''  - Mean 10-fold accuracy: {:2.2f} [running mean over all
            folds: {:2.2f}]'''.format(
                mean_accuracies[-1] * 100, np.mean(mean_accuracies) * 100)
        )

    logger.info(
        'Accuracy: {:2.2f} +- {:2.2f}'.format(
            np.mean(mean_accuracies) * 100, np.std(mean_accuracies) * 100)
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'FILES', nargs='+', help='Input graphs (in some supported format)'
    )

    # Controls behaviour of the classifier (will be treated as
    # a hyperparameter in the respective grid search scripts)
    parser.add_argument(
        '-b', '--balanced',
        action='store_true', help='Make random forest classifier balanced'
    )

    parser.add_argument('-d', '--dataset', help='Name of data set')
    parser.add_argument(
        '-l', '--labels', type=str, help='Labels file', required=True
    )

    parser.add_argument(
        '-n', '--num-iterations', default=3,
        type=int, help='Number of Weisfeiler-Lehman iterations'
    )

    parser.add_argument(
        '-g', '--grid-search', action='store_true',
        default=False, help='Whether to do hyperparameter grid search'
    )

    parser.add_argument(
        '-c', '--use-cycle-persistence', action='store_true', default=False,
        help='Indicates whether cycle persistence should be calculated or not'
    )

    parser.add_argument(
        '-o', '--use-original-features', action='store_true', default=False,
        help='Indicates that original features should be used as well'
    )

    # TODO: this flag is somewhat redundant given the flag above; need
    # to ensure that it is seen as an 'override', i.e. if this is set,
    # *no* other ways of calculating features can be used.
    parser.add_argument(
        '-s', '--use-subtree-features', action='store_true', default=False,
        help='''
            Use Weisfeiler--Lehman subtree kernel instead of topological
            features
        '''
    )

    ####################################################################
    # Metric selection options
    ####################################################################

    parser.add_argument(
        '-m', '--metric',
        type=str, default='minkowski',
        help='Metric to use for graph weight assignment'
    )

    parser.add_argument(
        '-u', '--use-uniform-metric',
        action='store_true', default=False,
        help='Use uniform metric for weight assignment'
    )

    parser.add_argument(
        '-p', '--power',
        type=float, default=2.0,
        help='Power parameter for Minkowski metric calculations'
    )

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
