#!/usr/bin/env python3
#
# main.py: main script for testing Persistent Weisfeiler--Lehman graph
# kernels.


import igraph as ig
import numpy as np

import argparse
import collections
import logging

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV

from tqdm import tqdm

from features import PersistentWeisfeilerLehman

def read_labels(filename):
    labels = []
    with open(filename) as f:
        labels = f.readlines()
        labels = [label.strip() for label in labels]

    return labels

def main(args, logger):

    graphs = [ig.read(filename) for filename in args.FILES]
    labels = read_labels(args.labels)

    logger.debug('Read {} graphs and {} labels'.format(len(graphs), len(labels)))

    assert len(graphs) == len(labels)

    y = np.array(labels)
    X = PersistentWeisfeilerLehman(num_iterations=args.num_iterations).fit_transform(graphs, y)

    np.random.seed(42)
    cv = StratifiedKFold(n_splits=10, shuffle=True)
    mean_accuracies = []

    grid_params = { 'n_estimators': [10, 20, 50, 100, 200], 'criterion': ['gini', 'entropy'], 'max_depth': [None, 5, 10, 25] }

    for i in range(10):

        # Contains accuracy scores for each cross validation step; the
        # means of this list will be used later on.
        accuracy_scores = []

        for train_index, test_index in cv.split(X, y):
            rf_clf = RandomForestClassifier(n_estimators=50)
            
            if args.grid_search:
                clf = GridSearchCV(rf_clf, grid_params, cv=5, scoring='accuracy', n_jobs=4)
            else:
                clf = rf_clf

            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            accuracy_scores.append(accuracy_score(y_test, y_pred))

        print('  - Mean 10-fold accuracy: {:2.2f}'.format(np.mean(accuracy_scores)))
        mean_accuracies.append(np.mean(accuracy_scores))

    print('Accuracy: {:2.2f} +- {:2.2f}'.format(np.mean(mean_accuracies) * 100, np.std(mean_accuracies) * 100))
    print(clf)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('FILES', nargs='+', help='Input graphs (in some supported format)')
    parser.add_argument('-l', '--labels', type=str, help='Labels file', required=True)
    parser.add_argument('-n', '--num-iterations', default=3, type=int, help='Number of Weisfeiler-Lehman iterations')
    parser.add_argument('-f', '--filtration', type=str, default='sublevel', help='Filtration type')
    parser.add_argument('-g', '--grid-search', type=bool, default=False, help='Whether to do hyperparameter grid search')

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('P-WL')

    args = parser.parse_args()
    main(args, logger)
