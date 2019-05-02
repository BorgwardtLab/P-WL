#!/usr/bin/env python3
#
# main.py: main script for testing Persistent Weisfeiler--Lehman graph
# kernels.

import copy
import igraph as ig
import numpy as np
import pandas as pd

import argparse
import collections
import logging
from os.path import dirname, exists

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
import graphkernels.kernels as gk

from tqdm import tqdm

from kernelgridsearchcv import KernelGridSearchCV

from features import FeatureSelector
from features import PersistentWeisfeilerLehman
from features import WeisfeilerLehmanSubtree

from utilities import read_labels
from sklearn.base import clone

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

    # Calculate graph kernel
    gram_matrix = gk.CalculateVertexHistKernel(graphs)

    y = LabelEncoder().fit_transform(labels)
    np.random.seed(42)
    
    mean_accuracies = []

    params = ['balanced'] 
    cv_results = []
    entry = {}
    for param in params:
        entry[param] = args.__dict__[param]
    entry['dataset'] = dirname(args.FILES[0]).split('/')[1]
    entry['baseline'] = 'vertex hist kernel'
    for i in range(10):
        # Contains accuracy scores for each cross validation step; the
        # means of this list will be used later on.
        accuracy_scores = []
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=i)
        for n, indices in enumerate(cv.split(graphs, y)):

            entry_fold = copy.copy(entry)
            train_index = indices[0]
            test_index = indices[1]

            pipeline = Pipeline(
                [
                    ('clf', SVC(class_weight='balanced' if
                                                   args.balanced else None,
                                random_state=42, kernel='precomputed'))
                ],
            )

            grid_params = {
                'clf__C': [1e1]
            }

            X_train, X_test = gram_matrix[train_index][:,train_index], gram_matrix[test_index][:,train_index]
            y_train, y_test = y[train_index], y[test_index]

            kgscv = KernelGridSearchCV(pipeline,
                                       param_grid=grid_params, cv=cv,
                                      random_state=42)
            kgscv.fit(X_train, y_train)
            p = kgscv._best_params
            sc = kgscv._best_score
            clf = kgscv._best_estimator
            clf.fit(X_train, y_train)

            y_pred = clf.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            accuracy_scores.append(acc)

            for param, param_val in kgscv._best_params.items():
                entry_fold[param] = param_val
                entry[param] = '' 
            entry_fold['fold'] = n + 1
            entry_fold['it'] = i
            entry_fold['acc'] = acc * 100
            entry_fold['std'] = 0.0
            cv_results.append(entry_fold)

            logger.info('Best classifier for this fold:{}'.format(kgscv._best_params))


        mean_accuracies.append(np.mean(accuracy_scores))
        logger.info('  - Mean 10-fold accuracy: {:2.2f} [running mean over all folds: {:2.2f}]'.format(mean_accuracies[-1] * 100, np.mean(mean_accuracies) * 100))
    entry['fold'] = 'all'
    entry['it'] = 'all'
    entry['acc'] = np.mean(mean_accuracies) * 100
    entry['std'] = np.std(mean_accuracies) * 100
    cv_results.append(entry)
    logger.info('Accuracy: {:2.2f} +- {:2.2f}'.format(np.mean(mean_accuracies) * 100, np.std(mean_accuracies) * 100))

    if exists(args.result_file):
        with open(args.result_file, 'a') as f:
            pd.DataFrame(cv_results).to_csv(f, index=False, header=None)
    else:
        pd.DataFrame(cv_results).to_csv(args.result_file, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('FILES', nargs='+', help='Input graphs (in some supported format)')
    parser.add_argument('-b', '--balanced', action='store_true', help='Make random forest classifier balanced')
    parser.add_argument('-d', '--dataset', help='Name of data set')
    parser.add_argument('-l', '--labels', type=str, help='Labels file', required=True)
    # TODO: this flag is somewhat redundant given the flag above; need
    # to ensure that it is seen as an 'override', i.e. if this is set,
    # *no* other ways of calculating features can be used.
    parser.add_argument('-r', '--result-file',
                        default='grid_search_results/results_baselines.csv', help='File in which to store results')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG,
        filename='{}.log'.format(args.dataset)
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
