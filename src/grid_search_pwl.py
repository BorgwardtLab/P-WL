#!/usr/bin/env python3

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
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection._validation import _fit_and_score
from sklearn.svm import SVC
from sklearn.base import clone

from tqdm import tqdm

from features import FeatureSelector
from features import PersistentWeisfeilerLehman
from features import WeisfeilerLehmanSubtree

from utilities import read_labels

def custom_grid_search_cv(pipeline, pipeline_grid_params, pwl_list, y, cv=5):
    cv = StratifiedKFold(n_splits=cv, shuffle=True)
    results = []
    for train_index, val_index in cv.split(pwl_list[0]['X_train'], y):
        split_results = []
        params = []
        
        # Run over feature matrices
        for i, pwl in enumerate(pwl_list):
            X = pwl['X_train']
            pwl_idx = i
            for p in ParameterGrid(pipeline_grid_params):
                sc = _fit_and_score(clone(pipeline), X, y, \
                                    scorer=make_scorer(accuracy_score), \
                                    train=train_index, test=val_index, \
                                    parameters=p, fit_params=None, verbose=0)
                split_results.append(sc)
                params.append({'pwl_idx': pwl_idx, 'params': p})
        
        results.append(split_results)
    # Collect results and average
    results = np.array(results)
    fin_results = results.mean(axis=0)
    # select the best results
    best_idx = np.argmax(fin_results)
    # return the best fitted model
    
    ret_model = clone(pipeline).set_params(**params[best_idx]['params'])
    return ret_model.fit(pwl_list[params[best_idx]['pwl_idx']]['X_train'], y, ), params[best_idx]
    
        

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

    pwl_list = []
    for p in [1,2]:
        pwl = PersistentWeisfeilerLehman(
                use_cycle_persistence=args.use_cycle_persistence,
                use_original_features=args.use_original_features,
                metric=args.metric,
                use_label_persistence=True,
                p=p)

        X, num_columns_per_iteration = pwl.transform(graphs, args.num_iterations)
        pwl_list.append({'p': p, 'X': X})
        
        logger.info(f'Finished persistent Weisfeiler-Lehman transformation for \
                    p={p}')
        logger.info('Obtained ({} x {}) feature matrix'.format(X.shape[0], X.shape[1]))


    if args.use_cycle_persistence:
        logger.info('Using cycle persistence')

    y = LabelEncoder().fit_transform(labels)

    np.random.seed(42)
    mean_accuracies = []

    params = ['balanced', 'num_iterations', 'filtration', 'use_cycle_persistence', 'use_original_features', 'metric'] 
    cv_results = []
    entry = {}
    for param in params:
        entry[param] = args.__dict__[param]
    entry['dataset'] = dirname(args.FILES[0]).split('/')[1]
    for i in range(10):
        # Contains accuracy scores for each cross validation step; the
        # means of this list will be used later on.
        accuracy_scores = []
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=i)
        for n, indices in enumerate(cv.split(X, y)):
            entry_fold = copy.copy(entry)
            train_index = indices[0]
            test_index = indices[1]
            y_train = y[train_index]
            y_test = y[test_index]

            # Override current full matrices
            for pwl_dict in pwl_list:

                scaler = StandardScaler()
                X_train = scaler.fit_transform(pwl_dict['X'][train_index])
                X_test = scaler.transform(pwl_dict['X'][test_index])

                scaler = MinMaxScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                pwl_dict['X_train'] = X_train
                pwl_dict['X_test'] = X_test

            pipeline = Pipeline(
                [
                    ('fs', FeatureSelector(num_columns_per_iteration)),
                    ('clf', RandomForestClassifier(class_weight='balanced' if
                                                   args.balanced else None,
                                                   random_state=42, n_jobs=4))
                ],
            )

            grid_params = {
                'fs__num_iterations': np.arange(0, args.num_iterations + 1),
                'clf__n_estimators': [25, 50, 100],
            }
            
            clf, best_params = custom_grid_search_cv(pipeline, grid_params, pwl_list, y_train)


            X_test = pwl_list[best_params['pwl_idx']]['X_test']
            y_pred = clf.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            accuracy_scores.append(acc)

            best_params['params']['p'] = best_params['pwl_idx'] + 1
            for param, param_val in best_params['params'].items():
                entry_fold[param] = param_val
                entry[param] = '' 
            entry_fold['fold'] = n + 1
            entry_fold['it'] = i
            entry_fold['acc'] = acc * 100
            entry_fold['std'] = 0.0
            cv_results.append(entry_fold)

            logger.info('Best classifier for this fold:{}'.format(best_params['params']))


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
    parser.add_argument('-n', '--num-iterations', default=3, type=int, help='Number of Weisfeiler-Lehman iterations')
    parser.add_argument('-f', '--filtration', type=str, default='sublevel', help='Filtration type')
    parser.add_argument('-c', '--use-cycle-persistence', action='store_true', default=False, help='Indicates whether cycle persistence should be calculated or not')
    parser.add_argument('-o', '--use-original-features', action='store_true', default=False, help='Indicates that original features should be used as well')
    # TODO: this flag is somewhat redundant given the flag above; need
    # to ensure that it is seen as an 'override', i.e. if this is set,
    # *no* other ways of calculating features can be used.
    parser.add_argument('-m', '--metric', type=str, default='minkowski', help='Metric to use for graph weight assignment')
    parser.add_argument('-r', '--result-file',
                        default='../grid_search_results/results_PWL.csv', help='File in which to store results')

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
