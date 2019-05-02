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
from os.path import dirname, exists, basename

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

from kernelgridsearchcv import KernelGridSearchCV

from tqdm import tqdm

from features import FeatureSelector
from features import PersistentWeisfeilerLehman
from features import WeisfeilerLehmanSubtree
import copy

from utilities import read_labels

def custom_grid_search_cv(pipeline, pipeline_grid_params, matrix_dict, y, cv=5):
    bsf_sc = 0
    bsv_clf = None
    bsf_params = {}

    K = np.zeros(shape=matrix_dict[0]['X_train'].shape)
    summed_matrices = {}
    
    for i in range(max(list(matrix_dict.keys()))+1):
        K += matrix_dict[i]['X_train']
        summed_matrices[i] = copy.deepcopy(K)

    # Run over feature matrices
    for h in matrix_dict.keys():
        # Generate kernel matrix
        kgscv = KernelGridSearchCV(clone(pipeline),
                                   param_grid=pipeline_grid_params, cv=cv,
                                  random_state=42)
        kgscv.fit(summed_matrices[h], y)
        p = kgscv._best_params
        sc = kgscv._best_score
        clf = kgscv._best_estimator

        if sc > bsf_sc:
            bsf_sc = sc
            bsf_clf = clf
            bsf_params = {'params': p, 'h': h}
    
    ret_model = bsf_clf.set_params(**bsf_params['params'])
    return ret_model.fit(summed_matrices[bsf_params['h']] ,y), bsf_params
    
        

def main(args, logger):

    labels = read_labels(args.labels)
    # Load matrices
    matrices = np.load(args.MATRICES)

    print(f"Loaded {len(list(matrices.keys()))} matrices, with shape {matrices['0'].shape}")

    matrix_dict = {}
    for h in matrices.keys():
        matrix_dict[int(h)] = {'gram': matrices[h]}

    y = LabelEncoder().fit_transform(labels)

    np.random.seed(42)
    mean_accuracies = []

    kernel_params = np.array(basename(args.MATRICES)[:-4].split('_'))[1:]

    params = ['dataset', 'max_h', 'sigma']
    cv_results = []
    entry = {}
    for i, param in enumerate(params):
        entry[param] = kernel_params[i]
    
    for i in range(10):
        # Contains accuracy scores for each cross validation step; the
        # means of this list will be used later on.
        accuracy_scores = []
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=i)
        for n, indices in enumerate(cv.split(matrix_dict[0]['gram'], y)):
            entry_fold = copy.copy(entry)

            train_index = indices[0]
            test_index = indices[1]
            y_train = y[train_index]
            y_test = y[test_index]

            # Override current full matrices
            for h, m_dict in matrix_dict.items():

                X_train = m_dict['gram'][train_index][:,train_index]
                X_test = m_dict['gram'][test_index][:,train_index]

                m_dict['X_train'] = X_train
                m_dict['X_test'] = X_test

            pipeline = Pipeline(
                [
                    ('clf', SVC(class_weight='balanced' if args.balanced else
                                None, random_state=42,
                                kernel='precomputed'))
                ],
            )

            grid_params = {
                'clf__C': [1e-1, 1e0, 1e1],
            }
            
            clf, best_params = custom_grid_search_cv(pipeline, grid_params,
                                                     matrix_dict, y_train)
            
            X_test = np.zeros(shape=matrix_dict[0]['X_test'].shape)
            counter = 0
            for h in range(best_params['h']+1):
                X_test += matrix_dict[h]['X_test']
                counter += 1

            y_pred = clf.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            accuracy_scores.append(acc)

            best_params['params']['h'] = best_params['h']
            for param, param_val in best_params['params'].items():
                entry_fold[param] = param_val
                entry[param] = '' 
            entry_fold['fold'] = n + 1
            entry_fold['it'] = i
            entry_fold['acc'] = acc * 100
            entry_fold['std'] = 0.0
            cv_results.append(entry_fold)

            print('Best classifier for this fold:{}'.format(best_params['params']))


        mean_accuracies.append(np.mean(accuracy_scores))
        print('  - Mean 10-fold accuracy: {:2.2f} [running mean over all folds: {:2.2f}]'.format(mean_accuracies[-1] * 100, np.mean(mean_accuracies) * 100))
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
    parser.add_argument('MATRICES', type=str, help='Matrices files')
    parser.add_argument('-b', '--balanced', action='store_true', help='Make random forest classifier balanced')
    parser.add_argument('-l', '--labels', type=str, help='Labels file', required=True)
    
    # TODO: this flag is somewhat redundant given the flag above; need
    # to ensure that it is seen as an 'override', i.e. if this is set,
    # *no* other ways of calculating features can be used.
    parser.add_argument('-r', '--result-file',
                        default='grid_search_results/results_pss_kernel.csv', help='File in which to store results')

    args = parser.parse_args()
    logger = logging.getLogger('PSS-Kernel')

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
