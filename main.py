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
from sklearn.metrics import accuracy_score

from tqdm import tqdm

from topology import PersistenceDiagramCalculator
from weight_assigner import WeightAssigner  # FIXME: put this in a different module
from WL import WL

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
    parser.add_argument('-f', '--filtration', type=str, default='sublevel', help='Filtration type')

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('P-WL')

    args = parser.parse_args()
    graphs = [ig.read(filename) for filename in args.FILES]
    labels = read_labels(args.labels)

    logger.debug('Read {} graphs and {} labels'.format(len(graphs), len(labels)))

    assert len(graphs) == len(labels)

    wl = WL()
    wa = WeightAssigner()
    pdc = PersistenceDiagramCalculator()  # FIXME: need to add order/filtration

    X = np.zeros((len(graphs), args.num_iterations + 1))
    y = np.array(labels)

    for index, (graph, label) in tqdm(enumerate(zip(graphs, labels))):
        wl.fit_transform(graph, args.num_iterations)

        # Stores the new multi-labels that occur in every iteration,
        # plus the original labels of the zeroth iteration.
        iteration_to_label = wl._multisets
        iteration_to_label[0] = wl._graphs[0].vs['label']

        total_persistence_values = []

        for iteration in sorted(iteration_to_label.keys()):
            graph.vs['label'] = iteration_to_label[iteration]
            graph = wa.fit_transform(graph)
            persistence_diagram = pdc.fit_transform(graph)
            X[index, iteration] = persistence_diagram.total_persistence()

    #import matplotlib
    #matplotlib.use('TkAgg')
    #import matplotlib.pyplot as plt

    #for label in set(labels):
    #    plt.matshow(X[y == label], aspect='auto')
    #plt.show()


    clf = LogisticRegression(solver='lbfgs')
    clf.fit(X, y)
    y_pred = clf.predict(X)

    print('Train accuracy: {:.2f}'.format(accuracy_score(y, y_pred)))
