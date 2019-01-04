'''
Contains auxiliary and utility functions that are shared among different
scripts.
'''


import numpy as np


def read_labels(filename):
    '''
    Reads labels from a file. Labels are supposed to be stored in each
    line of the file. No further pre-processing will be performed.
    '''

    labels = []
    with open(filename) as f:
        labels = f.readlines()
        labels = [label.strip() for label in labels]

    return labels


def to_probability_distribution(persistence_diagram, l, L):
    '''
    Converts a persistence diagram with labels to a (discrete)
    probability distribution.

    :param persistence_diagram: Persistence diagram
    :param l: Label lookup data structure for individual vertices
    :param L: Maximum number of labels of discrete distribution

    :return: Discrete probability distribution
    '''

    P = np.zeros(L)

    for x, y, v in persistence_diagram:

        label = l[v]

        # Just to make sure that this mapping can work
        assert label < L
        assert label >= 0

        # TODO: make power configurable?
        P[label] += (y - x)**2

    # Ensures that this distribution is valid, i.e. normalized to sum to
    # one; else, we are implicitly comparing distributions whose size or
    # weight varies.
    P = P / np.sum(P)
    return P
