'''
Distance measures (not necessarily metrics) between probability
distributions.
'''


import numpy as np


def kullback_leibler(p, q):
    '''
    Calculates the Kullback--Leibler divergence between two discrete
    probability distributions.

    :param p: First discrete probability distribution
    :param q: Second discrete probability distribution

    :return: Value of Kullback--Leibler divergence
    '''

    # TODO: this should not be necessary
    p = np.abs(p)
    q = np.abs(q)

    p += 1e-8
    q += 1e-8

    return np.sum(p * np.log(q / p))

    # FIXME: why does this not work? Even if the two conditions are
    # chained together...
    #return np.sum(np.where(p != 0, p * np.log(q / p), 0))


def jensen_shannon(p, q):
    '''
    Calculates the Jensen--Shannon divergence between two discrete
    probability distributions.

    :param p: First discrete probability distribution
    :param q: Second discrete probability distribution

    :return: Value of Jensen--Shannon divergence
    '''

    return 0.5 * (kullback_leibler(p, q) + kullback_leibler(q, p))
