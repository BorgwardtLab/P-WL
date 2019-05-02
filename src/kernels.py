'''
Contains classes for calculating kernels between persistence diagrams.
'''

import math

import numpy as np


from sklearn.metrics import pairwise_distances


class PersistenceScaleSpaceKernel:
    def __init__(self, sigma):
        self._sigma = sigma

    def mirror_along_diagonal(self, D):
        '''
        Mirrors a given persistence diagram along the diagonal.
        '''

        return np.matmul(D, np.array([[0.0, 1.0], [1.0, 0.0]]))

    def fit_transform(self, F, G):
        G_m = self.mirror_along_diagonal(G)

        X = F
        Y = np.concatenate([G, G_m], 0)
        c = 8 * self._sigma

        distances = np.exp(-pairwise_distances(X, Y, metric='sqeuclidean') / c)

        # Negative weights for the 'negative' part of the mirrored
        # persistence points in the other diagram.
        weights = np.ones((len(X), len(Y)))
        weights[:, len(G):] = -1

        return np.sum(np.multiply(weights, distances)) / (c * math.pi)

