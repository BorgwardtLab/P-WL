'''
topology.py: contains classes that represent topological information
about a data set.
'''

import collections.abc

import igraph as ig
import numpy as np


class PersistenceDiagram(collections.abc.Sequence):
    '''
    Represents a persistence diagram, i.e. a pairing of nodes in
    a graph. The purpose of this class is to provide a *simpler*
    interface for storing and accessing this pairing.
    '''

    def __init__(self):
        self._pairs = []

    def __len__(self):
        '''
        Returns the number of pairs in the persistence diagram.
        '''

        return len(self._pairs)

    def __getitem__(self, index):
        '''
        Returns the persistence pair at the given index.
        '''

        return self._pairs[index]

    def add(self, x, y):
        '''
        Adds a new persistence pair to the given diagram. Performs no other
        validity checks.
        '''

        self._pairs.append((x, y))

    def total_persistence(self, p=1):
        '''
        Calculates the total persistence of the current pairing.
        '''

        return sum([abs(x - y)**p for x, y in self._pairs])**(1.0 / p)


class UnionFind:
    '''
    An implementation of a Union--Find class. The class performs path
    compression by default. It uses integers for storing one disjoint
    set, assuming that vertices are zero-indexed.
    '''

    def __init__(self, num_vertices):
        '''
        Initializes an empty Union--Find data structure for a given
        number of vertices.
        '''

        self.parent = [x for x in range(num_vertices)]

    def find(self, u):
        '''
        Finds and returns the parent of u with respect to the hierarchy.
        '''

        if self.parent[u] == u:
            return u
        else:
            # Perform path collapse operation
            self.parent[u] = self.find(self.parent[u])
            return self.parent[u]

    def merge(self, u, v):
        '''
        Merges vertex u into the component of vertex v. Note the
        asymmetry of this operation.
        '''

        if u != v:
            self.parent[self.find(u)] = self.find(v)


class PersistenceDiagramCalculator:
    '''
    Given a weighted graph, calculates a persistence diagram. The client
    can modify the filtration order and the vertex weight assignment.
    '''

    def __init__(self, order='sublevel', fix_vertices=True):
        self._order = order
        self._fix_vertices = fix_vertices

    def fit_transform(self, graph):
        '''
        Applies a filtration to a graph and calculates its persistence
        diagram.
        '''

        print(graph.es['weight'])


# FIXME: hard-coded debug code
if __name__ == '__main__':
    graph = ig.read('data/MUTAG/000.gml')
    PersistenceDiagramCalculator().fit_transform(graph)
