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

    def append(self, x, y):
        '''
        Appends a new persistence pair to the given diagram. Performs no
        other validity checks.
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

        self._parent = [x for x in range(num_vertices)]

    def find(self, u):
        '''
        Finds and returns the parent of u with respect to the hierarchy.
        '''

        if self._parent[u] == u:
            return u
        else:
            # Perform path collapse operation
            self._parent[u] = self.find(self._parent[u])
            return self._parent[u]

    def merge(self, u, v):
        '''
        Merges vertex u into the component of vertex v. Note the
        asymmetry of this operation.
        '''

        if u != v:
            self._parent[self.find(u)] = self.find(v)

    def roots(self):
        '''
        Generator expression for returning roots, i.e. components that
        are their own parents.
        '''

        for vertex, parent in enumerate(self._parent):
            if vertex == parent:
                yield vertex


class PersistenceDiagramCalculator:
    '''
    Given a weighted graph, calculates a persistence diagram. The client
    can modify the filtration order and the vertex weight assignment.
    '''

    def __init__(self, order='sublevel', fix_vertices=True, unpaired_value=None):
        self._order = order
        self._fix_vertices = fix_vertices
        self._unpaired_value = unpaired_value

        if self._order not in ['sublevel', 'superlevel']:
            raise RuntimeError('Unknown filtration order \"{}\"'.format(self._order))

    def fit_transform(self, graph):
        '''
        Applies a filtration to a graph and calculates its persistence
        diagram.
        '''

        num_vertices = graph.vcount()
        uf = UnionFind(num_vertices)

        edge_weights = graph.es['weight']
        edge_indices = None

        if self._order == 'sublevel':
            edge_indices = np.argsort(edge_weights, kind='stable')
        elif self._order == 'superlevel':
            edge_indices = np.argsort(-edge_weights, kind='stable')

        assert edge_indices is not None

        # Will be filled during the iteration below. This will become
        # the return value of the function.
        pd = PersistenceDiagram()

        # Go over all edges and optionally create new points for the
        # persistence diagram.
        for edge_index, edge_weight in zip(edge_indices, edge_weights):
            u, v = graph.es[edge_index].tuple

            younger_component = uf.find(u)
            older_component = uf.find(v)

            # Nothing to do here: the two components are already the
            # same
            if younger_component == older_component:
                continue

            # Ensures that the older component precedes the younger one
            # in terms of its vertex index
            elif younger_component > older_component:
                u, v = v, u

            # TODO: this does not yet take into account any weights for
            # the vertices themselves.
            creation = 0.0              # x coordinate for persistence diagram
            destruction = edge_weight   # y coordinate for persistence diagram

            uf.merge(u, v)
            pd.append(creation, destruction)

        # By default, use the largest (sublevel set) or lowest
        # (superlevel set) weight, unless the user specified a
        # different one.
        unpaired_value = edge_weights[edge_indices[-1]]
        if self._unpaired_value:
            unpaired_value = self._unpaired_value

        # Add tuples for every root component in the Union--Find data
        # structure. This ensures that multiple connected components
        # are handled correctly.
        for root in uf.roots():

            # TODO: again, I am ignoring the weight of vertices and
            # forcing them to start at zero.
            creation = 0.0
            destruction = unpaired_value

            pd.append(creation, destruction)

        return pd


# FIXME: hard-coded debug code
if __name__ == '__main__':
    graph = ig.read('data/MUTAG/000.gml')
    pd = PersistenceDiagramCalculator().fit_transform(graph)

    print(graph.ecount())
    print(len(pd))

    for x, y in pd:
        print(x, y)
