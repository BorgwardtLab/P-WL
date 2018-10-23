'''
topology.py: contains classes that represent topological information
about a data set.
'''

import collections.abc


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
