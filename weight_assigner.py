import collections
import igraph as ig


class WeightAssigner:
    '''
    Given a labelled graph, this class assigns weights based on
    a similarity measure and an assignment strategy and returns
    the weighted graph.
    '''

    def __init__(self, ignore_label_order=False, similarity='hamming'):
        self._ignore_label_order = ignore_label_order
        self._similarity = similarity

    def fit_transform(self, graph):
        pass

    def _hamming(A, B):
        '''
        Computes the (normalized) Hamming distance between two sets of
        labels A and B. This amounts to counting how many overlaps are
        present in the sequences.
        '''

        # Normalization factor so that the result always depends on the
        # length of the larger string.
        n = max(len(A), len(B))

        counter = collections.Counter(A)
        counter.subtract(B)

        num_missing = sum([abs(c) for _, c in counter.most_common()])
        return num_missing / n

    def _jaccard(A, B):
        '''
        Computes the Jaccard index between two sets of labels A and B. The
        measure is in the range of [0,1], where 0 indicates no overlap and
        1 indicates a perfect overlap.
        '''

        A = set(A)
        B = set(B)

        return len(A.intersection(B)) / len(A.union(B))


# FIXME: remove after debug
if __name__ == '__main__':
    graph = ig.read('data/MUTAG/000.gml')
    print(graph)
