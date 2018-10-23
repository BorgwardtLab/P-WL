import igraph as ig


class WeightAssigner:
    '''
    Given a labelled graph, this class assigns weights based on
    a similarity measure and an assignment strategy and returns
    the weighted graph.
    '''

    def __init__(self, ignore_label_order=False, similarity='jaccard'):
        self._ignore_label_order = ignore_label_order
        self._similarity = similarity

    def fit_transform(self, graph):
        pass

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
    import sys
