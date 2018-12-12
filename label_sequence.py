#!/usr/bin/env python3
#
# label_sequence_distance.py: calculates a Weisfeiler--Lehman label
# sequence and uses it to assign distances between the vertices. As
# a result, a distance matrix will be stored.

import igraph as ig
import numpy as np

import argparse


from features import WeisfeilerLehman

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('FILES', nargs='+', help='Input graphs (in some supported format)')
    parser.add_argument('-n', '--num-iterations', default=3, type=int, help='Number of Weisfeiler-Lehman iterations')

    args = parser.parse_args()
    graphs = [ig.read(filename) for filename in args.FILES]

    wl = WeisfeilerLehman()
    label_dicts = wl.fit_transform(graphs, args.num_iterations)

    # Each entry in the list represents the label sequence of a single
    # graph. The label sequence contains the vertices in its rows, and
    # the individual iterations in its columns.
    #
    # Hence, (i, j) will contain the label of vertex i at iteration j.
    label_sequences = [
        np.zeros((len(graph.vs), args.num_iterations)) for graph in graphs
    ]

    for iteration in sorted(label_dicts.keys()):
        for graph_index, graph in enumerate(graphs):
            labels_raw, labels_compressed = label_dicts[iteration][graph_index]
