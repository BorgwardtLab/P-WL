#!/usr/bin/env python3
#
# show_metric.py: depicts how the intrinsic graph metric changes as
# a function of the number of Weisfeiler--Lehman hops.


import igraph as ig

import argparse


from features import WeisfeilerLehman
from features import WeightAssigner


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('FILE', help='Input graph')
    parser.add_argument('-n', '--num-iterations', default=3, type=int, help='Number of Weisfeiler-Lehman iterations')

    args = parser.parse_args()
    graphs = ig.read(args.FILE)

    wl = WeisfeilerLehman()
    wa = WeightAssigner(metric='minkowski', p=2.0)  # TODO: make configurable

    label_dicts = wl.fit_transform([graph], args.num_iterations)
    graph_index = 0

    for iteration in sorted(label_dicts.keys()):
        weighted_graph = graph.copy()

        labels_raw, labels_compressed = label_dicts[iteration][graph_index]

        weighted_graphs[graph_index].vs['label'] = labels_raw
        weighted_graphs[graph_index].vs['compressed_label'] = labels_compressed

        weighted_graphs[graph_index] = wa.fit_transform(weighted_graphs[graph_index])



