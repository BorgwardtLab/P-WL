#!/usr/bin/env python3
#
# main.py: main script for testing Persistent Weisfeiler--Lehman graph
# kernels.


import igraph as ig

import argparse
import logging

from weight_assigner import WeightAssigner  # FIXME: put this in a different module
from WL import WL


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('FILES', nargs='+', help='Input graphs (in some supported format)')
    parser.add_argument('-n', '--num-iterations', default=3, type=int, help='Number of Weisfeiler-Lehman iterations')
    parser.add_argument('-f', '--filtration', type=str, default='sublevel', help='Filtration type')

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('P-WL')

    args = parser.parse_args()
    graphs = [ig.read(filename) for filename in args.FILES]

    for graph in graphs:
            print(graph.diameter())

    logger.debug('Read {} graphs'.format(len(graphs)))

    wl = WL()
    wa = WeightAssigner()

    for graph in graphs:
        wl.fit_transform(graph, args.num_iterations)

        # Stores the new multi-labels that occur in every iteration,
        # plus the original labels of the zeroth iteration.
        iteration_to_label = wl._multisets
        iteration_to_label[0] = wl._graphs[0].vs['label']

        for iteration in sorted(iteration_to_label.keys()):
            graph.vs['label'] = iteration_to_label[iteration]
            graph = wa.fit_transform(graph)
            for edge in graph.es:
                u, v = edge.tuple
                print(graph.vs[u]['label'], graph.vs[v]['label'], edge['weight'])
            print('')
