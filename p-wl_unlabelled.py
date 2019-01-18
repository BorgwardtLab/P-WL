#!/usr/bin/env python3
#
# p-wl_unlabelled.py: script for calculating persistent Weisfeiler--Lehman
# features for graphs that have no attributes and no (node|edge) labels.


import igraph as ig

import argparse
import logging


from features import WeisfeilerLehmanAttributePropagation

from utilities import read_labels

from topology import assign_filtration_values
from topology import multiscale_persistence_diagram_kernel
from topology import PersistenceDiagramCalculator


def main(args, logger):

    graphs = [ig.read(filename) for filename in args.FILES]
    labels = read_labels(args.labels)

    for graph in graphs:

        # Make sure that no label information exists as a graph
        # attribute already.
        assert 'label' not in graph.vs.attributes()

        graph.vs['degree'] = graph.vs.degree()

    logger.info('Read {} graphs and {} labels'.format(len(graphs), len(labels)))

    assert len(graphs) == len(labels)

    prop = WeisfeilerLehmanAttributePropagation()
    attributes_per_iteration = prop.transform(
        graphs,
        'degree',
        args.num_iterations
    )

    pdc = PersistenceDiagramCalculator(vertex_attribute='degree')

    for iteration in sorted(attributes_per_iteration.keys()):
        for index, graph in enumerate(graphs):
            attributes = attributes_per_iteration[iteration][index]
            graph.vs['degree'] = attributes
            weighted_graph = assign_filtration_values(graph, attributes)

            pd, edge_indices_cycles = pdc.fit_transform(graph)
            multiscale_persistence_diagram_kernel(pd, pd, sigma=0.1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('FILES', nargs='+', help='Input graphs (in some supported format)')
    parser.add_argument('-b', '--balanced', action='store_true', help='Make random forest classifier balanced')
    parser.add_argument('-d', '--dataset', help='Name of data set')
    parser.add_argument('-l', '--labels', type=str, help='Labels file', required=True)
    parser.add_argument('-n', '--num-iterations', default=3, type=int, help='Number of Weisfeiler-Lehman iterations')
    parser.add_argument('-f', '--filtration', type=str, default='sublevel', help='Filtration type')
    parser.add_argument('-g', '--grid-search', action='store_true', default=False, help='Whether to do hyperparameter grid search')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG,
        filename='{}_{:02d}.log'.format(args.dataset, args.num_iterations)
    )

    logger = logging.getLogger('P-WL [unlabelled]')

    # Create a second stream handler for logging to `stderr`, but set
    # its log level to be a little bit smaller such that we only have
    # informative messages
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    # Use the default format; since we do not adjust the logger before,
    # this is all right.
    stream_handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
    logger.addHandler(stream_handler)

    main(args, logger)
