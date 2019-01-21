#!/usr/bin/env python3
#
# p-wl_unlabelled_persistence_diagrams.py: script for calculating
# persistent Weisfeiler--Lehman features for non-attributed graph
# data sets.
#
# This script will *not* perform any fitting.


import igraph as ig
import numpy as np

import argparse
import collections
import logging
import itertools
import os

from features import WeisfeilerLehmanAttributePropagation

from kernels import PersistenceScaleSpaceKernel

from utilities import read_labels

from topology import assign_filtration_values
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

    use_vertex_weights = args.vertex_weights

    # Stores *all* persistence diagrams because they will be used to
    # represent the data set later on.
    persistence_diagrams_per_iteration = collections.defaultdict(list)

    for iteration in sorted(attributes_per_iteration.keys()):

        # Determine maximum attribute value over *all* graphs and their
        # respective filtrations.
        max_attribute = max([np.max(attributes_per_iteration[iteration][index]) for index, _ in enumerate(graphs)])

        unpaired_value = 2 * max_attribute

        if use_vertex_weights:
            pdc = PersistenceDiagramCalculator(
                unpaired_value=unpaired_value,
                vertex_attribute='degree',
            )
        else:
            pdc = PersistenceDiagramCalculator(
                unpaired_value=unpaired_value
            )

        for index, graph in enumerate(graphs):
            attributes = attributes_per_iteration[iteration][index]

            graph.vs['degree'] = attributes

            weighted_graph = assign_filtration_values(
                graph,
                attributes,
                normalize=args.normalize
            )

            pd, edge_indices_cycles = pdc.fit_transform(graph)

            # Store the persistence diagram as a 2D array in order to
            # facilitate the subsequent kernel calculations.
            persistence_diagrams_per_iteration[iteration].append(
                np.array([(c, d) for c, d, _ in pd])
            )

            np.savetxt(
                '/tmp/{:04d}_d0_h{:d}.txt'.format(index, iteration),
                np.array([(c, d) for c, d, _ in pd]),
                fmt='%.f'
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('FILES', nargs='+', help='Input graphs (in some supported format)')
    parser.add_argument('-l', '--labels', type=str, help='Labels file', required=True)
    parser.add_argument('-H', '--num-iterations', default=3, type=int, help='Number of Weisfeiler-Lehman iterations')
    parser.add_argument('-f', '--filtration', type=str, default='sublevel', help='Filtration type')
    parser.add_argument('-s', '--sigma', type=float, default='1.0', help='Smoothing parameter')
    parser.add_argument('-n', '--normalize', action='store_true', default=False, help='Use degree normalization')
    parser.add_argument('-o', '--out-dir', type=str, default='.', help='Output directory')
    parser.add_argument('-v', '--vertex-weights', action='store_true', default=False, help='Use vertex weights in diagrams')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG,
    )

    logger = logging.getLogger('P-WL [unlabelled, persistence diagrams]')
    main(args, logger)
