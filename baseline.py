#!/usr/bin/env python3
#
# baseline.py: script for evaluating the baseline Weisfeiler--Lehman
# graph kernel on an input data set.


import graphkernels as gk
import igraph as ig

import argparse
import logging


# TODO: shamelessly copied from `main.py`; should become a separate
# function somewhere
def read_labels(filename):
    labels = []
    with open(filename) as f:
        labels = f.readlines()
        labels = [label.strip() for label in labels]

    return labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('FILES', nargs='+', help='Input graphs (in some supported format)')
    parser.add_argument('-l', '--labels', type=str, help='Labels file', required=True)
    parser.add_argument('-n', '--num-iterations', default=3, type=int, help='Number of Weisfeiler-Lehman iterations')

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('Baseline')

    graphs = [ig.read(filename) for filename in args.FILES]
    labels = read_labels(args.labels)

    assert len(graphs) == len(labels)

    K = gk.CalculateWLKernel(graphs, args.num_iterations)
