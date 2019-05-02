#!/usr/bin/env python3

import argparse
import collections

import igraph as ig
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


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
    
    args = parser.parse_args()

    graphs = [ig.read(filename) for filename in args.FILES]
    labels = read_labels(args.labels)

    cycles_per_label = collections.defaultdict(list)

    for graph, label in zip(graphs, labels):
        num_connected_components = len(graph.components().subgraphs())
        num_vertices = len(graph.vs)
        num_edges = len(graph.es)
        num_cycles = num_edges - num_vertices + num_connected_components 

        cycles_per_label[label].append(num_cycles)

    for key, cycle_distribution in cycles_per_label.items():
        sns.distplot(cycle_distribution)

    plt.show()
