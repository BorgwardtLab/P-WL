#!/usr/bin/env python3
#
# label_sequence_distance.py: calculates a Weisfeiler--Lehman label
# sequence and uses it to assign distances between the vertices. As
# a result, a distance matrix will be stored.

import igraph as ig

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

    print(label_dicts)
