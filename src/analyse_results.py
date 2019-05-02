#!/usr/bin/env python3
#
# analyse_results.py: analyses reported results from this paper and from
# other papers by checking for significant overlaps.


import numpy as np
import pandas as pd

import re
import sys

import matplotlib.pyplot as plt


def parse_accuracy(entry):
    r_token = r'([0-9]+\.[0-9]+)\s+\([±]?([0-9]+\.[0-9]+)\)'

    m = re.match(r_token, str(entry))
    if m:
        return float(m.group(1)), float(m.group(2))
    else:
        return np.nan, np.nan


def overlaps(accuracy1, sdev1, accuracy2, sdev2):
    a = accuracy1 - sdev1
    b = accuracy1 + sdev2
    c = accuracy2 - sdev2
    d = accuracy2 + sdev2

    return b >= c and a <= d


if __name__ == '__main__':
    df = pd.read_csv(sys.argv[1], index_col=0)

    for row in df.itertuples(index=True):
        data_set_name = None

        L = []  # labels
        x = []  # $x$ positions
        y = []  # $y$ positions
        e = []  # error bars
        intervals = []  # intervals

        for index, (name, value) in enumerate(row._asdict().items()):
            if name == 'Index':
                data_set_name = value
            else:
                accuracy, sdev = parse_accuracy(value)

                if not np.isnan(accuracy):
                    L.append(name)
                    x.append(len(x))
                    y.append(accuracy)
                    e.append(sdev)
                    intervals.append((accuracy, sdev))

        if x:
            if False:
                plt.title(data_set_name)
                plt.errorbar(x, y, e, fmt='o')
                plt.xticks(np.arange(len(x)), L)
                plt.show()

            # Create pairwise overlap matrix of all methods

            n = len(x)
            M = np.zeros((n, n))

            for i, (a, b) in enumerate(intervals):
                for j, (c, d) in enumerate(intervals):
                    M[i, j] = overlaps(a, b, c, d)

                # To make sure that there will be some variation in the
                # matrix.
                M[i, i] = False

            plt.matshow(M, cmap='binary')
            plt.title(data_set_name)
            plt.xticks(np.arange(len(x)), L)
            plt.yticks(np.arange(len(x)), L)
            plt.show()
