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


if __name__ == '__main__':
    df = pd.read_csv(sys.argv[1], index_col=0)

    for row in df.itertuples(index=True):
        data_set_name = None

        l = []
        x = []
        y = []
        e = []

        for index, (name, value) in enumerate(row._asdict().items()):
            if name == 'Index':
                data_set_name = value
            else:
                accuracy, sdev = parse_accuracy(value)

                if not np.isnan(accuracy):
                    l.append(name)
                    x.append(index)
                    y.append(accuracy)
                    e.append(sdev)

        if x:
            plt.title(data_set_name)
            plt.errorbar(x, y, e, fmt='o')
            plt.xticks(np.arange(1, len(x) + 1), l)
            plt.show()
