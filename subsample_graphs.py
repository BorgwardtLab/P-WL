#!/usr/bin/env python3
#
# subsample_graphs.py: Given a larger input data set, performs
# stratified subsampling and copies new files into an *output*
# directory.


import argparse
import os
import shutil

import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder


from utilities import read_labels


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('FILES', nargs='+', help='Input graphs (in some supported format)')
    parser.add_argument('-l', '--labels', type=str, help='Labels file', required=True)
    parser.add_argument('-n', '--num-graphs', type=int, required=True, help='Sample size')
    parser.add_argument('-o', '--out-dir', type=str, required=True, help='Output directory')

    args = parser.parse_args()
    labels = read_labels(args.labels)
    y = LabelEncoder().fit_transform(labels)
    n = len(y)

    sss = StratifiedShuffleSplit(
            n_splits=1,
            random_state=23,
            train_size=args.num_graphs
    )

    for train_index, _ in sss.split(range(n), y):
        train_index = sorted(train_index)

        files = np.array(args.FILES)
        files = files[train_index]

        try:
            os.makedirs(args.out_dir)
        except FileExistsError:
            pass

        for filename in files:
            source = filename
            target = os.path.join(
                args.out_dir,
                os.path.basename(source)
            )

            shutil.copy2(source, target)

        labels = y[train_index]
        np.savetxt(
            os.path.join(
                args.out_dir,
                'Labels.txt'
            ), labels, fmt='%d'
        )
