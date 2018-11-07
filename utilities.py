'''
Contains auxiliary and utility functions that are shared among different
scripts.
'''


def read_labels(filename):
    '''
    Reads labels from a file. Labels are supposed to be stored in each
    line of the file. No further pre-processing will be performed.
    '''

    labels = []
    with open(filename) as f:
        labels = f.readlines()
        labels = [label.strip() for label in labels]

    return labels

