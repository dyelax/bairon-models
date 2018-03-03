import numpy as np
import argparse
import os
from glob import glob

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help='The directory of the data.', required=True)

    args = parser.parse_args()

    npy_paths = glob(os.path.join(args.dir, '*.npy'))

    wordcount = 0
    for path in npy_paths:
        arr = np.load(path)
        poem = ' \n '.join(arr)
        words = poem.split(' ')

        wordcount += len(words)

    print wordcount