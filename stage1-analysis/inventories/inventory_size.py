#!/usr/bin/env python

import argparse
from itertools import chain
import pandas as pd

def read_train(path):
    with open(path) as f:
        return [line.strip().split() for line in f]

def read_langs(path):
    with open(path) as f:
        return [line.strip() for line in f]

def inventory_size(sequences):
    return len(set(chain.from_iterable(sequences)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_data', type=read_train)
    parser.add_argument('langs', type=read_langs)
    parser.add_argument('out')
    opt = parser.parse_args()
    data = pd.Series(data=opt.train_data, index=opt.langs)
    inv = data.groupby(level=0).apply(inventory_size)
    inv.to_csv(opt.out)

if __name__ == '__main__':
    main()
