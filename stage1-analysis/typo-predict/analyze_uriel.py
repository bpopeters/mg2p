#!/usr/bin/env python

import argparse
from collections import Counter
import pandas as pd

def read_uriel(path):
    df = pd.read_csv(path, na_values='--', index_col='G_CODE')
    df = df[[c for c in df.columns if c.startswith('P_') or c.startswith('INV_')]]
    return df

def read_index(path):
    with open(path) as f:
        return Counter(line.strip() for line in f)

def get_baselines(uriel, languages, threshold):
    usable_langs = [l for l, c in languages.items() if c >= threshold]
    return uriel.loc[usable_langs].sum() / len(usable_langs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-uriel', type=read_uriel, default='/home/bpop/thesis/mg2p/data/uriel_v0_3_0/features/predicted.csv')
    parser.add_argument('-lang_counts', type=read_index, default='/home/bpop/thesis/mg2p/models/stage1/langf/langf-all/corpus/lang_index.train')
    opt = parser.parse_args()
    print(get_baselines(opt.uriel, opt.lang_counts, 9000).sort_values().median())

if __name__ == '__main__':
    main()
