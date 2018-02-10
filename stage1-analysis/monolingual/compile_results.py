#!/usr/bin/env python

import argparse
import pandas as pd

def read_results(path):
    df = pd.read_csv(path, sep='\t', index_col=0)
    df = df.drop('all')
    df = df.reset_index().rename(columns={'index':'lang'})
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results', nargs='+', type=read_results)
    parser.add_argument('-out', default='mono_results.csv')
    opt = parser.parse_args()
    data = pd.concat(opt.results)
    data.to_csv(opt.out, index=False)

if __name__ == '__main__':
    main()
