#!/usr/bin/env python

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def read_results(path, drop_summary=True):
    df = pd.read_csv(path, sep='\t', index_col=0, na_filter=False)
    if drop_summary:
        df = df.drop(labels=['adapted', 'all', 'high', 'unseen'])
    return df

def plot_results(results, metric, names, out=None):
    """
    make a scatter plot of the results, I reckon
    """
    fig, axes = plt.subplots(ncols=len(results), sharey=True)
    for axis, df, name in zip(axes, results, names):
        x = df['train_count']
        y = df[metric]
        axis.scatter(x, y)
        axis.set_title(name)
    # plt.ylabel(metric)
    fig.text(0.5, 0.04, 'Training Size', ha='center')
    fig.text(0.04, 0.5, ' '.join(metric.upper().split('_')), va='center', rotation='vertical')
    if out is not None:
        plt.savefig(out, bbox_inches='tight')
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('results', nargs='+', type=read_results,
                        help="results tsv files created by evaluate.py.")
    parser.add_argument('-min_test_size', type=int, default=200)
    parser.add_argument('-metric', choices=['per', 'wer_1', 'wer_100'], default='per')
    parser.add_argument('-names', nargs='+', help='Subplot names')
    parser.add_argument('-out')
    opt = parser.parse_args()
    # results = opt.results.loc[opt.results['test_count'] >= opt.min_test_size]
    results = [r.loc[r['test_count'] >= opt.min_test_size] for r in opt.results]
    plot_results(results, opt.metric, opt.names, opt.out)

if __name__ == '__main__':
    main()
