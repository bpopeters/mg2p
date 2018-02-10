#!/usr/bin/env python

import argparse
import re
from collections import defaultdict
from itertools import count
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.dummy import DummyClassifier


def read_embeddings(emb_file):
    df = pd.read_csv(emb_file, sep=' ', header=None, converters={0: lambda x: re.sub(r'[<>]', '', x)}, index_col=0)
    df.index.names = ['name']
    return df


def usable_feature(df, label, min_count):
    col_counts = df[label].dropna().value_counts()
    return col_counts.size == 2 and col_counts.min() >= min_count


def read_uriel(uriel_file, categories=['P_', 'INV_']):
    df = pd.read_csv(uriel_file, sep=',', index_col='G_CODE', na_values='--')
    ph_columns = [c for c in df.columns if any(c.startswith(s) for s in categories)]
    df = df[ph_columns]
    df = df.mask(lambda x: x >= 0.5, 1)
    df = df.mask(lambda x: x < 0.5, 0)

    # df = df[[c for c in df.columns if usable_feature(df, c, min_values)]]
    return df


def read_phoible(phoible_file):
    df = pd.read_csv(
        phoible_file, sep='\t', na_values='0'
    )
    df = df.drop_duplicates(subset='segment')
    df = df.set_index('segment')
    # df = df.loc[df.isin(['+', '-', '0']).all(axis=1)]
    # df = df.drop(labels=['short', 'loweredLarynxImplosive'], axis=1)
    df = df.mask(lambda x: ~x.isin(['+', '-']))
    return df


def phoible_filter(series):
    # return (series.notnull()) & (series != '0')
    return series.isin(['+', '-'])


def uriel_filter(series):
    return series.isin([0, 1])


def classify(clf, data, label, folds, scoring, good_sample):
    """
    clf: classifier instance
    data: frame containing input features and output labels
    label: single label to build classifier for
    folds: for cross-validation
    returns: a Series containing results cross-validated classifier results
    """
    raw_y = data.loc[:, ('y', label)]
    y_filter = good_sample(raw_y)
    X = data.loc[y_filter, 'x']
    y = raw_y.loc[y_filter]

    scores = cross_validate(
        clf, X, y, cv=folds,
        scoring=scoring,
        return_train_score=False
    )
    result = pd.Series(
        data={k: v.mean() for k, v in scores.items() if 'test' in k}
    )
    result['feature'] = label
    result['samples'] = y.size
    return result


def usable(series, min_values):
    return (series.loc[series.notnull()].value_counts() >= min_values).all()


def main():
    phoible_path = '/home/bpop/thesis/mg2p/data/phoible-segments-features.tsv'
    # uriel_path = '/home/bpop/thesis/mg2p/data/uriel_v0_3_0/features/predicted.csv'  # achtung!
    uriel_path = '/home/bpop/thesis/mg2p/data/uriel_v0_3_0/features/avg.csv'
    parser = argparse.ArgumentParser()
    parser.add_argument('embeddings', type=read_embeddings)
    parser.add_argument('-knowledge', default='phoible', choices=['phoible', 'uriel'])
    parser.add_argument('-folds', type=int, default=5)
    parser.add_argument('-min_embeddings', type=int, default=100,
                        help="""Drop features if they aren't defined for at
                        least this many languages.""")
    parser.add_argument('-min_features', type=int, default=100,
                        help="""Drop languages if they don't have at
                        least this many features defined.""")
    parser.add_argument('-solver', default='liblinear')
    parser.add_argument('-max_iter', default=200, type=int)
    parser.add_argument('-metrics', nargs='+', default=['accuracy'])
    parser.add_argument('-out')
    opt = parser.parse_args()

    knowledge = read_phoible(phoible_path) if opt.knowledge == 'phoible' else read_uriel(uriel_path)

    # concatenate embeddings to knowledge
    data = pd.concat(
        [opt.embeddings, knowledge],
        join='inner',
        axis=1,
        keys=['x', 'y']
    )
    #print(data.loc[:, 'y'].apply(lambda c: c.dropna().value_counts().size).min())
    
    #data.loc[:, 'y'] = data.loc[:, ('y', [c for c in data.loc[:, 'y'].columns if usable_feature(data, c, opt.folds)])]
    #print(data.loc[:, 'y'].apply(lambda c: c.dropna().value_counts().size).min())
    
    # subsets of features for phoible
    if opt.knowledge == 'phoible':
        class_features = ['syllabic', 'consonantal', 'sonorant', 'approximant']
        manner_features = ['continuant', 'nasal', 'lateral', 'delayedRelease', 'periodicGlottalSource']
        # something 
        place_features = ['labial', 'coronal', 'anterior', 'distributed', 'strident', 'dorsal', 'high', 'low']
        features = class_features + manner_features + place_features
    else:
        features = knowledge.columns

    # if this part is relevant, it is relevant only to the language prediction
    # data = data.loc[:, data.notnull().sum() >= opt.min_embeddings]
    baseline_clf = DummyClassifier(strategy='most_frequent')
    bin_logit_clf = LogisticRegression()
    value_filter = phoible_filter if opt.knowledge == 'phoible' else uriel_filter

    baseline = pd.DataFrame(
        data = [classify(baseline_clf, data, label, opt.folds, opt.metrics, value_filter)
                for label in features if usable_feature(data.loc[:, 'y'], label, opt.folds)]
    ).set_index('feature')

    logit_results = pd.DataFrame(
        data = [classify(bin_logit_clf, data, label, opt.folds, opt.metrics, value_filter)
                for label in features if usable_feature(data.loc[:, 'y'], label, opt.folds)]
    ).set_index('feature')
    #print(logit_results)
    logit_results = logit_results.drop('samples', axis=1)
    # results = pd.concat([logit_results, baseline], axis=1, keys=['Logistic Regression', 'Baseline'])
    results = logit_results.join(baseline, lsuffix='_lr', rsuffix='_baseline')
    '''
    results.columns = results.columns.swaplevel(0, 1)
    results = results.sortlevel(0, axis=1)
    results = results.drop('samples', axis=1)
    '''
    # results['test_accuracy'] *= 100
    #results.to_csv()
    #print(results.to_latex(float_format='%.2f'))
    if opt.out is not None:
        results.to_csv(opt.out, sep='\t')
    else:
        print(results)

if __name__ == '__main__':
    main()
