#!/usr/bin/env python

"""
Functions for measuring and visualizing the performance of g2p models.
"""

from os.path import join
import pandas as pd
import numpy as np

@np.vectorize
def levenshtein(a, b):
    """
    computes the levenshtein distance between sequences a and b.
    New: it's broadcastable!
    """
    d = [[0 for i in range(len(b) + 1)] for j in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        d[i][0] = i
    for j in range(1, len(b) + 1):
        d[0][j] = j
    for j in range(1, len(b) + 1):
        for i in range(1, len(a) + 1):
            # future: abstraction. cost depends on phoneme similarity
            cost = int(a[i - 1] != b[j - 1])
            d[i][j] = min(d[i][j - 1] + 1, d[i - 1][j] + 1, d[i - 1][j - 1] + cost)
    return d[len(a)][len(b)]

def per(results):
    """
    results: DataFrame containing at minimum columns for predicted and
            gold standard phoneme sequences
    returns: phoneme error rate of the predictions
    """
    # this is one of two possible ways to compute PER
    return levenshtein(results['predicted'], results['gold']).sum() / results['gold'].apply(len).sum()

def wer(results):
    """
    results: DataFrame containing at minimum columns for predicted and
            gold standard phoneme sequences
    returns: word error rate of the predictions
    """
    return (results['predicted'] != results['gold']).sum() / results['predicted'].size
        
def language_labels(source_file):
    with open(source_file) as f:
        return pd.Series([line.split(None, 1)[0] for line in f])
        
def read_words(corpus_file):
    with open(corpus_file) as f:
        return pd.Series([line.strip().split() for line in f])
    
def results_table(model_path):
    """
    model_path: model directory, containing the corpus subdirectory and the
                results on src.test
    returns: DataFrame whose rows are (language, gold phonemes, predicted phonemes)
    """
    source_test = join(model_path, 'corpus', 'src.test')
    target_test = join(model_path, 'corpus', 'tgt.test')
    predicted_test = join(model_path, 'predicted.txt')
    
    lang_id = language_labels(source_test)
    gold_words = read_words(target_test)
    predicted_words = read_words(predicted_test)
    
    return pd.DataFrame.from_items([('lang', lang_id), ('gold', gold_words), ('predicted', predicted_words)])
    
def plot_trainsize_vs_wer():
    pass

if __name__ == '__main__':
    model_path = '/home/bpop/thesis/mg2p/models/latin-script-brnn-64batch'
    
    df = results_table(model_path)
    print(df.groupby('lang').apply(per, wer))
