#!/usr/bin/env python

import numpy as np
from collections import Counter
from itertools import chain

def replacements(gold_word, predicted_word):
    return [(g_char, p_char) for g_char, p_char in zip(predicted_word, gold_word) if p_char != g_char]
    

@np.vectorize
def levenshtein(a, b):
    """
    computes the levenshtein distance between sequences a and b.
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
    return levenshtein(results['predicted'], results['gold']).sum() / results['gold'].apply(len).sum()

def wer(results):
    """
    results: DataFrame containing at minimum columns for predicted and
            gold standard phoneme sequences
    returns: word error rate of the predictions
    """
    return (results['predicted'] != results['gold']).sum() / results['predicted'].size
    
def substitution_errors(results):
    """
    results: DataFrame containing columns for predicted and gold standard phonemes
    returns: a dictionary of Counters. er...ah
    """
    same_length = results.loc[results['gold'].apply(len) == results['predicted'].apply(len)]
    subs = Counter(chain(*[replacements(gw, pw) for gw, pw in zip(same_length['gold'], same_length['predicted'])])).most_common(10)
    return subs
