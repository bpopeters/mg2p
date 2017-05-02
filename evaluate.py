#!/usr/bin/env python3

import argparse
import sys
from os.path import join, basename
from itertools import groupby
from glob import glob

def levenshtein(a, b):
    """
    computes the levenshtein distance between strings a and b
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
    
def per(predicted, gold):
    """
    predicted: sequence of predicted phoneme sequences
    gold: gold standard phoneme sequences
    returns: phoneme error rate of predicted words
    """
    assert  len(predicted) == len(gold), 'Mismatched data'
    return sum(levenshtein(p, g) for p, g in zip(predicted, gold)) / sum(len(g) for g in gold)
    
def wer(predicted, gold):
    assert  len(predicted) == len(gold), 'Mismatched data'
    # this is the single-guess version.
    return sum(p != g for p, g in zip(predicted, gold)) / len(gold)
    
def read_words(corpus):
    with open(corpus, encoding='utf-8') as f:
        return [line.strip().split() for line in f]
        
def get_lang_labels(src_test):
    """
    src_test: path to the model's src.test file.
    returns: a sequence listing the language name at each line of the file
    """
    with open(src_test, encoding='utf-8') as f:
        return [line.split(None, 1)[0][1:4] for line in f]
        
def evaluate(predicted, gold, lang_labels):
    """
    predicted, gold: equal length sequences of phoneme sequences
    lang_labels
    returns: PER and WER for the data overall and 
    """
    result = dict()
    result['overall'] = {'WER': wer(predicted, gold), 'PER': per(predicted, gold)}
    if len(set(lang_labels)) > 1:
        # sorting the groups by their language label: otherwise there could
        # be multiple groups for each language
        lang_groups = groupby(sorted(zip(lang_labels, predicted, gold)), key=lambda x: x[0])
        for lang, triples in lang_groups:
            triples = list(triples)
            
            p = [t[1] for t in triples]
            g = [t[2] for t in triples]
            
            result[lang] = {'WER': wer(p, g), 'PER': per(p, g)}
    return result
    
def make_sequences(model_path):
    predicted = read_words(join(model_path, 'predicted.txt'))
    gold = read_words(join(model_path, 'corpus', 'tgt.test'))
    labels = get_lang_labels(join(model_path, 'corpus', 'src.test'))
    return predicted, gold, labels

if __name__ == '__main__':
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('predicted', type=read_words)
    parser.add_argument('gold', type=read_words)
    parser.add_argument('lang_labels', type=get_lang_labels)
    args = parser.parse_args()
    
    
    for model in sorted(glob('models/kat-full-spa*')):
        print('Evaluating {}'.format(basename(model)))
        print(evaluate(*make_sequences(model)))
    '''
    results = evaluate(*make_sequences(sys.argv[1]))
    for language in sorted(results, key=lambda lang: results[lang]['PER']):
        print(language, results[language])
