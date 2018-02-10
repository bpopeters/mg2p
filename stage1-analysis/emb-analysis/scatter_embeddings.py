#!/usr/bin/env python

import argparse
from itertools import groupby
import re
import unicodedata
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from count_training_data import count_phonemes
import pandas as pd

def read_phoible(phoible_file):
    df = pd.read_csv(phoible_file, sep='\t', na_filter=False)
    df = df.drop_duplicates(subset='segment')
    return df.set_index('segment')

def read_embeddings(emb_file):
    characters = []
    vectors = []
    with open(emb_file) as f:
        for line in f:
            split_line = line.split()
            characters.append(split_line[0])
            vectors.append([float(x) for x in split_line[1:]])
    return characters, np.array(vectors)

def segment_class(segment):
    """
    figure out what to call the segment
    """
    if segment not in opt.phoible.index:
        return 'unknown'
    if opt.phoible.loc[segment, 'syllabic'] == '+':
        return 'vowel'
    elif opt.phoible.loc[segment, 'consonantal'] == '-':
        return 'glide'
    elif opt.phoible.loc[segment, 'sonorant'] == '+':
        return 'sonorant'
    else:
        return 'obstruent'

def place(segment):
    if segment not in opt.phoible.index:
        return 'unknown'
    if opt.phoible.loc[segment, 'syllabic'] == '+':
        return 'vowel'
    elif opt.phoible.loc[segment, 'labial'] == '+':
        return 'labial'
    elif opt.phoible.loc[segment, 'coronal'] == '+':
        return 'coronal'
    elif opt.phoible.loc[segment, 'dorsal'] == '+':
        return 'dorsal'
    else:
        return 'other'

def height(vowel):
    if opt.phoible.loc[vowel, 'high'] == '+':
        return 'high'
    elif opt.phoible.loc[vowel, 'low'] == '+':
        return 'low'
    else:
        return 'mid'

def place_manner(segment):
    return

def is_language(char):
    return '<' in char and char not in {'<blank>', '<unk>', '<s>', '</s>'}

def script(char, options={'LATIN', 'CYRILLIC', 'GEORGIAN', 'GREEK', 'ARMENIAN', 'ARABIC', 'THAI', 'SYRIAC'}):
    if len(char) > 1:
        if char in {'<blank>', '<unk>', '<s>', '</s>'}:
            return 'meta'
        elif re.match(r'<[a-z][a-z][a-z]>', char):
            return 'LangID'
        else:
            return 'other'
    else:
        try:
            script_name = unicodedata.name(char).split(None, 1)[0]
            if script_name in options:
                return script_name
            else:
                return 'other'
        except ValueError:
            return 'unknown'

def pca_all_dims(X):
    """
    In order to figure out a good number of components to pca-reduce to
    """
    dim = X.shape[1]
    explained_variance = np.cumsum(
        PCA(n_components=dim).fit(X).explained_variance_ratio_)
    plt.plot(np.arange(1, explained_variance.shape[0] + 1), explained_variance)
    plt.show()

# still a problem with the groupby if 
def plot(characters, embeddings, key=None):
    """
    characters: the 
    """
    if key is not None:
        f = lambda x: key(x[0])
    else:
        f = None
    contiguous_keys = sorted(zip(characters, embeddings), key=f)
    for name, group in groupby(contiguous_keys, key=f):
        X = np.array(list(zip(*group))[1])
        plt.scatter(X[:, 0], X[:, 1], label=name)
    plt.legend()
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('vectors')
    parser.add_argument('-phoible', default='/home/bpop/thesis/mg2p/data/phoible-segments-features.tsv', type=read_phoible)
    opt = parser.parse_args()
    
    # is there noise from rarely-seen characters?
    labels, X = read_embeddings(opt.vectors)
    grapheme_counts = count_phonemes('../mg2p/models/lua/langid-all/corpus/src.train')
    # labels, X = zip(*[(l, x) for l, x in zip(labels, X) if script(l) in {'LATIN', 'GREEK', 'ARABIC', 'CYRILLIC', 'CJK'} and not is_language(l)])
    # labels, X = zip(*[(l, x) for l, x in zip(labels, X) if l in {'<eng>', '<deu>', '<kat>', '<nld>', '<ltz>'}])
    labels, X = zip(*[(l, x) for l, x in zip(labels, X) if grapheme_counts[l] >= 500 and not is_language(l)])
    
    '''
    phoneme_counts = count_phonemes('../mg2p/models/lua/langid-all/corpus/tgt.train')
    labels, X = zip(*[(l, x) for l, x in zip(labels, X) if phoneme_counts[l] >= 50])
    # labels, X = zip(*[(l, x) for l, x in zip(labels, X) if phoneme_counts[l] >= 50 and segment_class(l) != 'vowel'])
    '''
    X = np.array(X)
    
    #pca_all_dims(X)
    
    transformations = [PCA(n_components=40), TSNE(metric="cosine", verbose=2, n_iter=2000, perplexity=50)]
    transformations.pop(0)
    for transform in transformations:
        X = transform.fit_transform(X)
    # plot(labels, X, lambda s: '_'.join([segment_class(s), place(s)]))
    plot(labels, X, key=script)
    

if __name__ == '__main__':
    main()
