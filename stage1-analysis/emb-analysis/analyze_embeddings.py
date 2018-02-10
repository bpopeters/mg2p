#!/usr/bin/env python

import argparse

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd

def read_embeddings(emb_file):
    characters = []
    vectors = []
    with open(emb_file) as f:
        for line in f:
            split_line = line.split()
            characters.append(split_line[0])
            vectors.append([float(x) for x in split_line[1:]])
    return characters, np.array(vectors)

def read_embeddings_pd(emb_file):
    return pd.read_csv(emb_file, sep=' ', header=None, index_col=0)

def read_phoible(phoible_file):
    return pd.read_csv(phoible_file, sep='\t', index_col='segment', na_filter=False)

parser = argparse.ArgumentParser()
parser.add_argument('vectors', type=read_embeddings)
parser.add_argument('-out', default='vecs.tsne')
parser.add_argument('-phoible', default='/home/bpop/thesis/mg2p/data/phoible-segments-features.tsv', type=read_phoible)
opt = parser.parse_args()

def is_language(char):
    return '<' in char and char not in {'<blank>', '<unk>', '<s>', '</s>'}

def main():
    characters, vectors = opt.vectors
    
    vowelset = set(opt.phoible.loc[opt.phoible['syllabic'] == '+'].index)
    
    '''
    for i in range(1, 20):
        pca = PCA(n_components=i)
        projected_vectors = pca.fit_transform(vectors)
        print(i, np.sum(pca.explained_variance_ratio_))
        print()
    '''
    plt.plot(np.arange(1, 151), PCA(n_components=150).fit(vectors).explained_variance_ratio_)
    plt.show()
    #vowels = np.array([v for c, v in zip(characters, projected_vectors) if c in vowelset])
    #non_vowels = np.array([v for c, v in zip(characters, projected_vectors) if c not in vowelset])
    
    
    '''
    tsne = TSNE(n_iter=10000, perplexity=100, n_iter_without_progress=1000, learning_rate=50, metric="cosine", verbose=2, early_exaggeration=12, method='exact')
    tsne.fit(vectors)
    print(tsne.kl_divergence_)
    print(tsne.n_iter)
    print(tsne.n_iter_final)
    print(tsne.get_params())
    
    projected_vectors = tsne.embedding_
    # languages = np.array([v for c, v in zip(characters, projected_vectors) if is_language(c)])
    # graphemes = np.array([v for c, v in zip(characters, projected_vectors) if not is_language(c)])
    # vowels = projected_vectors.loc[projected_vectors.index.isin(vowels_ph.index)]
    #non_vowels = projected_vectors.loc[~projected_vectors.index.isin(vowels_ph.index)]
    vowels = np.array([v for c, v in zip(characters, projected_vectors) if c in vowelset])
    non_vowels = np.array([v for c, v in zip(characters, projected_vectors) if c not in vowelset])
    
    np.savetxt(opt.out, projected_vectors)
    '''
    
    
    # plt.scatter(projected_vectors[:, 0], projected_vectors[:, 1])
    '''
    plt.scatter(vowels[:, 0], vowels[:, 1], c='blue')
    plt.scatter(non_vowels[:, 0], non_vowels[:, 1], c='green')
    plt.show()
    '''

if __name__ == '__main__':
    main()
