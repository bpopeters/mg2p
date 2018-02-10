#!/usr/bin/env python

import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

'''
def read_embeddings(emb_file):
    with open(emb_file) as f:
        return np.array([[float(x) for x in line.split()[1:]]for line in f])
'''

def read_embeddings(emb_file):
    df = pd.read_csv(emb_file, sep=' ', header=None, index_col=0)
    df.index.names = ['character']
    return df

def transform(X, transformations):
    labels = X.index
    for transform in transformations:
        X = transform.fit_transform(X)
    df = pd.DataFrame(data=X, index=labels)
    df.columns = ['x', 'y']
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('vectors', type=read_embeddings)
    parser.add_argument('out')
    parser.add_argument('-pca', action='store_true')
    parser.add_argument('-n_components', type=int, default=40)
    parser.add_argument('-tsne', action='store_true')
    parser.add_argument('-metric', default='cosine')
    parser.add_argument('-n_iter', type=int, default=2000)
    parser.add_argument('-ppl', type=int, default=50)
    opt = parser.parse_args()
    assert opt.pca or opt.tsne, "Why don't you want to project anything?"
    if not opt.tsne:
        assert opt.n_components == 2, "The plot needs to be 2d"
    transformations = []
    if opt.pca:
        transformations.append(PCA(n_components=opt.n_components))
    if opt.tsne:
        transformations.append(
            TSNE(metric=opt.metric, verbose=2,
                 n_iter=opt.n_iter, perplexity=opt.ppl)
        )
    output = transform(opt.vectors, transformations)
    output.to_csv(opt.out, sep=' ')

if __name__ == '__main__':
    main()
