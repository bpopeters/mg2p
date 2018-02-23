#!/usr/bin/env python

import pandas as pd

data = pd.read_csv('/home/bpop/thesis/mg2p/data/dev-2014/data/FEATURES/phoible-segments-features.tsv', sep='\t', index_col='segment')

def tag(segments):
    """
    segments: sequence of IPA symbols
    returns: articulatory features for each segment
    """
    # this is in fact very general.
    # make the features a second argument and it still works
    
    # ughhhh
    features = data.loc[segments.split()].apply('￨'.join, axis=1)
    return pd.Series(segments).str.cat(features, sep='￨')
    
if __name__ == '__main__':
    
    print(tag(list('p'), lambda x: data.loc[x]))
    print(tag(['tʷ', 'p'], lambda x: data.loc[x]))
    print(tag(['a'], lambda x: x + ['b']))
