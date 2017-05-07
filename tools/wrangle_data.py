#!/usr/bin/env python

"""
Utilities for converting pronunciation data (from Deri & Knight's wiktionary
corpus) and linguistic data (from URIEL) into an mg2p model directory that
can subsequently be used for training and translation.
"""

import os
from os.path import join
import tools.wiktionary as wiki # purpose of wiki: turn D&K stuff into pandas data structures
# here: import a module for the uriel stuff
from tools.lua_functions import preprocess, serialize_vectors
import pandas as pd
from collections import Counter

# wouldn't it be better to just take arbitrarily many Series, 
def prepend_tokens(source_data, *args):
    """
    source_data: a Series of source-side orthographic data
    other arguments: str or Series. Values in prependers are put
                inside angle brackets to make sure they never get confused
                with normal orthographic symbols
    returns: a Series consisting of training samples with the appropriate
    """
    tokens = [arg.apply('<{}>'.format) for arg in args]
    return tokens[0].str.cat(tokens[1:] + [source_data], sep=' ')
        

def create_model_dir(path):
    os.makedirs(join(path, 'corpus'))
    os.makedirs(join(path, 'nn'))
    print('Made model directory at {}'.format(path))
    
def get_language(data):
    """
    data: DataFrame containing source side data, target side data, and the
            language
    returns: a Series identifying the language of each line
    """
    return data['lang']
    
def get_geo_clusters(data):
    clusters = pd.read_csv('/home/bpop/thesis/mg2p/data/latlongclusters.csv', names=['lang', 'cluster'], index_col='lang').squeeze()
    return clusters.loc[data['lang']]
    
def read_phoible(path):
    """
    path: location of PHOIBLE tsv 
    """
    df = pd.read_csv(path, sep='\t').replace(['-', '+', '-,+', '+,-'], [-1, 1, 0, 0])
    df = df.drop_duplicates('segment')
    df = df.set_index('segment')
    df.loc['<'] = 0
    return df
    
def get_vocab(path):
    """
    Returns the symbols in the target vocab in the ordering from the
    tgt.dict file. Any fictional character (one in angle brackets) is
    represented as an opening angle bracket.
    """
    with open(path) as f:
        return [line.split()[0] if '<' not in line else '<' for line in f]
    
def write_model(path, languages, scripts, features, phoneme_vectors):
    """
    path: location at which to write model
    languages: languages to include in model
    scripts: scripts to include in model
    features: feature tokens to 
    """
    feature_map = {'langid':get_language, 'geo':get_geo_clusters}
    create_model_dir(path)
    train, validate, test = wiki.generate_pron_data(languages, scripts) # exact format of each of these?
    # here: auxiliary data sources
    
    for name, frame in [('train', train), ('dev', validate), ('test', test)]:
        print('Writing file: ' + join(path, 'corpus', 'src.' + name))
        # convert the features to be used into the appropriate
        #for feature in features:
        prependers = [feature_map[feature](frame) for feature in features]
        if prependers:
            source_data = prepend_tokens(frame['spelling'], *prependers)
        else:
            source_data = frame['spelling']
        source_data.to_csv(join(path, 'corpus', 'src.' + name), index=False)
        print('Writing file: ' + join(path, 'corpus', 'tgt.' + name))
        frame['ipa'].to_csv(join(path, 'corpus', 'tgt.' + name), index=False)
        
    preprocess(path)
    if phoneme_vectors:
        # not gonna consider multiple data sources: just using phoible for now
        phoible = read_phoible('/home/bpop/thesis/mg2p/data/phoible-segments-features.tsv')
        tgt_phones = get_vocab(join(path, 'corpus', 'data.tgt.dict'))
        raw_vectors = phoible.loc[tgt_phones,:].fillna(0)
        raw_vectors.to_csv(join(path, 'corpus', 'raw_vectors.csv'), sep=',', index=False, header=False)
        serialize_vectors(join(path, 'corpus', 'raw_vectors.csv'), join(path, 'corpus', 'phones.vec'))
