#!/usr/bin/env python

"""
Utilities for converting pronunciation data (from Deri & Knight's wiktionary
corpus) and linguistic data (from URIEL) into an mg2p model directory that
can subsequently be used for training and translation.

The point of wrangle_data is to take several sources of data that have
been created elsewhere, combine them, put them in an OpenNMT-friendly
format, and write them to a directory that can be read by the training
and translating 
"""

import os
from os.path import join
import tools.wiktionary as wiki # purpose of wiki: turn D&K stuff into pandas data structures
from tools.lua_functions import preprocess, serialize_vectors
import tools.uriel_inventory as ur_inv
import pandas as pd
import numpy as np

# wouldn't it be better to just take arbitrarily many Series,
def prepend_tokens(source_data, *args):
    """
    source_data: a Series of source-side orthographic data
    other arguments: str or Series. Values in prependers are put
                inside angle brackets to make sure they never get confused
                with normal orthographic symbols
    returns: a Series consisting of training samples with the appropriate tokens attached to the front
    """
    # this is very ugly
    tokens = [arg.apply(lambda x: '<{}>'.format('_'.join(str(x).split()))) for arg in args]
    return tokens[0].str.cat(tokens[1:] + [source_data], sep=' ')

@np.vectorize    
def add_line_features(word, feature):
    return ' '.join([char + feature for char in word])
    
def word_level_features(source_data, *features):
    dummy = pd.Series('', source_data.index)
    features = dummy.str.cat(features, sep='￨')
    print(features.notnull().all())
    
    tagged_words = add_line_features(source_data.str.split(), features)
    return pd.Series(tagged_words)
        
# unexplained: why this is here
def get_language(data):
    """
    data: DataFrame containing source side data, target side data, and the
            language
    returns: a Series identifying the language of each line
    """
    return data['lang']
    
def create_model_dir(path):
    os.makedirs(join(path, 'corpus'))
    os.makedirs(join(path, 'nn'))
    print('Made model directory at {}'.format(path))
    
# quasi-main method
def write_model(path, languages, scripts, tokens, features):
    """
    path: location at which to write model
    languages: languages to include in model
    scripts: scripts to include in model
    tokens: artificial tokens to prepend to each source-side file
    """
    create_model_dir(path)
    # it takes 10 seconds from starting mg2p.py to get here. Why? Various imports?
    train, validate = wiki.generate_partitioned_train_validate(languages, scripts)
    test = wiki.generate_test() # every model gets the same test set
    
    for name, frame in [('train', train), ('dev', validate), ('test', test)]:
        print('Writing file: ' + join(path, 'corpus', 'src.' + name))
        source_data = frame['spelling']
        # having both tokens and features might not be compatible
        
        '''
        if 'langid' in tokens:
            source_data = prepend_tokens(source_data, get_language(frame))
            #source_data = word_level_features(source_data, get_language(frame))
        if 'langid' in features:
            source_data = word_level_features(source_data, get_language(frame))
        if 'vowels' in features:
            source_data = word_level_features(source_data, ur_inv.get_vowels(get_language(frame)))
        '''
        if 'langid' in features:
            lang_index = get_language(frame)
            source_data = word_level_features(source_data, lang_index)
            
            
        source_data.to_csv(join(path, 'corpus', 'src.' + name), index=False)
        print('Writing file: ' + join(path, 'corpus', 'tgt.' + name))
        frame['ipa'].to_csv(join(path, 'corpus', 'tgt.' + name), index=False)
        
    # new for evaluation: a file which specifies the language at each
    # line of the test.
    test['lang'].to_csv(join(path, 'corpus', 'lang_index.test'), index=False) # note the lack of angle brackets
        
    preprocess(path)
