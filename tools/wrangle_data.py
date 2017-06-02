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
    
def word_level_features(source_data, feature):
    """
    source_data: a Series of source-side orthographic data
    feature: a Series. Add the element at each index
                    of the Series to 
    """
    return source_data.str.split(expand=True).apply(lambda char: char.str.cat(feature, sep='|')).apply(lambda row: ' '.join(row.dropna()), axis=1)
        
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
    
def get_vocab(path):
    """
    Returns the symbols in the target vocab in the ordering from the
    tgt.dict file. Any fictional character (one in angle brackets) is
    represented as an opening angle bracket.
    """
    with open(path) as f:
        return [line.split()[0] if '<' not in line else '<' for line in f]
    
def write_model(path, languages, scripts, tokens, features):
    """
    path: location at which to write model
    languages: languages to include in model
    scripts: scripts to include in model
    tokens: artificial tokens to prepend to each source-side file
    """
    create_model_dir(path)
    train, validate = wiki.generate_partitioned_train_validate(languages, scripts)
    test = wiki.generate_test() # every model gets the same test set
    
    for name, frame in [('train', train), ('dev', validate), ('test', test)]:
        print('Writing file: ' + join(path, 'corpus', 'src.' + name))
        source_data = frame['spelling']
        if 'langid' in tokens:
            source_data = prepend_tokens(source_data, get_language(frame))
            #source_data = word_level_features(source_data, get_language(frame))
            
            
        source_data.to_csv(join(path, 'corpus', 'src.' + name), index=False)
        print('Writing file: ' + join(path, 'corpus', 'tgt.' + name))
        frame['ipa'].to_csv(join(path, 'corpus', 'tgt.' + name), index=False)
        
    # new for evaluation: a file which specifies the language at each
    # line of the test.
    test['lang'].to_csv(join(path, 'corpus', 'lang_index.test'), index=False) # note the lack of angle brackets
        
    preprocess(path)
