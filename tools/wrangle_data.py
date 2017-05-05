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
from tools.lua_functions import preprocess
        
# wouldn't it be better to just take arbitrarily many Series, 
def prepend_tokens(source_data, *args):
    """
    source_data: a Series of source-side orthographic data
    other arguments: str or Series. Values in prependers are put
                inside angle brackets to make sure they never get confused
                with normal orthographic symbols
    returns: a Series consisting of training samples with the appropriate
    """
    tokens = ['<' + arg + '>' for arg in args]
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
    
def write_model(path, languages, scripts, features):
    feature_map = {'langid':get_language}
    create_model_dir(path)
    train, validate, test = wiki.generate_pron_data(languages, scripts) # exact format of each of these?
    # here: auxiliary data sources
    
    for name, frame in [('train', train), ('dev', validate), ('test', test)]:
        print('Writing file: ' + join(path, 'corpus', 'src.' + name))
        # convert the features to be used into the appropriate
        #for feature in features:
        prependers = [feature_map[feature](frame) for feature in features]
        source_data = prepend_tokens(frame['spelling'], *prependers)
        source_data.to_csv(join(path, 'corpus', 'src.' + name), index=False)
        print('Writing file: ' + join(path, 'corpus', 'tgt.' + name))
        frame['ipa'].to_csv(join(path, 'corpus', 'tgt.' + name), index=False)
