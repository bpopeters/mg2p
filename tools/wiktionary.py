#!/usr/bin/env python

"""
Tools for reading Deri & Knight's wiktionary pronunciation data.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from os.path import join

WIKTIONARY_TRAINING_PATH = '/home/bpop/thesis/mg2p/data/deri-knight/pron_data/gold_data_train'
TEST_DATA_PATH = '/home/bpop/thesis/mg2p/data/deri-knight/pron_data/gold_data_test'
IPAHELP_PATH = '/home/bpop/thesis/mg2p/data/deri-knight/ipa_help/all.g-to-ipa.cleaned.table'

def read_data(path, languages=None, scripts=None):
    """
    path: location of one of Deri & Knight's pronunciation tables
    languages: languages to take from the data (default: take all)
    scripts: scripts to take from the data (default: take all)
    """
    # perhaps different processing of the converters will be necessary
    # if I do anything with diacritics
    # also if I clean the wiktionary data
    df = pd.read_csv(path, sep='\t', 
                names=['lang', 'script', 'spelling', 'ipa', 'raw_ipa'],
                usecols=['lang', 'script', 'spelling', 'ipa'],
                converters={'spelling': lambda word: ' '.join(w.lower().strip() for w in word)},
                na_filter=False, #because there's a language with ISO 639-3 code nan
                encoding='utf-8')
    selected_langs = select_rows(df, 'lang', languages)
    selected_langs_and_scripts = select_rows(selected_langs, 'script', scripts)
    return selected_langs_and_scripts
    
def read_ipahelp(path):
    return pd.read_csv(path, sep='\t', 
                    names=['lang', 'ignore', 'spelling', 'script', 
                    'ipa', 'prob'], usecols=['lang', 'script', 
                    'spelling', 'ipa'])
    
def select_rows(df, column, values):
    if values is not None:
        return df.loc[df[column].isin(values),:]
    else:
        # should this return a copy instead?
        return df
        
def sample(df, sample_size):
    """
    Returns a subset of the passed DataFrame with at most n rows for each language
    sample_size: int or dictionary from language codes to ints. If an int, 
                returns a DataFrame with that many training samples for 
                each language. Using a dictionary allows for a different
                number of samples for each language. If a language present
                in the DataFrame is not present in the dictionary, all
                samples from that language will be kept.
    """
    # using the sklearn train_test_split may have been a nice hack here
    def lang_sample(frame):
        
        if sample_size >= frame.shape[0]:
            return frame
        else:
            result, _ = train_test_split(frame, train_size=sample_size, random_state=0)
            return result
    return df.groupby('lang').apply(lang_sample)
    
def partition_data(df, validation_size):
    """
    validation_size: float: maximum portion of data per language size of the validation set
    returns: a partition of the data into training and validation
    """
    def lang_sample(frame):
        if frame.shape[0] >= 10:
            return frame.sample(frac=0.9, random_state=0)
        else:
            return frame
    train = df.groupby('lang').apply(lang_sample).reset_index(drop=True, level=0)
    validation = df[~df.index.isin(train.index)]
    print(train['lang'].unique().size)
    print(validation['lang'].unique().size)
    return train, validation
    
def generate_pron_data(languages, scripts):
    train_and_validate = sample(read_data(WIKTIONARY_TRAINING_PATH, languages, scripts), 10000)
    train, validate = partition_data(train_and_validate, 0.1)
    
    test = read_data(TEST_DATA_PATH, languages, scripts)
    return train, validate, test
    
def generate_partitioned_train_validate(languages, scripts):
    train_and_validate = sample(read_data(WIKTIONARY_TRAINING_PATH, languages, scripts), 10000)
    train, validate = partition_data(train_and_validate, 0.1)
    return train, validate
    
def generate_test(languages=None, scripts=None):
    return read_data(TEST_DATA_PATH, languages, scripts)


