#!/usr/bin/env python

"""
Tools for reading Deri & Knight's wiktionary pronunciation data.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from os.path import join

TRAINING_DATA_PATH = '/home/bpop/thesis/mg2p/data/deri-knight/pron_data/gold_data_train'
TEST_DATA_PATH = '/home/bpop/thesis/mg2p/data/deri-knight/pron_data/gold_data_test'

HIGH_RESOURCE = ['heb', 'kor', 'kur', 'zho', 'yue', 'ast', 'yid', 'ukr', 
                'ang', 'isl', 'nob', 'fin', 'mya', 'jpn', 'tgl', 'nan', 
                'ady', 'swe', 'tha', 'pus', 'vol', 'lao', 'san', 'amh', 
                'ltz', 'hun', 'sco', 'sga', 'fra', 'afr', 'nno', 'dsb', 
                'gle', 'por', 'epo', 'tel', 'nci', 'slv', 'pol', 'bak', 
                'fao', 'tur', 'gla', 'lat', 'slk', 'hbs', 'ben', 'vie', 
                'eng', 'hin', 'dan', 'cat', 'oci', 'ces', 'hye', 'deu', 
                'ron', 'lit', 'spa', 'ita', 'ido', 'tgk', 'nld', 'jbo', 
                'kat', 'aze', 'sqi', 'arc', 'fas', 'mkd', 'ara', 'lav', 
                'kbd', 'bre', 'scn', 'bul', 'cym', 'urd', 'rus', 'ain', 
                'ell', 'syc', 'msa', 'mlt', 'eus']

def read_data(path, languages=None, scripts=None, min_samples=50):
    """
    path: location of one of Deri & Knight's pronunciation tables
    languages: languages to take from the data (default: take all)
    scripts: scripts to take from the data (default: take all)
    min_samples: minimum number 
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
    selected_langs = select_languages(df, languages)
    selected_langs_and_scripts = select_scripts(selected_langs, scripts)
    
    # these next few lines are because of problems partitioning the data
    # if a language has only a few samples
    lang_counts = selected_langs_and_scripts.groupby('lang').size()
    sufficient_languages = lang_counts[lang_counts >= min_samples]
    indexer = selected_langs_and_scripts['lang'].isin(sufficient_languages.index)
    return selected_langs_and_scripts[indexer]
                
def select_languages(df, languages):
    """
    df: DataFrame representing wiktionary data
    returns: 
    """
    return _select_rows(df, 'lang', languages)
    
def select_scripts(df, scripts):
    return _select_rows(df, 'script', scripts)
    
def _select_rows(df, column, values):
    if values is not None:
        indexer = df[column].isin(values)
        return df.loc[indexer,:]
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
    
    #df.groupby('lang')
    
    # using the sklearn train_test_split may have been a nice hack here
    def lang_sample(frame):
        '''
        if isinstance(sample_size, int):
            return frame.sample(n=sample_size, random_state=0)
        else:
            # should be something dict-like
            language = frame.name
            if language in sample_size:
                return frame.sample(n=sample_size[language], random_state=0)
            else:
                return frame
        '''
        #return frame.sample(n=min(sample_size, frame.size), random_state=0)
        #print(frame.size)
        
        if sample_size >= frame.shape[0]:
            return frame
        else:
            result, _ = train_test_split(frame, train_size=sample_size, random_state=0)
            return result
            
    return df.groupby('lang').apply(lang_sample)
    
def partition_data(df, validation_size):
    """
    validation_size: integer or float: per-language size of the validation set
    returns: a partition of the data into training and validation
    """
    if validation_size > 1:
        # integer case: validation size specifies number of words
        num_languages = df['lang'].unique().size
        validation_size = num_languages * validation_size
    return train_test_split(df, test_size=validation_size, stratify=df['lang'], random_state=0)
        
def generate_pron_data(languages, scripts):
    if languages == ['high']:
        languages = HIGH_RESOURCE #a little hacky
    train_and_validate = sample(read_data(TRAINING_DATA_PATH, languages, scripts), 10000)
    train, validate = partition_data(train_and_validate, 0.1)
    test = read_data(TEST_DATA_PATH, languages, scripts)
    return train, validate, test
