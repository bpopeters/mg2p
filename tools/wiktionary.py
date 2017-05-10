#!/usr/bin/env python

"""
Tools for reading Deri & Knight's wiktionary pronunciation data.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from os.path import join

TRAINING_DATA_PATH = '/home/bpop/thesis/mg2p/data/deri-knight/pron_data/gold_data_train'
TEST_DATA_PATH = '/home/bpop/thesis/mg2p/data/deri-knight/pron_data/gold_data_test'

HIGH_RESOURCE = ['ady', 'afr', 'ain', 'amh', 'ang', 'ara', 'arc', 'ast', 
                'aze', 'bak', 'ben', 'bre', 'bul', 'cat', 'ces', 'cym', 
                'dan', 'deu', 'dsb', 'ell', 'eng', 'epo', 'eus', 'fao', 
                'fas', 'fin', 'fra', 'gla', 'gle', 'hbs', 'heb', 'hin', 
                'hun', 'hye', 'ido', 'isl', 'ita', 'jbo', 'jpn', 'kat', 
                'kbd', 'kor', 'kur', 'lao', 'lat', 'lav', 'lit', 'ltz', 
                'mkd', 'mlt', 'msa', 'mya', 'nan', 'nci', 'nld', 'nno', 
                'nob', 'oci', 'pol', 'por', 'pus', 'ron', 'rus', 'san', 
                'scn', 'sco', 'sga', 'slk', 'slv', 'spa', 'sqi', 'swe', 
                'syc', 'tel', 'tgk', 'tgl', 'tha', 'tur', 'ukr', 'urd', 
                'vie', 'vol', 'yid', 'yue', 'zho']
                
LOW_RESOURCE = ['aar', 'abk', 'abq', 'ace', 'ach', 'agr', 'aka', 'akl', 
                'akz', 'ale', 'alt', 'ami', 'aqc', 'arg', 'arw', 'arz', 
                'asm', 'ava', 'aym', 'bal', 'bam', 'bcl', 'bel', 'bis', 
                'bod', 'bos', 'bua', 'bug', 'ceb', 'cha', 'che', 'chk', 
                'chm', 'cho', 'chv', 'cic', 'cjs', 'cor', 'crh', 'dar', 
                'est', 'ewe', 'fij', 'fil', 'frr', 'fry', 'fur', 'gaa', 
                'gag', 'glg', 'grc', 'grn', 'gsw', 'guj', 'hak', 'hat', 
                'hau', 'haw', 'hil', 'hit', 'hrv', 'iba', 'ilo', 'ind', 
                'inh', 'jam', 'jav', 'kaa', 'kab', 'kal', 'kan', 'kaz', 
                'kea', 'ket', 'khb', 'kin', 'kir', 'kjh', 'kom', 'kum', 
                'lin', 'lld', 'lug', 'luo', 'lus', 'lzz', 'mah', 'mal', 
                'mar', 'mlg', 'mnk', 'mns', 'moh', 'mon', 'mri', 'mus', 
                'mww', 'myv', 'mzn', 'nah', 'nap', 'nau', 'nds', 'nep', 
                'new', 'nia', 'niu', 'non', 'nor', 'nso', 'oss', 'osx', 
                'pag', 'pam', 'pan', 'pau', 'pon', 'ppl', 'prs', 'que', 
                'roh', 'rom', 'rtm', 'ryu', 'sac', 'sah', 'sat', 'sei', 
                'sme', 'sna', 'snd', 'som', 'sot', 'srd', 'srp', 'sun', 
                'swa', 'tam', 'tat', 'tay', 'tir', 'tkl', 'tly', 'tpi', 
                'tsn', 'tuk', 'tvl', 'twi', 'tyv', 'udm', 'uig', 'umb', 
                'unk', 'uzb', 'wbp', 'wol', 'wuu', 'xal', 'xho', 'xmf', 
                'yap', 'yij', 'yor', 'yua', 'zha', 'zul', 'zza']

def read_data(path, languages=None, scripts=None):
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
    selected_langs = select_languages(df, languages) # sorta spaghetti
    selected_langs_and_scripts = select_scripts(selected_langs, scripts)
    return selected_langs_and_scripts
                
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
    # using the sklearn train_test_split may have been a nice hack here
    def lang_sample(frame):
        
        if sample_size >= frame.shape[0]:
            return frame
        else:
            result, _ = train_test_split(frame, train_size=sample_size, random_state=0)
            return result
    return df.groupby('lang').apply(lang_sample)
    
# this part here is the problem
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
    if languages == ['high']:
        languages = HIGH_RESOURCE #a little hacky
    elif languages == ['all_lang']:
        languages = HIGH_RESOURCE + LOW_RESOURCE
    train_and_validate = sample(read_data(TRAINING_DATA_PATH, languages, scripts), 10000)
    train, validate = partition_data(train_and_validate, 0.1)
    test = read_data(TEST_DATA_PATH, languages, scripts)
    return train, validate, test
