#!/usr/bin/env python

import pandas as pd

WALS_LANGUAGE_PATH = '/home/bpop/thesis/mg2p/data/wals/language.csv'

def make_wals_data(path):
    wf = pd.read_csv(path)
    wf['iso_code'] = wf['iso_code'].replace(to_replace='bos', value='hbs') # iso 639-3 problems
    return wf
    
wals_frame = make_wals_data(WALS_LANGUAGE_PATH)

def get_genus(data):
    """
    data: DataFrame containing source side data, target side data, and the
            language
    returns: a Series identifying the language family (genus level, like
            Germanic) of each line
    """
    genus_frame = wals_frame[['iso_code', 'genus']]
    genus = genus_frame.groupby('iso_code').first().squeeze()
    return genus.loc[data['lang']]
    
def get_countries(data):
    """
    data: DataFrame containing source side data, target side data, and the
            language
    returns: a Series identifying the countries associated with the
            language of each line
    """
    # note: when I actually used this one, it was without bosnian being 
    # replaced by hbs
    
    # problem: there are duplicate country tokens for many languages, ie
    # two <SE>s for Swedish
    country_frame = wals_frame[['iso_code', 'countrycodes']]
    countries = country_frame.groupby('iso_code').apply(lambda x: ' '.join(str(u) for u in x['countrycodes'].unique())).squeeze()
    return countries.loc[data['lang']]
