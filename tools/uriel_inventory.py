#!/usr/bin/env python

"""
Script for providing access to the phonology and inventory features from
PHOIBLE as stored in the URIEL database v0_3_0. Functionality is probably
similar to lang2vec; however, this covers only a portion of things one
might want to use lang2vec for and is designed to fit nicely with the
mg2p tools.
"""

import pandas as pd

DATA = '/home/bpop/thesis/mg2p/data/uriel_v0_3_0/features/avg.csv'

def get_vowels(languages):
    df = pd.read_csv(DATA, na_values='--', index_col='G_CODE', keep_default_na=False)
    vowels = ['G_CODE', 'INV_VOW_10_MORE', 'INV_VOW_9', 'INV_VOW_8', 'INV_VOW_7', 'INV_VOWEL_6', 'INV_VOWEL_5', 'INV_VOWEL_4', 'INV_VOWEL_3']
    vowel_inv = df.loc[:,vowels].idxmax(axis=1).fillna('<unk>')
    
    lang_inventories = vowel_inv.loc[languages]
    return lang_inventories.fillna('<unk>')
    
'''
def main():
    vowels = ['INV_VOW_10_MORE', 'INV_VOW_9', 'INV_VOW_8', 'INV_VOW_7',
    'INV_VOWEL_6', 'INV_VOWEL_5', 'INV_VOWEL_4', 'INV_VOWEL_3']
    df = pd.read_csv(DATA, na_values='--', index_col='G_CODE', keep_default_na=False)
    
    print(df.loc[:,vowels].idxmax(axis=1).notnull().sum())
    print(df.loc[df['INV_VOW_10_MORE'] > 0].index)
    
    print(df.loc[:,vowels].sum(axis=1).sort_values())
    print(df.loc[['zap','zho'],vowels])
    
    #df.notnull().sum().sort_values().to_csv('avg_counts.csv', sep='\t', float_format='%.1f')
'''
    
