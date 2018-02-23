#!/usr/bin/env python

import pandas as pd

class G2PRule(object):
    
    unknown = '<unk>'
    
    def __init__(self):
        
        df = pd.read_csv('/home/bpop/thesis/mg2p/data/deri-knight/ipa_help/all.g-to-ipa.cleaned.table', 
                        sep='\t', names=['lang', 'junk', 'g', 'script', 'p', 'prob'])
        df['g'] = df['g'].str.replace(r' ', '')
        df['p'] = df['p'].str.replace(r' ', '')
        self.lookup = df.groupby(('lang', 'g')).apply(lambda x: '_'.join(x['p']))
                        
    def get_feature(self, data):
        """
        data: corpus frame
        returns:
        """
        return pd.Series([self.tag_word(word, lang) for word, lang in zip(data['src'].str.split(), data['lang'])], index=data.index)
        
    def tag_char(self, character, lang):
        try:
            return self.lookup.loc[lang, character]
        except pd.core.indexing.IndexingError:
            return self.unknown
        except KeyError:
            return self.unknown
            
    def tag_word(self, word, lang):
        # a future thing: something less naive?
        return [self.tag_char(g, lang) for g in word]
            
class AvgG2PFeature(object):
    """Totally hacking this together for now"""
    
    unknown = '￨'.join(['<unk>'] * 37)
    
    def __init__(self):
        df = pd.read_csv('/home/bpop/thesis/mg2p/data/avg_grapheme_vectors.csv', sep='\t', index_col='g').astype(str)
        df = df.apply(lambda column: column.str.replace(r'1\.0', 'y'))
        df = df.apply(lambda column: column.str.replace(r'0\.0', 'n'))
        df = df.apply(lambda column: column.str.replace(r'0\.5', 'm'))
        self.lookup = df.apply('￨'.join, axis=1)
        
    def get_feature(self, data):
        """
        data: corpus frame
        returns:
        """
        return pd.Series([self.tag_word(word) for word in data['src'].str.split()], index=data.index)
        
    def tag_char(self, character):
        try:
            return self.lookup.loc[character]
        except pd.core.indexing.IndexingError:
            return self.unknown
        except KeyError:
            return self.unknown
            
    def tag_word(self, word):
        # a future thing: something less naive?
        return [self.tag_char(g) for g in word]
