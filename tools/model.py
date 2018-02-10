#!/usr/bin/env python

import sys
from os.path import join, exists
import os
import subprocess
import tools.wiktionary as wiki # weird things happen with imports
from tools.features import G2PRule, AvgG2PFeature
import pandas as pd
from itertools import chain, cycle
import re
import tools.eval_funcs as eval_funcs

def feature_cycles(features):
    result = []
    for feat in features:
        if isinstance(feat, str):
            result.append(cycle([feat]))
        else:
            result.append(cycle(feat))
    return result

def tag_line(word, *feats):
    """
    word: a list of singleton strings, ie characters
    feats: a list. Each element of the list is a string. Each of those strings
            should be joined to each character.
            However, what if 
    """
    tagged_characters = ['￨'.join(chr_and_feats) for chr_and_feats in zip(word, *feature_cycles(feats))]
    return ' '.join(tagged_characters)

def tag(data, side, *features):
    """
    data: the wiktionary data table
    side: src or tgt
    features: each feature is a function that takes a single argument:
                the data (should it also take the side?)
    returns: a Series consisting of the source or target word sequence
            to use for the experiment.
    """
    # pandas vectorized string options are way too slow. This is better.
    if features:
        words = data[side]
        feature_sequences = [f(data) for f in features]
        tokenized_words = words.str.split() # make sure there isn't weird NA stuff
        tagged_words = [tag_line(*word_and_features) for word_and_features in zip(tokenized_words, *feature_sequences)]
        return pd.Series(tagged_words, words.index)
    else:
        return data[side].copy() # copy necessary?
        
def epoch_number(path):
    return int(re.search(r'(?<=epoch)[0-9]+', path).group(0))
    
def negative_ppl(path):
    return -float(re.search(r'[0-9]+\.[0-9]+(?!\.t7)', path).group(0))
    
def dummy_feature(data):
    """
    data: the corpus table
    returns: 
    """
    return data['src'].str.split()

class G2PModel(object):
    
    opennmt_path = '/home/bpop/OpenNMT/'
    mg2p_path = '/home/bpop/thesis/mg2p'
    model_dir = 'models/'
    
    training_corpus = '/home/bpop/thesis/mg2p/data/deri-knight/pron_data/gold_data_train'
    test_corpus = '/home/bpop/thesis/mg2p/data/deri-knight/pron_data/gold_data_test'
    
    feature_lookup = {'langid': lambda data: data['lang'],
                        'rules':G2PRule().get_feature,
                        'grapheme_feat':AvgG2PFeature().get_feature} # this bit may change
            
    def __init__(self, model_name, train_langs=None, train_scripts=None, src_features=[], tgt_features=[]):
        """
        model_name: unlike in previous versions, does not need to be formatted like a path.
                    The object will know where to put it.
        """
        self.path = join(self.mg2p_path, model_name)
        
        if not exists(self.path):
            self.create_model_dir()
            
            # select the data and add the relevant features
            data = self.make_data(train_langs, train_scripts, train_langs, train_scripts)
            data = data.loc[(data['src'].str.len() > 0) & (data['tgt'].str.len() > 0)]

            src_sequence = tag(data, 'src', *self._look_up_features(src_features))
            tgt_sequence = tag(data, 'tgt', *self._look_up_features(tgt_features))
            
            # write the src and tgt data to files
            for partition in ('train', 'dev', 'test'):
                src_sequence.loc[data['Partition'] == partition].to_csv(self.corpus_file('src', partition), index=False)
                tgt_sequence.loc[data['Partition'] == partition].to_csv(self.corpus_file('tgt', partition), index=False)

            data.loc[data['Partition'] == 'train','lang'].to_csv(join(self.path, 'corpus', 'lang_index.train'), index=False)
            data.loc[data['Partition'] == 'test','lang'].to_csv(join(self.path, 'corpus', 'lang_index.test'), index=False)
        else:
            print('Proceeding with already created data at {}'.format(self.path))
            if any([train_langs, train_scripts, src_features, tgt_features]):
                print('Disregarding some of the passed arguments: if dir already exists, only pass the model_name')
                
    def _look_up_features(self, feature_names):
        return [self.feature_lookup[name] for name in feature_names if name in self.feature_lookup]
        
    def create_model_dir(self):
        os.makedirs(join(self.path, 'corpus'))
        os.makedirs(join(self.path, 'nn'))
        print('Made model directory at {}'.format(self.path))
        
    def make_data(self, train_langs, train_scripts, test_langs=None, test_scripts=None):
        training, validation = wiki.generate_partitioned_train_validate(train_langs, train_scripts)
        
        test = wiki.generate_test(test_langs, test_scripts) # achtung!
        
        # Keeping wiki as it is now so as not to break the old way of doing things.
        training['Partition'] = 'train'
        validation['Partition'] = 'dev'
        test['Partition'] = 'test'
        raw_data = pd.concat([training, validation, test]).reset_index(drop=True)
        return raw_data.rename(columns={'spelling':'src', 'ipa':'tgt'})
        
    def corpus_file(self, side, partition):
        """
        side: src or tgt
        partition: train, dev, or test
        returns: path to use for the corresponding corpus file
        """
        return join(self.path, 'corpus', '{}.{}'.format(side, partition))
        
    def pick_network(self, how='latest'):
        """
        Returns the path to the network to be used for translation
        """
        # if latest, then you pick the one with the biggest epoch number
        assert how in ['latest', 'best'], 'Please use a defined how'
        if how == 'latest':
            keyfunc = epoch_number
        elif how == 'best':
            keyfunc = negative_ppl
        return max((p for p in os.listdir(join(self.path, 'nn'))), key=keyfunc)
        
    def preprocess(self):
        print('Beginning preprocessing...')
        os.chdir(self.opennmt_path)
        subprocess.run(['th', 'preprocess.lua', '-train_src', self.corpus_file('src', 'train'), 
                        '-train_tgt', self.corpus_file('tgt', 'train'), 
                        '-valid_src', self.corpus_file('src', 'dev'), 
                        '-valid_tgt', self.corpus_file('tgt', 'dev'), 
                        '-save_data', join(self.path, 'corpus', 'data')]) # removed len-150 src/tgt sequences
        os.chdir(self.mg2p_path)
            
    def train(self, train_config=None):
        command = ['th', 'train.lua', 
                '-data', join(self.path, 'corpus', 'data-train.t7'), 
                '-save_model', join(self.path, 'nn', 'model')]
        if train_config:
            # consider copying train_config to the directory: it's useful
            command.extend(['-config', join(self.mg2p_path, train_config)])
        os.chdir(self.opennmt_path)
        subprocess.run(command)
        os.chdir(self.mg2p_path)
        
    def translate(self, how='latest', source=None, target=None):
        """
        todo: optionally translating something other than src.test
        """
        network = self.pick_network(how=how)
        print('Translating with model {}'.format(network))
        
        if not source:
            source = self.corpus_file('src', 'test')
        if not target:
            target = self.corpus_file('tgt', 'test')
        
        os.chdir(self.opennmt_path)
        subprocess.run(['th', 'translate.lua', '-model', join(self.path, 'nn', network), 
            '-src', source, '-tgt', target, 
            '-output', join(self.path, 'predicted.txt'),
            '-log_file', join(self.path, 'translate.log')])
        os.chdir(self.mg2p_path)
        
    def evaluate(self, gold_path=None, predicted_path=None, lang_index_path=None, out_path=None, lang_subsets=[]):
        if not gold_path:
            gold_path = self.corpus_file('tgt', 'test')
        if not predicted_path:
            predicted_path = join(self.path, 'predicted.txt')
        if not lang_index_path:
            lang_index_path = join(self.path, 'corpus', 'lang_index.test')
            
        lang_index_test = pd.read_csv(
                    lang_index_path, header=None, na_filter=False).squeeze()
        lang_index_train = pd.read_csv(
                    join(self.path, 'corpus', 'lang_index.train'),
                    header=None, na_filter=False).squeeze()
        gold = pd.read_csv(gold_path, header=None,
                           squeeze=True, skip_blank_lines=False).str.split()
        predicted = pd.read_csv(predicted_path, header=None,
                                squeeze=True, skip_blank_lines=False,
                                na_filter=False).str.split()
        df = pd.DataFrame.from_items([('lang', lang_index_test),
                                      ('gold', gold),
                                      ('predicted', predicted)])
        test_counts = df.groupby('lang').size()
        train_counts = lang_index_train.value_counts().loc[test_counts.index]
        train_counts = train_counts.fillna(0)
        per = df.groupby('lang').apply(eval_funcs.per)
        wer = df.groupby('lang').apply(eval_funcs.wer)
        #sub_errors = df.groupby('lang').apply(eval_funcs.substitution_errors)
        
        results = pd.DataFrame.from_items([('WER', wer),
                                           ('PER', per),
                                           ('Train', train_counts),
                                           ('Test', test_counts)])
        results.loc['all',:] = results.mean()
        results['Train'] = results['Train'].astype(int)
        results['Test'] = results['Test'].astype(int)
        for n, lang_sub in enumerate(lang_subsets, 1):
            results.loc['subset ' + str(n),:] = results.loc[lang_sub,:].mean()
        if not out_path:
            return results
        results.to_csv(out_path, sep='\t', float_format='%.3f')
        
class G2PModel_py(G2PModel):
    """"""
    
    opennmt_path = None # shouldn't need to do this kind of io stuff
    mg2p_path = None
    
    def preprocess(self):
        """
        gotta do something with preprocess.py instead.
        for which, gotta import the module and such
        """
        pass
    
        
