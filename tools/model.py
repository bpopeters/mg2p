#!/usr/bin/env python

import sys
from os.path import join, exists
import os
import subprocess
import wiktionary as wiki
import pandas as pd
from itertools import chain
import re

# check: correct behavior if there are no features?
# also, missing thing: artificial tokens (it's okay, those are easy)
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
        tagged_words = []
        for word_and_features in zip(tokenized_words, *feature_sequences):
            word, feats = word_and_features[0], word_and_features[1:]
            tagged_word = ' '.join(['￨'.join(chain([w], feats)) for w in word])
            tagged_words.append(tagged_word)
        result = pd.Series(tagged_words, words.index)
        return result
    else:
        return data[side].copy() # copy necessary?
        
def epoch_number(path):
    return int(re.search(r'(?<=epoch)[0-9]+', path).group(0))
    
def negative_ppl(path):
    return -float(re.search(r'[0-9]+\.[0-9]+(?!\.t7)', path).group(0))

class G2PModel(object):
    
    opennmt_path = '/home/bpop/OpenNMT/'
    mg2p_path = '/home/bpop/thesis/mg2p'
    
    training_corpus = '/home/bpop/thesis/mg2p/data/deri-knight/pron_data/gold_data_train'
    test_corpus = '/home/bpop/thesis/mg2p/data/deri-knight/pron_data/gold_data_test'
            
    # two cases: you've already made the data and put it in the directory
    # (i.e. you've already done mg2p.py -preprocess)
    # (then you just need to make sure self.path and self.train_config
    # are going to the correct place (actually, maybe demote train_config...)
    def __init__(self, model_name, train_langs=None, train_scripts=None, src_features=[], tgt_features=[]):
        """
        model_name: unlike in previous versions, does not need to be formatted like a path.
                    The object will know where to put it.
        """
        self.path = join(self.mg2p_path, 'models', model_name)
        
        if not exists(self.path):
            self.create_model_dir()
            
            # select the data and add the relevant features
            data = self.make_data(train_langs, train_scripts)
            src_sequence = tag(data, 'src', *src_features)
            tgt_sequence = tag(data, 'tgt', *tgt_features)
            
            # write the src and tgt data to files
            for partition in ('train', 'dev', 'test'):
                src_sequence.loc[data['Partition'] == partition].to_csv(self.corpus_file('src', partition), index=False)
                tgt_sequence.loc[data['Partition'] == partition].to_csv(self.corpus_file('tgt', partition), index=False)
                    
            data.loc[data['Partition'] == 'test','lang'].to_csv(join(self.path, 'corpus', 'lang_index.test'), index=False)
                
            # do the torch preprocessing part
            self.preprocess()
        else:
            print('Proceeding with already created data at {}'.format(self.path))
            if any([train_langs, train_scripts, src_features, tgt_features]):
                print('Disregarding some of the passed arguments: if dir already exists, only pass the model_name')
        
    def create_model_dir(self):
        os.makedirs(join(self.path, 'corpus'))
        os.makedirs(join(self.path, 'nn'))
        print('Made model directory at {}'.format(self.path))
        
    def make_data(self, train_langs, train_scripts):
        training, validation = wiki.generate_partitioned_train_validate(train_langs, train_scripts)
        test = wiki.generate_test()
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
        if how == 'latest':
            keyfunc = epoch_number
        elif how == 'best':
            keyfunc = negative_ppl
        else:
            raise ValueError
        return max((p for p in os.listdir(join(self.path, 'nn'))), key=keyfunc)
        
    def preprocess(self):
        print('Beginning preprocessing...')
        os.chdir(self.opennmt_path)
        subprocess.run(['th', 'preprocess.lua', '-train_src', self.corpus_file('src', 'train'), 
                        '-train_tgt', self.corpus_file('tgt', 'train'), 
                        '-valid_src', self.corpus_file('src', 'dev'), 
                        '-valid_tgt', self.corpus_file('tgt', 'dev'), 
                        '-save_data', join(self.path, 'corpus', 'data'),
                        '-src_seq_length', '150',
                        '-tgt_seq_length', '150'])
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
        
    def translate(self, how='latest'):
        """
        todo: optionally translating something other than src.test
        """
        network = self.pick_network(how=how)
        print('Translating with model {}'.format(network))
        
        source = self.corpus_file('src', 'test')
        target = self.corpus_file('tgt', 'test')
        
        os.chdir(self.opennmt_path)
        subprocess.run(['th', 'translate.lua', '-model', join(self.path, 'nn', network), 
            '-src', source, '-tgt', target, 
            '-output', join(self.path, 'predicted.txt'),
            '-log_file', join(self.path, 'translate.log')])
        os.chdir(self.mg2p_path)
