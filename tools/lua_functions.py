#!/usr/bin/env python

"""
A script that provides access and python functions for running the lua/torch
scripts in the OpenNMT directory. Those scripts have only worked for me
when I've run them from within the OpenNMT directory, which is why there's
so much cd'ing around in here.

In the long run, I would rather use the pytorch version of OpenNMT, but
I'm not there yet.
"""

import subprocess
import os
from os.path import join

OPENNMT_PATH = '/home/bpop/OpenNMT/'
MG2P_PATH = '/home/bpop/thesis/mg2p'

def preprocess(path):
    print('Beginning preprocessing...')
    corpus = join(MG2P_PATH, path, 'corpus')
    os.chdir(OPENNMT_PATH)
    subprocess.run(['th', 'preprocess.lua', '-train_src', join(corpus, 'src.train'), 
                    '-train_tgt', join(corpus, 'tgt.train'), 
                    '-valid_src', join(corpus, 'src.dev'), 
                    '-valid_tgt', join(corpus, 'tgt.dev'), 
                    '-save_data', join(corpus, 'data'),
                    'src_seq_length', 120,
                    'tgt_seq_length', 250])
    os.chdir(MG2P_PATH)
    
def train(path, config):
    # note the -fix_word_vecs_dec: not permanent
    command = ['th', 'train.lua', 
            '-data', join(MG2P_PATH, path, 'corpus/data-train.t7'), 
            '-save_model', join(MG2P_PATH, path, 'nn/model')]
    if config:
        command.extend(['-config', join(MG2P_PATH, config)])
    os.chdir(OPENNMT_PATH)
    subprocess.run(command)
    os.chdir(MG2P_PATH)
    
def translate(path, epoch='epoch13'):
    network = next((p for p in os.listdir(join(MG2P_PATH, path, 'nn')) if epoch in p))
    print('Translating with model {}'.format(network))
    os.chdir(OPENNMT_PATH)
    subprocess.run(['th', 'translate.lua', '-model', join(MG2P_PATH, path, 'nn', network), 
        '-src', join(MG2P_PATH, path, 'corpus/src.test'), '-tgt', join(MG2P_PATH, path, 'corpus/tgt.test'), 
        '-output', join(MG2P_PATH, path, 'predicted.txt'),
        '-log_file', join(MG2P_PATH, path, 'best_five.txt')])
    os.chdir(MG2P_PATH)
    
def extract_embeddings(path, epoch='epoch13'):
    network = next((p for p in os.listdir(join(MG2P_PATH, path, 'nn')) if epoch in p))
    os.chdir(OPENNMT_PATH)
    subprocess.run(['th', 'tools/extract_embeddings.lua', 
                    '-model', join(MG2P_PATH, path, 'nn', network), 
                    '-output_dir', path])
    os.chdir(MG2P_PATH)
    
def serialize_vectors(raw_path, serialized_path):
    subprocess.run(['th', '/home/bpop/thesis/mg2p/tools/serialize.lua', raw_path, serialized_path])
