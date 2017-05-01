#!/usr/bin/env python

from os.path import join
import os
from tools import wiktionary as wikt
from tools.lua_functions import preprocess, train, translate
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('name', help="Path to model")
parser.add_argument('-preprocess', action='store_true')
parser.add_argument('-train', action='store_true')
parser.add_argument('-translate', action='store_true')
parser.add_argument('-train_config', default=None)
parser.add_argument('-l', '--lang',
        nargs='*',
        default=None, # can't think of a wildcard that would make the thing work
        help='Languages for which to select data (default: all)') # doesn't work yet
parser.add_argument('-s', '--script',
        nargs='*',
        default=None, # can't think of a wildcard that would make
        help='Languages for which to select data (default: all)')
opt = parser.parse_args()

OPENNMT_PATH = '/home/bpop/OpenNMT/'
MG2P_PATH = '/home/bpop/thesis/mg2p'
DATA_PATH = '/home/bpop/thesis/mg2p/data/pron_data/'

def create_model_dir(path):
    os.makedirs(join(path, 'corpus'))
    os.makedirs(join(path, 'nn'))
    print('Made model directory at {}'.format(path))
    
# this should probably go in the wiktionary thing
def populate_model_dir(path, languages, scripts):
    training_path = join(DATA_PATH, 'gold_data_train')
    test_path = join(DATA_PATH, 'gold_data_test')
    train_and_validate = wikt.read_data(training_path, languages, scripts)
    train, validate = wikt.partition_data(train_and_validate, 0.1)
    test = wikt.read_data(test_path, languages, scripts)
    for name, frame in [('train', train), ('dev', validate), ('test', test)]:
        print('Writing file: ' + join(path, 'corpus', 'src.' + name))
        wikt.write_file(join(path, 'corpus', 'src.' + name), frame['spelling'], frame['lang'])
        print('Writing file: ' + join(path, 'corpus', 'tgt.' + name))
        wikt.write_file(join(path, 'corpus', 'tgt.' + name), frame['ipa'])
    
def main():
    if not any([opt.preprocess, opt.train, opt.translate]):
        print('Specify at least one action (preprocess, train, test)')
        sys.exit()
    if opt.preprocess:
        create_model_dir(opt.name)
        populate_model_dir(opt.name, opt.lang, opt.script)
        preprocess(opt.name)
    if opt.train:
        train(opt.name, opt.train_config) # but with the right configuration file
        
    if opt.translate:
        translate(opt.name)

if __name__ == '__main__':
    main()
