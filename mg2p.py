#!/usr/bin/env python

from os.path import join
import os
from tools.wrangle_data import write_model
from tools.lua_functions import preprocess, train, translate
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('name', help="Path to model")
parser.add_argument('-preprocess', action='store_true',
        help='Create model directory and populate it with wiktionary data')
parser.add_argument('-f', '--features',
        nargs='*',
        default=['langid'],
        help='Fake tokens to add to the beginning of each source-side line (default: langid)')
parser.add_argument('-train', action='store_true',
        help='Train the model')
parser.add_argument('-translate', action='store_true',
        help='Translate the model')
parser.add_argument('-train_config', default=None,
        help='OpenNMT parameters for training')
parser.add_argument('-l', '--lang',
        nargs='*',
        default=None,
        help='If preprocessing, languages for which to select data (default: all)')
parser.add_argument('-s', '--script',
        nargs='*',
        default=None,
        help='If preprocessing, scripts for which to select data (default: all)')
opt = parser.parse_args()

OPENNMT_PATH = '/home/bpop/OpenNMT/'
MG2P_PATH = '/home/bpop/thesis/mg2p'
    
def main():
    if not any([opt.preprocess, opt.train, opt.translate]):
        print('Specify at least one action (preprocess, train, test)')
        sys.exit()
    if opt.preprocess:
        write_model(opt.name, opt.lang, opt.script, opt.features)
        preprocess(opt.name)
    if opt.train:
        train(opt.name, opt.train_config) # but with the right configuration file
        
    if opt.translate:
        # todo: make it possible to infer which model to use better
        translate(opt.name, 'epoch13')

if __name__ == '__main__':
    main()
