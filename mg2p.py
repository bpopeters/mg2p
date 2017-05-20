#!/usr/bin/env python

from os.path import join
import os
from tools.wrangle_data import write_model
from tools.lua_functions import train, translate
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('name', help="Path to model")
parser.add_argument('-preprocess', action='store_true',
        help='Create model directory and populate it with wiktionary data')
parser.add_argument('-f', '--features',
        nargs='*',
        default=[],
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
parser.add_argument('-d', '--data', default='wiktionary',
        help='Whether to train with wiktionary or ipahelp (default: wiktionary)')
parser.add_argument('-p', '--phoneme_vectors', default=None,
        help='Data source for fixed phoneme embeddings for the decoder (are there really multiple options?)')
opt = parser.parse_args()
    
def main():
    if not any([opt.preprocess, opt.train, opt.translate]):
        print('Specify at least one action (preprocess, train, test)')
        sys.exit()
    if opt.preprocess:
        write_model(opt.name, opt.lang, opt.script, opt.features, opt.data, opt.phoneme_vectors)
    if opt.train:
        train(opt.name, opt.train_config)
    if opt.translate:
        translate(opt.name, 'epoch13')

if __name__ == '__main__':
    main()
