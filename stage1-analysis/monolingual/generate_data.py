#!/usr/bin/env python

import argparse
import os
from os.path import join
import pandas as pd

def lang_index(path):
    with open(path) as f:
        return [line.strip() for line in f]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-languages', default='languages.txt', type=lang_index)
    parser.add_argument('-train_index', default='lang_index.train', type=lang_index)
    parser.add_argument('-test_index', default='lang_index.test', type=lang_index)
    parser.add_argument('-valid_index', default='lang_index.dev', type=lang_index)
    parser.add_argument('-train_src', default='src.train', type=lang_index)
    parser.add_argument('-valid_src', default='src.dev', type=lang_index)
    parser.add_argument('-test_src', default='src.test', type=lang_index)
    parser.add_argument('-train_tgt', default='tgt.train', type=lang_index)
    parser.add_argument('-valid_tgt', default='tgt.dev', type=lang_index)
    parser.add_argument('-test_tgt', default='tgt.test', type=lang_index)
    parser.add_argument('-model_dir', default='mono-models')
    opt = parser.parse_args()
    train = pd.DataFrame(data={'src': opt.train_src, 'tgt': opt.train_tgt}, index=opt.train_index)
    test = pd.DataFrame(data={'src': opt.test_src, 'tgt': opt.test_tgt}, index=opt.test_index)
    valid = pd.DataFrame(data={'src': opt.valid_src, 'tgt': opt.valid_tgt}, index=opt.valid_index)

    for lang in opt.languages:
        if lang in train.index and lang in test.index and lang in valid.index:
            lang_dir = join(opt.model_dir, lang)
            os.mkdir(lang_dir)
            os.mkdir(join(lang_dir, 'corpus'))
            os.mkdir(join(lang_dir, 'nn'))
            for side in ['src', 'tgt']:
                for label, dataset in zip(['train', 'test', 'dev'], [train, test, valid]):
                    dataset.loc[lang, side].to_csv(join(lang_dir, 'corpus', side + '.' + label), index=False)
        else:
            print(lang)

if __name__ == '__main__':
    main()
