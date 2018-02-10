#!/usr/bin/env python

"""
Take a language.
Take the test data
"""

import os
from os.path import join
from itertools import chain
import pandas as pd


    
def make_src_test(lang, test, no_langid):
    if no_langid:
        test_path = join(lang, 'no_langid.src.test')
    else:
        test_path = join(lang, 'src.test')
    ool_test_data = test.loc[test['lang'] != lang, 'spelling'].apply(
        lambda word: ' '.join(w.lower().strip() for w in word)
    )
    if not no_langid:
        ool_test_data = '<{}> '.format(lang) + ool_test_data
    ool_test_data.to_csv(test_path, index=False)
    return join(opt.this_dir, test_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-test_data', default='pron_data/gold_data_test')
    opt = parser.parse_args()
    train, test = read_data(opt.test_data)
    for lang in sorted(test['lang'].unique()):
        print(lang)
        os.makedirs(lang, exist_ok=True)
        make_inventory(train, lang)
        src_path = make_src_test(lang, test, opt.no_langid)
        print(src_path)
        # now: make the test data
        #other_langs = test.loc[test['lang'] != lang, 'spelling'].apply(lambda word: '<{}> '.format(lang) + ' '.join(w.lower().strip() for w in word))
        # other_langs.to_csv(join(lang, 'src.test'), index=False)
        pred_path = join(opt.this_dir, lang, pred_name)
        translate(lang, model, src_path, pred_path)

        

if __name__ == '__main__':
    main()
