#!/usr/bin/env python

import argparse
import sys
from itertools import chain
import pandas as pd

def read_embeddings(emb_file):
    df = pd.read_csv(emb_file, sep=' ', header=None, index_col=0)
    df.index.names = ['character']
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('grapheme_embs', type=read_embeddings)
    parser.add_argument('language_embs', type=read_embeddings)
    parser.add_argument('-g', nargs='+', default=[chr(i + 97) for i in range(26)])
    parser.add_argument('-l', nargs='+', default=['eng', 'spa'])
    opt = parser.parse_args()
    graphemes = opt.grapheme_embs.loc[opt.g]
    languages = opt.language_embs.loc[opt.l]
    for grapheme in graphemes.index:
        g_emb = graphemes.loc[grapheme]
        for lang in languages.index:
            l_emb = languages.loc[lang]
            sys.stdout.write(
                ' '.join(
                    chain(['+'.join([grapheme, lang])],
                    map(str, g_emb),
                    map(str, l_emb))) + '\n')

if __name__ == '__main__':
    main()
