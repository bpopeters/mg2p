#!/usr/bin/env python

import argparse
from ggplot import *
import pandas as pd

def read_projections(path):
    return pd.read_csv(path, sep=' ', index_col='character')

def read_phoible(phoible_file):
    df = pd.read_csv(phoible_file, sep='\t', na_filter=False)
    df = df.drop_duplicates(subset='segment')
    df = df.set_index('segment')
    # sonority = (df[['syllabic', 'approximant', 'sonorant', 'continuant', 'delayedRelease']] == '+').sum(axis=1)
    # df['sonority_hierarchy'] = sonority
    return df

def plot_qual(emb, phoible, column):
    data = emb.join(phoible, how='left').fillna('unknown')
    plot = ggplot(aes(x='x', y='y', color=column), data=data) +\
        geom_point() +\
        scale_color_brewer(type='qual', palette=6)
    return plot

def get_name(symbol):
    if symbol == '+':
        return 'Vowel'
    elif symbol == '-':
        return 'Consonant'
    elif symbol == '0':
        return 'Tone'
    else:
        return 'Unknown'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('projections', type=read_projections)
    parser.add_argument('-plot_path')
    parser.add_argument('-phoible', type=read_phoible)
    opt = parser.parse_args()
    # print(opt.phoible)
    # print(pd.concat([opt.projections, opt.phoible], axis=1))
    data = opt.projections.join(opt.phoible, how='left').fillna('unknown')
    # data = opt.projections.join(opt.phoible, how='left')
    data = data.reset_index()
    
    data['Category'] = data['syllabic'].apply(get_name)

    # if you use brewer with qual, 6 is the palette you want
    
    '''
    plot = ggplot(aes(x='x', y='y', label='character'), data=data) +\
        geom_text()
    '''
    plot = ggplot(aes(x='x', y='y', color='Category', label='character'), data=data) +\
        geom_text() +\
        scale_color_brewer(type='qual', palette=6)
    # plot = plot_qual(opt.projections, opt.phoible, 'syllabic')
    if opt.plot_path:
        plot.save(opt.plot_path, width=10, height=10)
    else:
        plot.show()

if __name__ == '__main__':
    main()
