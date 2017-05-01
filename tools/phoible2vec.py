#!/usr/bin/env python3

"""
Takes the tsv containing articulatory features and converts it into a
csv representing each with 3 bits in the manner described by Deri & Knight
"""

import pandas as pd
import re

if __name__ == '__main__':
    convert = {'+':'1,1,0', '-':'1,0,1', '0':'0,0,0'}
    df = pd.read_csv('data/phoible-segments-features.tsv', sep='\t', index_col='segment')
    
    # problem: some features have multiple values specified
    df = df.apply(lambda s: s.apply(lambda p: convert.get(p, '1,1,1'))) # in the case of +,- or -,+, give it 1,1,1
    #df.to_csv('data/phoible-3-bit.csv', sep=',', header=False)
    string_table = df.to_csv(None, sep=',', header=False) # neat trick
    with open('data/phoible-3-bit.csv', 'w', encoding='utf-8') as f:
        f.write(','.join(map(str, range(37 * 3 + 1))) + '\n')
        f.write(re.sub(r'"', '', string_table))
        f.write('<,' + ','.join(['0' for i in range(37 * 3)]))
