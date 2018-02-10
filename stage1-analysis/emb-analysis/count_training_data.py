#!/usr/bin/env python

from itertools import chain
from collections import Counter
import sys

def count_phonemes(tgt_file):
    with open(tgt_file) as f:
        return Counter(chain(*(line.split() for line in f)))

if __name__ == '__main__':
    phoneme_counts = count_phonemes(sys.argv[1])
    print(sum(phoneme_counts.values()))
    print(sum(v < 50 for v in phoneme_counts.values()))
    print(len(phoneme_counts))
    '''
    for phoneme, count in phonem_counts.most_common():
        print(phoneme, count)
    '''
