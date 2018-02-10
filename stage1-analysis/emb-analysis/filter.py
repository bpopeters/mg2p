#!/usr/bin/env python

import argparse
import re
import unicodedata
import io
import sys
from collections import Counter
from itertools import chain

def count_tokens(path):
    with open(path) as f:
        return Counter(chain(*(line.split() for line in f)))

def writing_system(char):
    if len(char) > 1:
        if re.match(r'<[a-z][a-z][a-z]>', char):
            return 'LangID'
        else:
            return 'other'
    else:
        try:
            script_name = unicodedata.name(char).split(None, 1)[0]
            return script_name
        except ValueError:
            return 'unknown'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, default=1)
    parser.add_argument('-scripts', nargs='*')
    parser.add_argument('-train', type=count_tokens)
    opt = parser.parse_args()
    scripts = None if opt.scripts is None else set(opt.scripts)
    if opt.train is not None:
        included = {t 
                    for t in opt.train
                    if opt.train[t] >= opt.n
                    and (scripts is None or writing_system(t) in scripts)}
    else:
        included = None

    input_stream = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
    for line in input_stream:
        c = line.split()[0]
        if included is None or c in included:
            sys.stdout.write(line)

if __name__ == '__main__':
    main()
