#!/usr/bin/env python3

"""
in: lines from the wiktionary data
out: those lines written to two files, src.all and tgt.all, in the passed
    directory
    
"""

import sys
from os.path import join, dirname, splitext

for filename in sys.argv[1:]:
    directory = dirname(filename)
    datatype = splitext(filename)[1]
    src_name = join(directory, 'src' + datatype)
    tgt_name = join(directory, 'tgt' + datatype)
    with open(filename, encoding='utf-8') as infile, open(src_name, 'w', encoding='utf-8') as src, open(tgt_name, 'w', encoding='utf-8') as tgt:
        for line in infile:
            columns = line.split('\t')
            src.write(columns[0].lower() + '\n')
            tgt.write(columns[1])
    
