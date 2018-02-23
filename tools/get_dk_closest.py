#!/usr/bin/env python

import argparse
from os.path import join
from collections import Counter
import re
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('indir')
parser.add_argument('outdir')
parser.add_argument('-dist', nargs='+',
                    help="Which distance metrics to annotate mg2p data with")
parser.add_argument('-lang2lang',
                    default='../data/deri-knight/lang2lang/lang.dists')
parser.add_argument('-train_index', 
                    default='../models/lua/langid-all/corpus/lang_index.train')
parser.add_argument('-train_threshold', type=int)
opt = parser.parse_args()

DK_HIGH = ['ady', 'afr', 'ain', 'amh', 'ang', 'ara', 'arc', 'ast', 
                'aze', 'bak', 'ben', 'bre', 'bul', 'cat', 'ces', 'cym', 
                'dan', 'deu', 'dsb', 'ell', 'eng', 'epo', 'eus', 'fao', 
                'fas', 'fin', 'fra', 'gla', 'gle', 'hbs', 'heb', 'hin', 
                'hun', 'hye', 'ido', 'isl', 'ita', 'jbo', 'jpn', 'kat', 
                'kbd', 'kor', 'kur', 'lao', 'lat', 'lav', 'lit', 'ltz', 
                'mkd', 'mlt', 'msa', 'mya', 'nan', 'nci', 'nld', 'nno', 
                'nob', 'oci', 'pol', 'por', 'pus', 'ron', 'rus', 'san', 
                'scn', 'sco', 'sga', 'slk', 'slv', 'spa', 'sqi', 'swe', 
                'syc', 'tel', 'tgk', 'tgl', 'tha', 'tur', 'ukr', 'urd', 
                'vie', 'vol', 'yid', 'yue', 'zho']
                
ADAPTED = ['aar', 'abk', 'abq', 'ace', 'ach', 'ady', 'afr', 'agr', 
                'aka', 'akl', 'akz', 'ale', 'alt', 'ami', 'aqc', 'ara', 
                'arg', 'arw', 'arz', 'asm', 'ava', 'aym', 'aze', 'bak', 
                'bal', 'bam', 'bcl', 'bel', 'ben', 'bis', 'bod', 'bos', 
                'bre', 'bua', 'bug', 'bul', 'cat', 'ceb', 'ces', 'cha', 
                'che', 'chk', 'chm', 'cho', 'chv', 'cic', 'cjs', 'cor', 
                'crh', 'cym', 'dan', 'dar', 'deu', 'dsb', 'eng', 'est', 
                'eus', 'ewe', 'fao', 'fas', 'fij', 'fil', 'fin', 'fra', 
                'frr', 'fry', 'fur', 'gaa', 'gag', 'gla', 'gle', 'glg', 
                'grc', 'grn', 'gsw', 'guj', 'hak', 'hat', 'hau', 'haw', 
                'hbs', 'heb', 'hil', 'hin', 'hit', 'hrv', 'hun', 'iba', 
                'ilo', 'ind', 'inh', 'isl', 'ita', 'jam', 'jav', 'kaa', 
                'kab', 'kal', 'kan', 'kaz', 'kbd', 'kea', 'ket', 'khb', 
                'kin', 'kir', 'kjh', 'kom', 'kum', 'kur', 'lat', 'lav', 
                'lin', 'lit', 'lld', 'lug', 'luo', 'lus', 'lzz', 'mah', 
                'mal', 'mar', 'mkd', 'mlg', 'mlt', 'mnk', 'mns', 'moh', 
                'mon', 'mri', 'msa', 'mus', 'mww', 'mya', 'myv', 'mzn', 
                'nah', 'nap', 'nau', 'nci', 'nds', 'nep', 'new', 'nia', 
                'niu', 'nld', 'nob', 'non', 'nor', 'nso', 'oci', 'oss', 
                'osx', 'pag', 'pam', 'pan', 'pau', 'pol', 'pon', 'por', 
                'ppl', 'prs', 'pus', 'que', 'roh', 'rom', 'ron', 'rtm', 
                'rus', 'ryu', 'sac', 'sah', 'san', 'sat', 'scn', 'sei', 
                'slv', 'sme', 'sna', 'snd', 'som', 'sot', 'spa', 'sqi', 
                'srd', 'srp', 'sun', 'swa', 'swe', 'tam', 'tat', 'tay', 
                'tel', 'tgk', 'tgl', 'tir', 'tkl', 'tly', 'tpi', 'tsn', 
                'tuk', 'tur', 'tvl', 'twi', 'tyv', 'udm', 'uig', 'ukr', 
                'umb', 'unk', 'urd', 'uzb', 'vie', 'wbp', 'wol', 'wuu', 
                'xal', 'xho', 'xmf', 'yap', 'yid', 'yij', 'yor', 'yua', 
                'yue', 'zha', 'zho', 'zul', 'zza']

def get_donor(language, metric):
    if language in {'hbs', 'bos', 'hrv'}:
        language = 'bos'
    language = re.sub(r'"', '', language)
    return metric.get(language, language)

def annotate_feature(infile, outfile, distance_dicts):
    """
    infile: src.{train, dev, test}
    outfile: that with the extra feature
    closest: dict-like, maps from receiver to donor
    """
    with open(infile) as inf, open(outfile, 'w') as outf:
        for line in inf:
            in_tokens = [tok.split("￨") for tok in line.strip().split()]
            out_tokens = ("￨".join(tok + [get_donor(tok[1], closest) for closest in distance_dicts])
                          for tok in in_tokens)
            outf.write(" ".join(out_tokens) + '\n')


def get_closest_language(frame, metric):
    return frame.loc[frame[metric].argmin(), 'code2']
            

def main():
    columns = ['code1', 'code2'] + opt.dist
    lang2lang = pd.read_csv(
        opt.lang2lang, sep='\t', usecols=columns, na_values='None',
        keep_default_na=False
    )
    if opt.train_threshold:
        with open(opt.train_index) as f:
            train_counts = Counter(line.strip() for line in f)
        high_resource = [lang for lang, count in train_counts.items()
                         if count >= opt.train_threshold]
    else:
        high_resource = DK_HIGH
    # what I'd really like to do: have multiple train thresholds
    # so there can be features for most similar with at least 200, 500, etc.
    # lang2lang = lang2lang.loc[lang2lang['code1'].isin(ADAPTED) & lang2lang['code2'].isin(HIGH_RESOURCE)]
    lang2lang.loc[:, opt.dist] = lang2lang.loc[:, opt.dist].fillna(1)
    lang2lang = lang2lang.loc[lang2lang['code2'].isin(high_resource)]

    distance_dicts = []
    for metric in opt.dist:
        closest = lang2lang.groupby('code1').apply(
                lambda frame: get_closest_language(frame, metric)
        ).to_dict()
        distance_dicts.append(closest)
    files = ['src.train', 'src.dev', 'src.test']
    for f in files:
        annotate_feature(join(opt.indir, f), join(opt.outdir, f), distance_dicts)

if __name__ == '__main__':
    main()
