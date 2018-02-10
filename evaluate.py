#!/usr/bin/env python

"""
evaluate.py computes result statistics for arbitrarily many
"""

import argparse
from itertools import groupby, count
from collections import Counter
import pandas as pd


high = ['ady', 'afr', 'ain', 'amh', 'ang', 'ara', 'arc', 'ast',
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

adapted = ['aar', 'abk', 'abq', 'ace', 'ach', 'ady', 'afr', 'agr',
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

unseen = ['ruo', 'kmv', 'nhv', 'kum', 'ota', 'nau', 'eto', 'afb', 'lug',
        'apy', 'pam', 'gnc', 'gez', 'gaa', 'nch', 'ave', 'mlv', 'kmg',
        'tkl', 'pms', 'iku', 'sea', 'rgn', 'pnw', 'pny', 'jam', 'gmh',
        'vec', 'yij', 'nov', 'prs', 'kal', 'lkt', 'pis', 'pro', 'pac',
        'mns', 'lac', 'sty', 'lld', 'cqd', 'sgs', 'bor', 'nso', 'meo',
        'ntj', 'hoi', 'rif', 'abq', 'div', 'chc', 'aln', 'dbl', 'alq',
        'ckt', 'aot', 'stq', 'asm', 'liv', 'blc', 'niu', 'gub', 'hau',
        'rup', 'nij', 'nab', 'mco', 'mwl', 'rmo', 'fur', 'jav', 'dng',
        'srn', 'ace', 'kir', 'cho', 'yor', 'pcd', 'sun', 'yap', 'tyz',
        'pdt', 'inh', 'tly', 'msn', 'amn', 'sva', 'cim', 'agr', 'wym',
        'nep', 'orv', 'pjt', 'srd', 'ude', 'tsd', 'hit', 'nio', 'csi',
        'bdq', 'kea', 'nhn', 'hat', 'vma', 'pag', 'ssf', 'kut', 'twi',
        'mah', 'crs', 'cal', 'niv', 'kom', 'lin', 'aqc', 'frr', 'sna',
        'gwi', 'ext', 'rtm', 'tir', 'mnk', 'rad', 'aii', 'yii', 'snd',
        'avd', 'fil', 'crg', 'evn', 'abe', 'dar', 'kjh', 'abz', 'aak',
        'bio', 'ing', 'mop', 'zai', 'new', 'abt', 'gbb', 'nxn', 'mus',
        'hei', 'apn', 'wgy', 'bzg', 'kxo', 'bam', 'gzi', 'chl', 'oji',
        'pap', 'shn', 'sjn', 'arw', 'mzn', 'adj', 'led', 'smk', 'akk',
        'gqn', 'wwo', 'myp', 'amp', 'ket', 'alr', 'oge', 'hak', 'kpv',
        'are', 'szl', 'kky', 'tay', 'nuk', 'lmo', 'nia', 'tqw', 'lus',
        'akl', 'wbp', 'xmf', 'blq', 'mvi', 'lzz', 'ckb', 'bug', 'ilo',
        'mis', 'xto', 'koy', 'swh', 'aym', 'mga', 'pre', 'axm', 'azn',
        'agg', 'pau', 'sje', 'sac', 'che', 'aau', 'taa', 'nhg', 'pdc',
        'apc', 'odt', 'iba', 'duj', 'vro', 'fax', 'vls', 'non', 'acw',
        'wnw', 'mcm', 'grn', 'mar', 'kuu']

SUBSETS = {'high': high, 'adapted': adapted, 'unseen': unseen}


def chunks(iterable, size):
    index = count()
    groups = groupby(iterable, key=lambda x: next(index) // size)
    return [list(g) for k, g in groups]


def levenshtein(a, b):
    """
    Why is dynamic programming always so ugly?
    """
    d = [[0 for i in range(len(b) + 1)] for j in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        d[i][0] = i
    for j in range(1, len(b) + 1):
        d[0][j] = j
    for j in range(1, len(b) + 1):
        for i in range(1, len(a) + 1):
            cost = int(a[i - 1] != b[j - 1])
            d[i][j] = min(d[i][j - 1] + 1,
                          d[i - 1][j] + 1, d[i - 1][j - 1] + cost)
    return d[len(a)][len(b)]


def wer(predicted, gold, n=1):
    """
    predicted, gold: equal length sequences of phonemes
    returns:
    """
    assert len(predicted) == len(gold)
    incorrect = sum(g not in set(p[:n]) for p, g in zip(predicted, gold))
    return incorrect / len(predicted)


def per(predicted, gold):
    assert len(predicted) == len(gold)
    total_distance = sum(levenshtein(p, g) for p, g in zip(predicted, gold))
    gold_length = sum(len(g) for g in gold)
    return total_distance / gold_length


def aligned_data(gold_file, pred_file, langs=None):
    with open(gold_file) as f:
        gold = [tuple(line.strip().split()) for line in f]
    with open(pred_file) as f:
        predicted = [tuple(line.strip().split()) for line in f]
    assert len(predicted) % len(gold) == 0

    predicted = chunks(predicted, len(predicted) // len(gold))
    best_pred = [p[0] for p in predicted]
    if langs is None:
        langs = ['lang' for p in predicted]

    data = pd.DataFrame(
        data={'gold': gold, 'all_pred': predicted, 'best_pred': best_pred},
        index=langs
    )
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('gold',
                        help="""Path to gold data.""")
    parser.add_argument('predicted',
                        help="""Path to model predictions. The file is allowed
                        to contain more than one prediction per word.""")
    parser.add_argument('-test_langs',
                        help="""Labels identifying the languages. Necessary
                        for computing error metrics per language.""")
    parser.add_argument('-train_langs',
                        help="""Labels identify the language of each training
                        sample. This allows us to study the relationship
                        between training data size and error rate""")
    parser.add_argument('out', help='Outfile name')
    parser.add_argument('-wer', nargs='+', type=int, default=[1])
    parser.add_argument('-no_subsets', action='store_true')
    parser.add_argument('-monolingual', action='store_true')
    opt = parser.parse_args()

    langs = None
    test_counts = Counter()
    train_counts = Counter()
    if opt.test_langs is not None:
        with open(opt.test_langs) as f:
            langs = [line.strip() for line in f]
            test_counts = Counter(langs)
    if opt.train_langs is not None:
        with open(opt.train_langs) as f:
            train_counts = Counter(line.strip() for line in f)

    data = aligned_data(opt.gold, opt.predicted, langs)

    metric_columns = dict()
    for n in opt.wer:
        name = 'wer_' + str(n)
        metric_columns[name] = data.groupby(level=0).apply(
            lambda df: wer(df['all_pred'], df['gold'], n)
        )
    metric_columns['per'] = data.groupby(level=0).apply(
        lambda df: per(df['best_pred'], df['gold'])
    )
    metrics = pd.DataFrame(data=metric_columns)
    metrics['train_count'] = [train_counts[l] for l in metrics.index]
    metrics['test_count'] = [test_counts[l] for l in metrics.index]

    averages = {'all': metrics.mean()}
    if not opt.no_subsets:
        for subset, langs in SUBSETS.items():
            averages[subset] = metrics.loc[langs, :].mean()

    summary = pd.DataFrame.from_dict(data=averages, orient='index')
    metrics = metrics.append(summary)
    metrics.to_csv(opt.out, sep='\t', float_format='%.4f')


if __name__ == '__main__':
    main()
