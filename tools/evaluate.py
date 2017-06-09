#!/usr/bin/env python

"""
Functions for measuring the performance of g2p models.
"""

from os.path import join, dirname
import pandas as pd
import numpy as np
import argparse
import sys

HIGH_RESOURCE = ['ady', 'afr', 'ain', 'amh', 'ang', 'ara', 'arc', 'ast', 
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
                
UNSEEN = ['ruo', 'kmv', 'nhv', 'kum', 'ota', 'nau', 'eto', 'afb', 'lug', 
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

@np.vectorize
def levenshtein(a, b):
    """
    computes the levenshtein distance between sequences a and b.
    New: it's broadcastable!
    """
    d = [[0 for i in range(len(b) + 1)] for j in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        d[i][0] = i
    for j in range(1, len(b) + 1):
        d[0][j] = j
    for j in range(1, len(b) + 1):
        for i in range(1, len(a) + 1):
            # future: abstraction. cost depends on phoneme similarity
            cost = int(a[i - 1] != b[j - 1])
            d[i][j] = min(d[i][j - 1] + 1, d[i - 1][j] + 1, d[i - 1][j - 1] + cost)
    return d[len(a)][len(b)]

def per(results):
    """
    results: DataFrame containing at minimum columns for predicted and
            gold standard phoneme sequences
    returns: phoneme error rate of the predictions
    """
    # this is one of two possible ways to compute PER
    return levenshtein(results['predicted'], results['gold']).sum() / results['gold'].apply(len).sum()

def wer(results):
    """
    results: DataFrame containing at minimum columns for predicted and
            gold standard phoneme sequences
    returns: word error rate of the predictions
    """
    return (results['predicted'] != results['gold']).sum() / results['predicted'].size
    
def wer100(results):
    """
    results: DataFrame containing at minimum columns for top 100 predicted and
            gold standard phoneme sequences
    returns: WER 100 of the predictions
    """
    return (not_in(results['gold'], results['predicted-100'])).sum() / results['predicted'].size
    
def wer1(results):
    return (not_in(results['gold'], results['predicted-1'])).sum() / results['predicted'].size
        
def word_series(corpus_file):
    with open(corpus_file) as f:
        return pd.Series([tuple(line.strip().split()) for line in f])
    
def corpus_size(data, index_to_use=None):
    """
    data: src.train or src.test file
    returns quantity of training data per language
    """
    training_size = language_labels(data).value_counts()
    if index_to_use is not None:
        training_size = training_size[index_to_use].fillna(0)
    return training_size.astype('int64')
    
def raw_output(model_path):
    """
    returns a table containing columns for the language, predicted phonemes,
    and gold phonemes for the given model
    """
    source_test = join(model_path, 'corpus', 'src.test')
    target_test = join(model_path, 'corpus', 'tgt.test')
    predicted_test = join(model_path, 'predicted.txt')
    
    lang_id = language_labels(source_test)
    gold_words = read_words(target_test)
    predicted_words = read_words(predicted_test)
    
    return pd.DataFrame.from_items([('lang', lang_id), ('gold', gold_words), ('predicted', predicted_words)])

def evaluate_single_model(path):
    """
    model_path: model directory, containing the corpus subdirectory and the
                results on src.test
    returns: DataFrame: rows are for languages and summaries over test sets,
            columns are for WER and PER
    """
    name_mapper = {'langid-85':'LangID-High', 'langid-229':'LangID-Adapted', 'langid-all':'LangID-All',
                    'nolangid-85':'NoLangID-High', 'nolangid-229':'NoLangID-Adapted', 'nolangid-all':'NoLangID-All'}
    raw_name = path.split('/')[1] # change this
    model_name = name_mapper.get(raw_name, raw_name)
    
    predicted = word_series(join(path, 'predicted.txt')) # is it necessary to do it like this?
    
    print(predicted)
    
    '''
    predicted_1 = read_nbest(join(path, 'predicted-100.txt'), n=1)
    predicted_100 = read_nbest(join(path, 'predicted-100.txt'), n=100)
    '''
    gold = word_series(join(path, 'corpus', 'tgt.test'))
    
    
    lang_id = pd.read_csv(join(path, 'corpus', 'lang_index.test'), na_filter=False, header=None).squeeze()
    
    #results = pd.DataFrame.from_items([('lang', lang_id), ('gold', gold), ('predicted', predicted), ('predicted-100', predicted_100), ('predicted-1', predicted_1)])
    results = pd.DataFrame.from_items([('lang', lang_id), ('gold', gold), ('predicted', predicted)])
    #results.loc[results['lang'] == 'spa',:]
    phones = results.groupby('lang').apply(per)
    words = results.groupby('lang').apply(wer)
    
    '''
    words1 = results.groupby('lang').apply(wer1)
    words100 = results.groupby('lang').apply(wer100)
    df = pd.DataFrame.from_items([('WER', words1), ('WER 100', words100), ('PER', phones)])
    '''
    df = pd.DataFrame.from_items([('WER', words), ('PER', phones)])
    df = df * 100
    df.loc['all',:] = df.mean() # seems to work
    df.loc['high resource',:] = df.loc[HIGH_RESOURCE,:].mean()
    df.loc['adapted',:] = df.loc[ADAPTED,:].mean()
    df.loc['unseen',:] = df.loc[UNSEEN,:].mean()
    df['model'] = model_name
    return df
    
def exp_result_table(experiment_results, test_set):
    print(experiment_results.groupby('model').apply(lambda df: df.loc[test_set,['WER', 'WER 100', 'PER']]).to_latex(float_format='%.2f'))
    

def evaluate(models):
    """
    models: a sequence of directories containing neural nets and corpora.
            Models should already have been translated on the full test set.
    returns: A table. The index has rows for each language present in the test data,
            plus additional rows summarizing the results on the 85-language high
            resource languages, the 229-language adapted languages (see
            Deri and Knight's paper for details), and on the entire test
            set.
    """
    return pd.concat([evaluate_single_model(path) for path in models])
    
def read_nbest(path, beam=100, n=100):
    with open(path) as f:
        args = [(tuple(line.strip().split()) for line in f)] * beam
        return pd.Series(list(zip(*args))).apply(lambda seq: set(seq[:n]))
        
@np.vectorize
def not_in(x, y):
    return x not in y
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('models', nargs='*', default=[], help="Paths to models")
    opt = parser.parse_args()
    
    model_stats = evaluate(opt.models)
    '''
    exp_result_table(model_stats, 'adapted')
    exp_result_table(model_stats, 'high resource')
    exp_result_table(model_stats, 'unseen')
    '''
    model_stats.to_csv('langid-wals-pfeatures-all-results.csv', sep='\t', float_format='%.3f')
    
if __name__ == '__main__':
    main()
    '''
    path = '/home/bpop/thesis/mg2p/models/langid-all'
    predicted_100 = read_nbest(join(path, 'predicted.txt'), n=100)
    gold = word_series(join(path, 'corpus', 'tgt.test'))
    '''
        
