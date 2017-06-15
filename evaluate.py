#!/usr/bin/env python

"""
evaluate.py computes result statistics for arbitrarily many 
"""

import argparse
from tools.model import G2PModel

parser = argparse.ArgumentParser()
parser.add_argument('-models',
            nargs='*', 
            help="Paths to models")
opt = parser.parse_args()

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
    
def main():
    for name in opt.models:
        model = G2PModel(name)
        model.evaluate().to_csv('langid_rules_results.csv', sep='\t', float_format='%.3f')
    
if __name__ == '__main__':
    main()
