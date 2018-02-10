#!/usr/bin/env python

from tools.model import G2PModel
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('name', help="Path to model")
'''
parser.add_argument('-t', '--tokens',
        nargs='*',
        default=[],
        help='Artificial tokens to add to the beginning of each source-side line, in practice always the langid feature (default: none of them)')
'''
parser.add_argument('-src_features',
        nargs='*',
        default=[],
        help='Character-level features to concatenate to the source input at each time step (default: none of them)')
parser.add_argument('-tgt_features',
        nargs='*',
        default=[],
        help='Character-level features to concatenate to the target input at each time step (default: none of them)')
parser.add_argument('-preprocess', action='store_true',
        help='Apply torch preprocessing to the training and validation data')
parser.add_argument('-train', action='store_true',
        help='Train the model')
parser.add_argument('-translate', action='store_true',
        help='Translate the model')
parser.add_argument('-train_config', default=None,
        help='OpenNMT parameters for training')
parser.add_argument('-l', '--lang',
        nargs='*',
        default=None,
        help='If preprocessing, languages for which to select data (default: all)')
parser.add_argument('-s', '--script',
        nargs='*',
        default=None,
        help='If preprocessing, scripts for which to select data (default: all)')
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
                
TYPO = ['aar', 'aau', 'abk', 'abq', 'abt', 'ace', 'ach', 'adj', 'ady', 
        'afr', 'agm', 'agr', 'aht', 'ain', 'aka', 'akl', 'akz', 'ale', 
        'alp', 'alr', 'alt', 'amh', 'amn', 'amp', 'apn', 'apy', 'aqc', 
        'are', 'arg', 'arn', 'arw', 'arz', 'asm', 'ava', 'ayl', 'bak', 
        'bam', 'ban', 'bar', 'bbc', 'bcl', 'bdr', 'ben', 'blc', 'bod', 
        'bor', 'bre', 'brg', 'bug', 'bul', 'cat', 'ccc', 'ceb', 'ces', 
        'cha', 'che', 'chl', 'cho', 'chr', 'chv', 'cic', 'cjs', 'ckt', 
        'cmn', 'com', 'cqd', 'crg', 'crh', 'cri', 'dbl', 'deu', 'dob', 
        'dru', 'duj', 'ell', 'eng', 'epo', 'eto', 'eus', 'evn', 'ewe', 
        'fij', 'fil', 'fin', 'fra', 'frr', 'fur', 'gaa', 'gag', 'gle', 
        'glg', 'gqn', 'gub', 'guj', 'gvf', 'hak', 'hau', 'haw', 'hdn', 
        'heb', 'hil', 'hin', 'hrv', 'hts', 'hun', 'hye', 'iba', 'ilo', 
        'ind', 'inh', 'isl', 'ita', 'itl', 'jam', 'jav', 'jpn', 'kaa', 
        'kab', 'kac', 'kal', 'kan', 'kat', 'kay', 'kaz', 'kbd', 'kca', 
        'kea', 'ket', 'kgp', 'khb', 'khm', 'kij', 'kin', 'kir', 'kjh', 
        'kmg', 'kmv', 'kor', 'kpv', 'krc', 'kum', 'kut', 'kxo', 'lac', 
        'lao', 'lav', 'led', 'lez', 'lin', 'lit', 'lkt', 'ltz', 'lug', 
        'luo', 'lus', 'lzz', 'mal', 'mar', 'mco', 'mkd', 'mlt', 'mlv', 
        'mnk', 'mns', 'mri', 'mtq', 'mww', 'mya', 'myp', 'mzn', 'nab', 
        'naq', 'nav', 'nch', 'nds', 'nep', 'new', 'nhg', 'nia', 'nio', 
        'niv', 'nld', 'nob', 'ntj', 'oss', 'pac', 'pag', 'pam', 'pan', 
        'pap', 'pau', 'pdt', 'pjt', 'pny', 'pol', 'pon', 'por', 'ppl', 
        'pre', 'rap', 'rif', 'ron', 'run', 'rus', 'sah', 'san', 'sat', 
        'sco', 'sei', 'shn', 'sin', 'slv', 'sna', 'snd', 'som', 'spa', 
        'squ', 'srp', 'stp', 'str', 'sun', 'swe', 'swh', 'tam', 'tat', 
        'tay', 'tel', 'tgl', 'tha', 'tir', 'tli', 'tpi', 'tqw', 'tsi', 
        'tuk', 'tur', 'twf', 'tyv', 'tzm', 'ude', 'uig', 'ukr', 'umb', 
        'unk', 'unm', 'urb', 'urd', 'vie', 'vma', 'wiy', 'wol', 'wuu', 
        'xal', 'xho', 'xmf', 'yap', 'yii', 'yor', 'yrk', 'yua', 'yue', 
        'zai', 'zpq', 'zul']
    
def main():
    if not any([opt.preprocess, opt.train, opt.translate]):
        print('Specify at least one action (preprocess, train, test)')
        sys.exit()
    if opt.lang == ['high']:
        lang = HIGH_RESOURCE
    elif opt.lang == ['adapted']:
        lang = ADAPTED
    elif opt.lang == ['typo']:
        lang = TYPO
    else:
        lang = opt.lang
    model = G2PModel(opt.name, train_langs=lang, train_scripts=opt.script,
                        src_features=opt.src_features, tgt_features=opt.tgt_features)
    if opt.preprocess:
        model.preprocess()
    if opt.train:
        model.train(opt.train_config)
    if opt.translate:
        model.translate()
    
if __name__ == '__main__':
    main()
