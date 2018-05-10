from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import regex as re
import codecs
import os
import gzip

from collections import Counter
import pandas as pd
import itertools
import tqdm


class DataLoader(object):
    """
    Args:
        path: str, directory that contains data folders (=download path)
        languages: list, language list that will be used for training
    """

    def __init__(self, path, languages):
        self.path = path
        self.languages = languages
        self._make_keys()

    def _make_keys(self):
        self.keys = []
        for lang1, lang2 in itertools.combinations(self.languages, 2):
            lang = [lang1.lower(), lang2.lower()]
            lang.sort()
            id_ = "-".join(lang)
            self.keys.append(id_)

    def _get_dir(self):
        for lang1, lang2 in itertools.combinations(self.languages, 2):
            lang = [lang1.lower(), lang2.lower()]
            lang.sort()
            id_ = "-".join(lang)
            yield id_

    @staticmethod
    def _read_table(path):
        with gzip.open(path) as f:
            data = f.read().split('\n')

        ref_idx = path.find('.gz')
        lang = path[ref_idx - 2: ref_idx].upper()

        return lang, data

    def get_df(self):
        df = {}
        for folder in tqdm.tqdm(self._get_dir(),
                                desc='Load data : ',
                                total=int((len(self.languages) * (len(self.languages) - 1)) / 2)):
            assert os.listdir(os.path.join(self.path, folder))
            temp = {}
            for file_ in os.listdir(os.path.join(self.path, folder)):
                abs_path = os.path.join(self.path, folder, file_)
                key, data = self._read_table(path=abs_path)
                temp[key] = data

            df[folder] = pd.DataFrame(temp)

        return df


class DataSaver(object):

    def __init__(self, df, keys, save_path):
        self.df = df
        self.keys = keys
        self.save_path = save_path

    @staticmethod
    def _preprocess(text):
        pre_text = re.sub("[\p{P}]+", "", text)
        pre_text = re.sub("[ ]{2,}", " ", pre_text)
        pre_text = re.sub("^[ ]", "", pre_text)
        return pre_text

    @staticmethod
    def _make_token(lang, text):
        token = '<' + lang.upper() + '> '
        text = token + text
        return text

    def get_df(self, dev, dev_size):
        print('MERGE DATA & MAKE TOKENS...')
        dev_from = dev[:2].upper()
        dev_to = dev[2:].upper()
        dev_key = [dev_from.lower(), dev_to.lower()]
        dev_key.sort()
        dev_key = "-".join(dev_key)
        assert dev_key in self.keys

        self.train_df = pd.DataFrame(columns=['FROM', 'TO'])
        for key in self.keys:
            if key == dev_key: continue
            for i, col in enumerate(DF[key].columns):
                FROM = DF[key][col].apply(lambda text: self._make_token(DF[key].columns[(i + 1) % 2], text))
                FROM = FROM.apply(self._preprocess)
                TO = DF[key][DF[key].columns[(i + 1) % 2]]
                TO = TO.apply(self._preprocess)
                temp = pd.DataFrame({'FROM': FROM, 'TO': TO})
                self.df = self.df.append(temp)

        FROM = DF[dev_key][dev_from].sample(frac=1.0).iloc[:dev_size].apply(lambda text: self._make_token(dev_to, text))
        FROM = FROM.apply(self._preprocess)
        TO = DF[dev_key][dev_to].iloc[FROM.index]
        TO = TO.apply(self._preprocess)
        self.dev_df = pd.DataFrame({'FROM': FROM, 'TO': TO})

        self.train_df = self.train_df.drop_duplicates()
        self.train_df = self.train_df[(self.train_df.FROM != "") | (self.train_df.TO != "")]
        self.dev_df = self.dev_df.drop_duplicates()
        self.dev_df = self.dev_df[(self.dev_df.FROM != "") | (self.dev_df.TO != "")]

    def write_df(self):
        path = self.save_path
        train_path = os.path.join(path, 'train')
        dev_path = os.path.join(path, 'dev')

        print("Writing TRAINING DATA...")
        if not os.path.isdir(train_path):
            os.mkdir(train_path)
        with codecs.open(os.path.join(train_path, 'FROM'), 'w', encoding='utf-8') as f:
            for line in self.df.FROM:
                f.write(line.decode('utf-8') + '\n')
        with codecs.open(os.path.join(train_path, 'TO'), 'w', encoding='utf-8') as f:
            for line in self.df.TO:
                f.write(line.decode('utf-8') + '\n')

        print("WRITING DEV DATA...")
        if not os.path.isdir(dev_path):
            os.mkdir(dev_path)
        with codecs.open(os.path.join(dev_path, 'FROM'), 'w', encoding='utf-8') as f:
            for line in self.dev_df.FROM:
                f.write(line.decode('utf-8') + '\n')
        with codecs.open(os.path.join(dev_path, 'TO'), 'w', encoding='utf-8') as f:
            for line in self.dev_df.TO:
                f.write(line.decode('utf-8') + '\n')

    def write_vocab(self):
        print("WRITING VOCAB FILES...")
        langlist = self.keys

        path = os.path.join(self.save_path, 'vocab')
        if not os.path.exists(path): os.mkdir(path)

        vocab = {}
        for i in range(len(langlist)):
            vocab[langlist[i]] = Counter([word for sentence in df_in[langlist[i]] for word in sentence.split()])
            file_name = os.path.join(path, 'vocab.' + langlist[i].lower())
            print(file_name)
            count = vocab[langlist[i]]

            with codecs.open(file_name, 'w', encoding='utf-8') as f:
                for word, cnt in count.most_common():
                    f.write('\t'.join([word.decode('utf-8'), str(cnt)]) + '\n')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--data_path', type=str, default='/media/disk1/public_milab/translation/DATA/OpenSubtitle2018',
                        help='Help Messages', required=True)
    parser.add_argument('--languages', action='append',
                        help='Help Messages', required=True)
    parser.add_argument('--save_path', type=str,
                        default='/media/disk1/public_milab/translation/\
                        zeroshot_exp/exp_zeroshot_2nd/zeroshot_SUBTITLE2018_jako/data',
                        help='Help Messages')
    parser.add_argument('--lang_to', type=str, help='Help Messages')
    args = parser.parse_args()

    DATA_PATH = args.data_path
    LANGUAGES = ['KO', 'JA', 'EN', 'ES']
    MAX_DATA = 1000000
    SAVE_PATH = args.save_path
    DEV_SIZE = 10000

    Loader = DataLoader(path=DATA_PATH, languages=LANGUAGES)
    DF = Loader.get_df()

    # To be reviesed
    DF['en-es'] = DF['en-es'].sample(MAX_DATA)
    DF['es-ja'] = DF['es-ja'].sample(MAX_DATA)
    DF['en-ja'] = DF['en-ja'].sample(MAX_DATA)

    # Logger
    for key in Loader.keys:
        print('{} : {}'.format(key, len(DF[key])))

    Saver = DataSaver(DF, Loader.keys, SAVE_PATH)
    Saver.get_df(dev='JAKO', dev_size=DEV_SIZE)
    Saver.write_df()