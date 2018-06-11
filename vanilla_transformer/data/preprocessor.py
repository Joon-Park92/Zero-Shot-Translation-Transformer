# -*- coding: utf-8 -*-
#/usr/bin/python2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import regex as re
import codecs
import os
from collections import Counter
import pandas as pd
from utils.hyperparams import hp


class Preprocesser(object):

    def __init__(self, extract_path, save_path, lang_from, lang_to, dev_size, min_len, max_len):

        self.extract_path = extract_path
        self.save_path = save_path
        self.lang_from = str(lang_from)
        self.lang_to = str(lang_to)
        self.max_len = max_len
        self.min_len = min_len
        self.dev_size = dev_size
        self.train_df = None
        self.dev_df = None

    @staticmethod
    def _preprocess(text):
        pre_text = re.sub("[\p{P}]+", "", text)
        pre_text = re.sub("[ ]{2,}", " ", pre_text)
        pre_text = re.sub("^[ ]", "", pre_text)
        return pre_text

    def _df_prerpocess(self, df):
        for col in df.columns:
            df[col] = df[col].apply(lambda x: self._preprocess(x))

    def preprocess(self):
        print("PREPROCESS & SPLIT DATA...")

        save_path = os.path.join(self.save_path)
        file_list = os.listdir(self.extract_path)

        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        dic = {}
        for lang in file_list:
            with codecs.open(os.path.join(self.extract_path, lang), 'r', encoding='utf-8') as f:
                data = f.read().splitlines()
            dic[lang] = data

        df = pd.DataFrame(dic)
        self._df_prerpocess(df=df)

        # filter too short / long
        columns = df.columns
        idx_1 = df[columns[0]].apply(lambda x: len(x.split())) >= self.min_len
        idx_2 = df[columns[0]].apply(lambda x: len(x.split())) <= (self.max_len - 2)  # for (</S>)
        idx_3 = df[columns[1]].apply(lambda x: len(x.split())) >= self.min_len
        idx_4 = df[columns[1]].apply(lambda x: len(x.split())) <= (self.max_len - 2)  # for (<S> / </S>)
        df = df[idx_1 & idx_2 & idx_3 & idx_4]

        # shuffle
        df = df.sample(frac=1.0)
        df.index = range(len(df))

        # split train / dev
        train_df = df.iloc[self.dev_size:]
        train_df = train_df.drop_duplicates()
        train_df = train_df[(train_df[self.lang_from] != "") & (train_df[self.lang_to] != "")]
        dev_df = df.iloc[:self.dev_size]
        dev_df = dev_df.drop_duplicates()
        dev_df = dev_df[(dev_df[self.lang_from] != "") & (dev_df[self.lang_to] != "")]

        self.train_df = train_df
        self.dev_df = dev_df

    def write_files(self):

        path = self.save_path
        train_path = os.path.join(path, 'train')
        dev_path = os.path.join(path, 'dev')

        print("WRITE TRAINING DATA...")
        if not os.path.isdir(path):
            os.mkdir(path)
        if not os.path.isdir(train_path):
            os.mkdir(train_path)
        with codecs.open(os.path.join(train_path, 'FROM'), 'w', encoding='utf-8') as f:
            for line in self.train_df[self.lang_from]:
                f.write(line + '\n')
        with codecs.open(os.path.join(train_path, 'TO'), 'w', encoding='utf-8') as f:
            for line in self.train_df[self.lang_to]:
                f.write(line + '\n')

        print("WRITE DEV DATA...")
        if not os.path.isdir(dev_path):
            os.mkdir(dev_path)
        with codecs.open(os.path.join(dev_path, 'FROM'), 'w', encoding='utf-8') as f:
            for line in self.dev_df[self.lang_from]:
                f.write(line + '\n')
        with codecs.open(os.path.join(dev_path, 'TO'), 'w', encoding='utf-8') as f:
            for line in self.dev_df[self.lang_to]:
                f.write(line + '\n')

    def write_vocab(self):
        print("MAKE & WRITE VOCAB FILES...")

        path = os.path.join(self.save_path, 'vocab')
        if not os.path.exists(path): os.mkdir(path)

        for lang in [self.lang_from, self.lang_to]:
            count = Counter([word for sentence in self.train_df[lang] for word in sentence.split()])
            file_name = os.path.join(path, 'vocab.' + lang.lower())
            with codecs.open(file_name, 'w', encoding='utf-8') as f:
                for word, cnt in count.most_common():
                    f.write('\t'.join([word, str(cnt)]) + '\n')


if __name__ == '__main__':

    # Preprocess & Save data
    preprocessor = Preprocesser(extract_path=hp.extract_path,
                                save_path=hp.save_path,
                                lang_from=hp.FROM,
                                lang_to=hp.TO,
                                dev_size=hp.dev_size,
                                min_len=hp.min_len,
                                max_len=hp.max_len)
    preprocessor.preprocess()
    preprocessor.write_files()
    preprocessor.write_vocab()

    # Print information about size of data
    print('SIZE OF DATA: ')
    print('TRAINING: {} '.format(len(preprocessor.train_df)))
    print('DEVELOPMENT: {} '.format(len(preprocessor.dev_df)))
    print("PREPROCESS IS DONE...")

