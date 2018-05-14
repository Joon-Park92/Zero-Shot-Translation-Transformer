# -*- coding: utf-8 -*-
#/usr/bin/python2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import regex as re
import codecs
import os
import gzip
import zipfile

from collections import Counter
import pandas as pd
import itertools
import tqdm

from os.path import isfile, isdir
from hyperparams import hp

#Python2
from urllib import urlretrieve
# Python 3
# from urllib.request import urlretrieve


class DLProgress(tqdm.tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


class DownLoader(object):
    """
    Class used for downloading Multi UN dataset

    Args:
        path: str, download directory
        languages: list, languages which will be downloaded
    """

    def __init__(self, downlaod_path, languages):
        self.path = downlaod_path
        self.languages = languages
        self.get_url()

    def get_url(self):

        base_url = 'http://opus.nlpl.eu/download.php?f=MultiUN/'
        end_url = '.txt.zip'
        mk_url = lambda langs: base_url + langs + end_url

        self.url_list = []
        for lang1, lang2 in itertools.combinations(self.languages, 2):
            lang = [lang1.lower(), lang2.lower()]
            lang.sort()
            id_ = "-".join(lang)
            self.url_list.append(mk_url(id_))

    def _download_inner(self, url):

        sub_dir = url[url.find('UN') + 3: -8]
        file_name = sub_dir + '.txt.zip'

        if not isdir(os.path.join(self.path, sub_dir)):
            os.mkdir(os.path.join(self.path, sub_dir))

        if not isfile(os.path.join(self.path, sub_dir, file_name)):
            with DLProgress(unit='B', unit_scale=True, miniters=1, desc=file_name) as pbar:
                urlretrieve(url, os.path.join(self.path, sub_dir, file_name), pbar.hook)

    def download(self):
        for url in self.url_list:
            self._download_inner(url)


class DataExtractor(object):

    def __init__(self, download_path, extract_path, lang_from, lang_to):
        self.download_path = download_path
        self.extract_path = extract_path
        self.lang_from = lang_from
        self.lang_to = lang_to

    def _get_key(self):
        lang1 = self.lang_from
        lang2 = self.lang_to
        lang = [lang1.lower(), lang2.lower()]
        lang.sort()
        key = "-".join(lang)
        return key

    def _get_df(self):

        key = self._get_key()
        file_name = key + '.txt.zip'
        full_path = os.path.join(self.download_path, key, file_name)

        with zipfile.ZipFile(full_path) as f:
            namelist = f.namelist()
            df = pd.DataFrame({name: f.read(namelist[i]).split('\n') for i, name in enumerate(namelist)})

        if len(df.columns) >= 3:
            df.drop(df.columns[2], axis=1, inplace=True)

        lang1 = str(key).upper()[:2]
        lang2 = str(key).upper()[3:]
        df.columns = [lang1, lang2]

        return df

    def extract_file(self):
        save_path = self.extract_path
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        if not (os.path.isfile(os.path.join(save_path, self.lang_from)) &
                os.path.isfile(os.path.join(save_path, self.lang_to))):
            df = self._get_df()
            for lang in df.columns:
                with codecs.open(os.path.join(save_path, lang), 'w', encoding='utf-8') as f:
                    for line in df[lang]:
                        f.write(line.decode('utf-8') + '\n')

        print('EXTRACTED...!')


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

    # Download data from OpenSubtitle2018
    downloader = DownLoader(downlaod_path=hp.download_path, languages=hp.languages)
    downloader.download()
    print('Download is Done...!')

    # Extract download files
    extractor = DataExtractor(download_path=hp.download_path,
                              extract_path=hp.extract_path,
                              lang_from=hp.FROM,
                              lang_to=hp.TO)
    extractor.extract_file()

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

