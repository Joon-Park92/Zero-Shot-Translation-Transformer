# -*- coding: utf-8 -*-
#/usr/bin/python2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import regex as re
import codecs
import os
import gzip

from collections import Counter
import pandas as pd
import itertools
import tqdm

from os.path import isfile, isdir
#Python2
from urllib import urlretrieve
# Python 3
# from urllib.request import urlretrieve
from hyperparams import hp


class DLProgress(tqdm.tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


class DownLoader(object):
    """
    Class used for downloading OpenSubtitles2018 dataset

    Args:
        path: str, download directory
        languages: list, languages which will be downloaded
    """

    def __init__(self, path, languages):
        self.path = path
        self.languages = languages
        self.get_url()

    def get_url(self):
        base = 'http://opus.nlpl.eu/download/OpenSubtitles2018/'
        self.url_list = []
        for lang1, lang2 in itertools.combinations(self.languages, 2):
            lang = [lang1.lower(), lang2.lower()]
            lang.sort()
            id_ = "-".join(lang)

            file_1 = 'c.clean.' + lang[0] + '.gz'
            file_2 = 'c.clean.' + lang[1] + '.gz'
            url_1 = base + id_ + '/' + file_1
            url_2 = base + id_ + '/' + file_2
            self.url_list.append(url_1)
            self.url_list.append(url_2)

    def _download_inner(self, url):
        file_name = url[url.rfind('/') + 1:]

        if not isdir(os.path.join(self.path, url[url.find('2018') + 5:url.rfind('/')])):
            os.mkdir(os.path.join(self.path, url[url.find('2018') + 5:url.rfind('/')]))

        if not isfile(os.path.join(self.path, url[url.find('2018') + 5:url.rfind('/')], file_name)):
            with DLProgress(unit='B', unit_scale=True, miniters=1, desc=file_name) as pbar:
                urlretrieve(url, os.path.join(self.path, url[url.find('2018') + 5:url.rfind('/')], file_name),
                            pbar.hook)

    def download(self):
        for url in self.url_list:
            self._download_inner(url)


class DataLoader(object):
    """
    Load data from disk to RAM
    Make Pandas DataFrame dictionary for each language

    Args:
        path: str, directory that contains data folders (=download path)
        languages: list, language list that will be used for training
    """

    def __init__(self, path, languages):
        self.path = path
        self.languages = languages
        self._make_keys()

    def _make_keys(self):
        # Make key property of class
        self.keys = []
        for lang1, lang2 in itertools.combinations(self.languages, 2):
            lang = [lang1.lower(), lang2.lower()]
            lang.sort()
            id_ = "-".join(lang)
            self.keys.append(id_)

    def _dir_generator(self):
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

    def get_df_dic(self):
        df_dic = {}
        for folder in tqdm.tqdm(self._dir_generator(),
                                desc='Load data : ',
                                total=int((len(self.languages) * (len(self.languages) - 1)) / 2)):
            assert os.listdir(os.path.join(self.path, folder))
            temp = {}
            for file_ in os.listdir(os.path.join(self.path, folder)):
                abs_path = os.path.join(self.path, folder, file_)
                key, data = self._read_table(path=abs_path)
                temp[key] = data

            df_dic[folder] = pd.DataFrame(temp)

        return df_dic


class DataSaver(object):

    def __init__(self, df_dic, keys, save_path):
        self.df_dic = df_dic
        self.train_df = None
        self.dev_df = None
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

    def get_df(self, resampling_size, dev_from, dev_to, dev_size):
        """  Shuffle / Make Token for training( like <2KO> <2EN> <2JA> ...)

        Return:
            None, but update two properties: self.train_df / self.dev_df
            self.train_df = pd.DataFrame object, zero-shot training dataset ( input : <2EN> 안녕하세요 / target: Hello )
            self.dev_df = pd.DataFrame object, evaluation dataset ( inputs : こんにちは / target: 안녕하세요 )
        """
        dev_from = dev_from.upper()
        dev_to = dev_to.upper()

        dev_key = [dev_from.lower(), dev_to.lower()]
        dev_key.sort()
        dev_key = "-".join(dev_key)
        assert dev_key in self.keys, "development language pair ( {} ) is not found".format(dev_key)

        # Resample to reduce data size & accommodate ratio of data
        print("\nRESAMPLING....")
        for key in self.keys:
            if key == dev_key:
                self.df_dic[key] = self.df_dic[key].sample(n=dev_size)
                print("{} pair(DEV) is resampled, size ({})".format(key, len(self.df_dic[key])))
            else:
                if len(self.df_dic[key]) >= resampling_size:
                    print("{} pair is resampled ({}->{}, {})".format(key,
                                                                     len(df_dic[key]),
                                                                     resampling_size,
                                                                     resampling_size / len(df_dic[key])))
                    self.df_dic[key] = self.df_dic[key].sample(n=resampling_size)
                else:
                    print("\nWARNING: Unbalanced language data size")
                    print("{} pair smaller than resample size ({}, Not changed)".format(key, len(self.df_dic[key])))
                    self.df_dic[key] = self.df_dic[key].sample(frac=1.0)

        # Make training data set
        print('\nMERGE & MAKE TOKENS...')
        self.train_df = pd.DataFrame(columns=['FROM', 'TO'])
        for key in self.keys:
            if key == dev_key: continue # make training pair data except for development pair
            for i, col in enumerate(self.df_dic[key].columns):
                FROM_lang = col
                TO_lang = self.df_dic[key].columns[(i + 1) % 2]

                FROM = self.df_dic[key][FROM_lang]
                FROM = FROM.apply(lambda text: self._make_token(self.df_dic[key].columns[(i + 1) % 2], text))
                FROM = FROM.apply(self._preprocess)
                TO = self.df_dic[key][TO_lang]
                TO = TO.apply(self._preprocess)

                mono_pair = pd.DataFrame({'FROM': FROM, 'TO': TO})
                self.train_df = self.train_df.append(mono_pair)

        FROM = self.df_dic[dev_key][dev_from]
        FROM = FROM.apply(lambda text: self._make_token(dev_to, text))
        FROM = FROM.apply(self._preprocess)
        TO = self.df_dic[dev_key][dev_to]
        TO = TO.apply(self._preprocess)
        self.dev_df = pd.DataFrame({'FROM': FROM, 'TO': TO})

        self.train_df = self.train_df.drop_duplicates()
        self.train_df = self.train_df[(self.train_df.FROM != "") | (self.train_df.TO != "")]
        self.train_df = self.train_df.sample(frac=1.0)
        self.dev_df = self.dev_df.drop_duplicates()
        self.dev_df = self.dev_df[(self.dev_df.FROM != "") | (self.dev_df.TO != "")]

    def write_df(self):
        path = self.save_path
        train_path = os.path.join(path, 'train')
        dev_path = os.path.join(path, 'dev')

        print("WRITE TRAINING DATA...")
        if not os.path.isdir(train_path):
            os.mkdir(train_path)
        with codecs.open(os.path.join(train_path, 'FROM'), 'w', encoding='utf-8') as f:
            for line in self.train_df.FROM:
                f.write(line.decode('utf-8') + '\n')
        with codecs.open(os.path.join(train_path, 'TO'), 'w', encoding='utf-8') as f:
            for line in self.train_df.TO:
                f.write(line.decode('utf-8') + '\n')

        print("WRITE DEV DATA...")
        if not os.path.isdir(dev_path):
            os.mkdir(dev_path)
        with codecs.open(os.path.join(dev_path, 'FROM'), 'w', encoding='utf-8') as f:
            for line in self.dev_df.FROM:
                f.write(line.decode('utf-8') + '\n')
        with codecs.open(os.path.join(dev_path, 'TO'), 'w', encoding='utf-8') as f:
            for line in self.dev_df.TO:
                f.write(line.decode('utf-8') + '\n')

    def write_vocab(self, languages):
        print("MAKE & WRITE VOCAB FILES...")

        path = os.path.join(self.save_path, 'vocab')
        if not os.path.exists(path): os.mkdir(path)

        for lang in languages:
            lang = lang.upper()

            temp_df = pd.DataFrame(columns=[lang])
            for key in self.keys:
                for col in self.df_dic[key].columns:
                    if col == lang:
                        temp_df = temp_df.append(self.df_dic[key].xs([col], axis=1))

            count = Counter([word for sentence in temp_df[lang] for word in sentence.split()])
            file_name = os.path.join(path, 'vocab.' + lang.lower())
            with codecs.open(file_name, 'w', encoding='utf-8') as f:
                for word, cnt in count.most_common():
                    f.write('\t'.join([word.decode('utf-8'), str(cnt)]) + '\n')


if __name__ == '__main__':

    # Download data from OpenSubtitle2018
    Downloader = DownLoader(path=hp.data_path, languages=hp.languages)
    Downloader.download()
    print('Download is Done...!')

    # Load law data from hard disk
    Loader = DataLoader(path=hp.data_path, languages=hp.languages)
    df_dic = Loader.get_df_dic()

    # Print information about size of data
    print('SIZE OF LAW DATA: ')
    for key in Loader.keys:
        print('\t{} : {}'.format(key, len(df_dic[key])))

    # Pre-process and write(save) data to hard disk
    Saver = DataSaver(df_dic=df_dic, keys=Loader.keys, save_path=hp.save_path)
    Saver.get_df(resampling_size=hp.resampling_size,
                 dev_from=hp.dev_from,
                 dev_to=hp.dev_to,
                 dev_size=hp.dev_size)
    Saver.write_df()
    Saver.write_vocab(languages=hp.languages)
    print("PREPROCESSING DONE...")

