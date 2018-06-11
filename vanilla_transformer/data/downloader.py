# -*- coding: utf-8 -*-
#/usr/bin/python2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import codecs
import zipfile
import itertools
import tqdm
import pandas as pd
from os.path import isfile, isdir
from utils.hyperparams import hp

# Python2
if sys.version_info[0] == 2:
    from urllib import urlretrieve
# Python 3
else:
    from urllib.request import urlretrieve


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
