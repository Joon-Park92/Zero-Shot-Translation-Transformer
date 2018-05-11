import os
import itertools

# Python2
from urllib import urlretrieve
# Python 3
# from urllib.request import urlretrieve

from os.path import isfile, isdir
from tqdm import tqdm


class DLProgress(tqdm):
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

if __name__ == '__main__':
    Downloader = DownLoader(path='/media/disk1/public_milab/tmp', languages=['KO', 'JA', 'EN'])
    Downloader.download()
    print('Download Done...!')
