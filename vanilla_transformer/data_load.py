# -*- coding: utf-8 -*-
#/usr/bin/python2

from __future__ import print_function
from __future__ import division
from hyperparams import Hyperparams as hp

import tensorflow as tf
import codecs
import os 


class VocabMaker(object):

    def __init__(self, vocab_path, minimum_count, lang_from, lang_to):
        self.vocab_path = vocab_path
        self.minimum_count = minimum_count
        self.lang_from = lang_from
        self.lang_to = lang_to

        # will be updated by followed functions
        self.vocab_dic = None
        self.from_int2voca = None
        self.from_voca2int = None
        self.to_int2voca = None
        self.to_voca2int = None

        self._get_each_vocab()  # update self.vocab_dic
        self._get_vocab()  # update voca2int / int2voca
        self.print_vocab_info()  # print vocab information

    def _get_each_vocab(self):
        """
        Arg:
            vocab_path : path containing vocab files.

        Return:
            vocab_dict : A dictionary that contains lists of words (count > mininmun_count) for each language.
        """
        vocab_path = self.vocab_path
        vocab_files = os.listdir(vocab_path)
        vocab_dict ={}

        def _get_lang_from_file_name(file_name):
            lang = file_name[-2:]  # assume that file format is like (vocab.en / vocab.ko last two char represent lang)
            return lang.upper()

        for i in range(len(vocab_files)):
            file_name = vocab_files[i]
            lang = _get_lang_from_file_name(file_name)
            full_path = os.path.join(vocab_path, file_name)
            with codecs.open(full_path, 'r', 'utf-8') as f:
                data = f.read().splitlines()
                vocab_ = [line.split('\t')[0] for line in data if int(line.split('\t')[1])>=self.minimum_count]
                vocab_dict[lang] = vocab_

        self.vocab_dic = vocab_dict

    def _get_vocab(self):

        for key in self.vocab_dic.keys():

            if key == self.lang_from:
                voca = self.vocab_dic[key]
                voca += ['<UNK>', '<PAD>']
                # to make <PAD> to be 0 / <UNK> 1 /
                voca.reverse()

                self.from_voca2int = {i : word for i, word in enumerate(voca)}
                self.from_int2voca = {word: i for i, word in enumerate(voca)}

            else:
                voca = self.vocab_dic[key]
                voca += ['</S>', '<S>', '<UNK>', '<PAD>']
                voca.reverse()

                self.to_voca2int = {i: word for i, word in enumerate(voca)}
                self.to_int2voca = {word: i for i, word in enumerate(voca)}

    def print_vocab_info(self):
        print('\nThe number of words with a minimum count greater than {}...\n'.format(self.minimum_count))
        for lang in self.vocab_dic.keys():
            print("{} : {}".format(lang, self.vocab_dic[lang]))


class TFDataSetMaker(object):

    def __init__(self, from_voca2int, from_int2voca, to_voca2int, to_int2voca, is_training):

        self.from_voca2int = from_voca2int
        self.from_int2voca = from_int2voca
        self.to_voca2int = to_voca2int
        self.to_int2voca = to_int2voca
        self.is_training = is_training

        self.train_iterator = None
        self.dev_iterator = None
        self._make_datset()  # update self.train / self.dev iterator

    @staticmethod
    def _get_dataset(mode, input_hash_table, target_hash_table):

        assert mode == 'train' or mode == 'dev'

        input_dataset = tf.data.TextLineDataset([hp.train_input if mode == 'train' else hp.dev_input])
        output_dataset = tf.data.TextLineDataset([hp.train_output if mode == 'train' else hp.dev_output])
        output_dataset = output_dataset.map(lambda string: '<S> ' + string + ' </S>')
        dataset = tf.data.Dataset.zip((input_dataset, output_dataset))

        dataset = dataset.map(lambda string_in, string_out: (tf.string_split([string_in]).values,
                                                             tf.string_split([string_out]).values))
        dataset = dataset.map(lambda words_in, words_out:
                              (input_hash_table.lookup(words_in), target_hash_table.lookup(words_out)))

        dataset = dataset.padded_batch(batch_size=hp.batch_size,
                                       padded_shapes=(tf.TensorShape([hp.max_len]), tf.TensorShape([hp.max_len])))

        if mode == 'dev':
            dataset = dataset.repeat()

        elif mode == 'train':
            dataset = dataset.repeat(hp.num_epochs)

        iterator = dataset.make_one_shot_iterator()

        return iterator

    def _make_datset(self):

        # Make hash_table
        def _get_hash(voca2int):
            hash_key = tf.convert_to_tensor([key.encode('utf-8') for key in voca2int.keys()], tf.string)
            hash_value = tf.convert_to_tensor([voca2int.get(key) for key in voca2int.keys()], tf.int32)
            table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(hash_key, hash_value),
                                                voca2int['<UNK>'])
            return table

        # train / dev data_set iterator
        input_hash_table = _get_hash(voca2int=self.from_voca2int)
        target_hash_table = _get_hash(voca2int=self.to_voca2int)

        self.train_iterator = self._get_dataset(mode='train',
                                                input_hash_table=input_hash_table,
                                                target_hash_table=target_hash_table)

        self.dev_iterator = self._get_dataset(mode='dev',
                                              input_hash_table=input_hash_table,
                                              target_hash_table=target_hash_table)

    def get_input_tensor(self):
        return (tf.cond(pred=self.is_training,
                        true_fn=lambda: self.train_iterator.get_next(),
                        false_fn=lambda: self.dev_iterator.get_next()))

