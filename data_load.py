# -*- coding: utf-8 -*-
#/usr/bin/python2

from __future__ import print_function
from __future__ import division
from hyperparams import Hyperparams as hp

import tensorflow as tf
import codecs
import os 


class ZeroShotVocabMaker(object):

    def __init__(self, vocab_path, minimum_count):
        self.vocab_path = vocab_path
        self.minimum_count = minimum_count
        self.vocab_dic = None
        self.zeroshot_voca2int = None
        self.zeroshot_int2voca = None

        self._get_each_vocab()  # update self.vocab_dic
        self._get_zeroshot_vocab()  # update self.zeroshot voca2int / int2voca
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

    def _get_zeroshot_vocab(self):
        """
        Args:
            vocab_dict: A dictionary that contains lists of words (count > mininmun_count) for each language.
        Return:
            zeroshot_vocab2int: A dictionary maps from vocab used in zeroshot to integers(idx).
            zeroshot_int2vocab: A dictionary maps from integers(idx) to vocab used in zeroshot.

        """
        vocab_dict = self.vocab_dic
        keys = vocab_dict.keys()
        make_token = lambda token : '<2' + str(token) + '>'

        zeroshot_vocab = list({vocab for key in keys for vocab in vocab_dict[key]})
        zeroshot_vocab += [make_token(key) for key in keys]
        zeroshot_vocab += ['</S>', '<S>', '<UNK>', '<PAD>']
        zeroshot_vocab.reverse()

        # <PAD> : 0 / <UNK> : 1 / <S> : 2 / </S> :3 / <2KO> : 4 etc...
        zeroshot_vocab2int = {vocab: idx for idx, vocab in enumerate(zeroshot_vocab)}
        zeroshot_int2vocab = {idx: vocab for idx, vocab in enumerate(zeroshot_vocab)}

        self.zeroshot_voca2int, self.zeroshot_int2voca = zeroshot_vocab2int, zeroshot_int2vocab

    def print_vocab_info(self):
        language = self.vocab_dic.keys()
        print('The number of words with a minimum count greater than {}...\n'.format(self.minimum_count))
        for i in range(len(language)):
            print('size of vaocab {} : {}'.format(language[i], len(self.vocab_dic[language[i]])))
        print('size of vocab total : {}'.format(sum([len(self.vocab_dic[lang]) for lang in language])))


class TFDataSetMaker(object):
    """
    Args:

    controller: tensorflow placehodler tensor
        if feed_dict : contoller == True => input tensor(get_input_tensor) comes from training dataset
        if feed_dict: contoller == False => input tensor(get_input_tensor) extracted from dev dataset
    zeroshot_voca2int: a dictionary mapping vocab to integer
    zeroshot_int2voca: a dictionary mapping integer to vocab


    Notice : Initialize following operations before training
        1. global_variables_initializer()
        2. tables.initializer()
        3. train_iterator_initializer()
        4. dev_iterator_initializer()

        Example:

        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), train_iterator.initializer,
            dev_iterator.initializer, tf.tables_initializer()])

        or

        with tf.Session() as sess:
        sess.run( INSTANCE.get_init_opts() )

    """
    def __init__(self, zeroshot_voca2int, zeroshot_int2voca):
        self.controller = tf.placeholder(tf.bool)
        self.zeroshot_voca2int = zeroshot_voca2int
        self.zeroshot_int2voca = zeroshot_int2voca
        self.train_iterator = None
        self.dev_iterator = None
        self._make_datset()  # update self.train / self.dev iterator

    @staticmethod
    def _get_dataset(mode, input_hash_table, target_hash_table):
        """
        Args:
            mode: 'train' / 'dev'
            input hash_table : tf.contrib.lookup.HashTable() object, mapping from string to intger(index)
            target hash_table : tf.contrib.lookup.HashTable() object, mapping from string to intger(index)

        Returns:
            tf.data.Dataset Iterator object
        """
        assert mode == 'train' or mode == 'dev'

        input_dataset = tf.data.TextLineDataset([hp.zeroshot_train_input if mode == 'train' else hp.zeroshot_dev_input])
        output_dataset = tf.data.TextLineDataset(
            [hp.zeroshot_train_output if mode == 'train' else hp.zeroshot_dev_output])
        output_dataset = output_dataset.map(lambda string: string + ' </S>')
        dataset = tf.data.Dataset.zip((input_dataset, output_dataset))
        dataset = dataset.map(lambda string_in, string_out: (tf.string_split([string_in]).values[:hp.max_len],
                                                             tf.string_split([string_out]).values[:hp.max_len]))
        dataset = dataset.map(lambda words_in, words_out:
                              (input_hash_table.lookup(words_in), target_hash_table.lookup(words_out)))
        dataset = dataset.padded_batch(batch_size=hp.batch_size,
                                       padded_shapes=(tf.TensorShape([hp.max_len]), tf.TensorShape([hp.max_len])))

        if mode == 'dev':
            dataset = dataset.repeat()

        elif mode == 'train':
            dataset = dataset.repeat(hp.num_epochs)

        iterator = dataset.make_initializable_iterator()

        return iterator

    def _make_datset(self):
        zeroshot_vocab2int = self.zeroshot_voca2int

        # Make hash_table
        hash_key = tf.convert_to_tensor([key.encode('utf-8') for key in zeroshot_vocab2int.keys()], tf.string)
        hash_value = tf.convert_to_tensor([zeroshot_vocab2int.get(key) for key in zeroshot_vocab2int.keys()], tf.int32)
        table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(hash_key, hash_value),
                                            zeroshot_vocab2int['<UNK>'])

        # train / dev data_set iterator
        self.train_iterator = self._get_dataset(mode='train', input_hash_table=table, target_hash_table=table)
        self.dev_iterator = self._get_dataset(mode='dev', input_hash_table=table, target_hash_table=table)

    def get_iterator(self):
        return self.train_iterator, self.dev_iterator

    def get_input_tensor(self):
        return (tf.cond(pred=self.controller,
                        true_fn=lambda: self.train_iterator.get_next(),
                        false_fn=lambda: self.dev_iterator.get_next()))

    def get_init_ops(self):
        return [tf.global_variables_initializer(),
                self.train_iterator.initializer,
                self.dev_iterator.initializer,
                tf.tables_initializer()]




