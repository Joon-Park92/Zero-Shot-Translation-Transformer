from __future__ import print_function
from __future__ import division
from hyperparams import Hyperparams as hp

import tensorflow as tf
import codecs
import os 


def get_each_vocab(vocab_path):
    """
    Arg:
        vocab_path : path containing vocab files.
    
    Return:
        vocab_dict : A dictionary that contains lists of words (count > mininmun_count) for each language.
    """
    vocab_files = os.listdir(vocab_path)
    vocab_dict ={}

    def _get_lang_from_file_name(file_name):
        lang = file_name[-2:]  # assume that file format is like (vocab.en / vocab.ko last two char for lang)
        return lang.upper()

    for i in range(len(vocab_files)):
        file_name = vocab_files[i]
        lang = _get_lang_from_file_name(file_name)
        full_path = os.path.join(vocab_path, file_name)
        with codecs.open(full_path, 'r', 'utf-8') as f:
            data = f.read().splitlines()
            vocab_ = [line.split('\t')[0] for line in data if int(line.split('\t')[1])>=minimum_count]    
            vocab_dict[lang] = vocab_
            
    return vocab_dict


def get_zeroshot_vocab(vocab_dict):
    """
    Args:
        vocab_dict: A dictionary that contains lists of words (count > mininmun_count) for each language.
    Return:
        zeroshot_vocab2int: A dictionary maps from vocab used in zeroshot to integers(idx).
        zeroshot_int2vocab: A dictionary maps from integers(idx) to vocab used in zeroshot.
        
    """
    keys = vocab_dict.keys()
    make_token = lambda token : '<2' + str(token) + '>'
    
    zeroshot_vocab = list({vocab for key in keys for vocab in vocab_dict[key]})
    zeroshot_vocab += [make_token(key) for key in keys]
    zeroshot_vocab += ['</S>', '<S>', '<UNK>', '<PAD>']
    zeroshot_vocab.reverse()
    
    # <PAD> : 0 / <UNK> : 1 / <S> : 2 / </S> :3 / <2KO> : 4 etc...
    zeroshot_vocab2int = {vocab: idx for idx, vocab in enumerate(zeroshot_vocab)}
    zeroshot_int2vocab = {idx: vocab for idx, vocab in enumerate(zeroshot_vocab)}
    
    return zeroshot_vocab2int, zeroshot_int2vocab


def get_each_lang_idx(vocab_dict, zeroshot_vocab2int):
    """
    Return:
        A dictionary that contains zeroshot vocab idx for each language (FOR DECODING)
    """   
    lang_idx_dict = {}
    languages = vocab_dict.keys()    
    for lang in languages:
        lang_idx_dict[lang] = [zeroshot_vocab2int[word] for word in vocab_dict[lang]]
        
    return lang_idx_dict


def print_vocab_info():
    language = vocab_dict.keys()
    print('The number of words with a minimum count greater than {}...\n'.format(minimum_count))
    for i in range(len(language)):
        print('size of vaocab {} : {}'.format(language[i], len(vocab_dict[language[i]])))
    print ('size of vocab total : {}'.format(sum([len(vocab_dict[lang]) for lang in language])))


def load_vocabs():
    """
    Retrun: zeroshot_vocab2int, zeroshot_int2vcab, lang_idx_dict
    """
    return zeroshot_vocab2int, zeroshot_int2vocab, lang_idx_dict


def load_data_set():
    """
    Return: train_iterator, dev_iterator 
    ( (input_batch, target_batch) / (input_batch, target_batch) )
    """
    return train_iterator, dev_iterator


class InputController(object):
    """
    controller: tensorflow placehodler tensor
        if feed_dict : contoller == True => input tensor extracted from training dataset
        if feed_dict: contoller == False => input tensor extracted from dev dataset


    Notice : Initialize following operations
        1. global_variables_initializer()
        2. tables.initializer()
        3. train_iterator_initializer()
        4. dev_iterator_initializer()

        Example:

        with tf.Session() as sess:
            sess.run([tf.global_variables_initializer(), train_iterator.initializer,
            dev_iterator.initializer, tf.tables_initializer()])

    """

    def __init__(self, train_iterator, dev_iterator):
        self._controller = tf.placeholder(tf.bool)
        self.train_iterator = train_iterator
        self.dev_iterator = dev_iterator

    def get_input(self):
        return (tf.cond(pred=self.controller,
                        true_fn=lambda: self.train_iterator.get_next(),
                        false_fn=lambda: self.dev_iterator.get_next()))

    @property
    def train_initializer(self):
        return self.train_iterator.initializer

    @property
    def dev_initializer(self):
        return self.dev_iterator.initializer

    @property
    def controller(self):
        return self._controller


# Input tensor for Training / Evaluation ( Developement )
def get_dataset(mode, input_hash_table, target_hash_table):
    """
    Args:
        mode: 'train' / 'dev'
        hash_table : tf.contrib.lookup.HashTable() object, mapping from string to intger(index)

    Returns:
        tf.data.Dataset Iterator object
    """
    assert mode == 'train' or mode == 'dev'

    input_dataset = tf.data.TextLineDataset([hp.zeroshot_train_input if mode == 'train' else hp.zeroshot_dev_input])
    output_dataset = tf.data.TextLineDataset([hp.zeroshot_train_output if mode == 'train' else hp.zeroshot_dev_output])
    output_dataset = output_dataset.map(lambda string: string + ' </S>')
    dataset = tf.data.Dataset.zip((input_dataset, output_dataset))
    dataset = dataset.map(lambda string_in, string_out: (tf.string_split([string_in]).values[:hp.maxlen],
                                                         tf.string_split([string_out]).values[:hp.maxlen]))
    dataset = dataset.map(lambda words_in, words_out: \
                              (input_hash_table.lookup(words_in), target_hash_table.lookup(words_out)))
    dataset = dataset.padded_batch(batch_size=hp.batch_size,
                                   padded_shapes=(tf.TensorShape([hp.maxlen]), tf.TensorShape([hp.maxlen])))

    # dev deataset => infinetly iterative.
    if mode == 'dev':
        dataset = dataset.repeat()

    elif mode == 'train':
        dataset = dataset.repeat(hp.num_epochs)

    iterator = dataset.make_initializable_iterator()

    return iterator

## Args
vocab_path = hp.vocab_path
minimum_count = hp.min_count

# Make zeroshot vocabulary / vocabulary of each language
vocab_dict = get_each_vocab(vocab_path)
zeroshot_vocab2int, zeroshot_int2vocab = get_zeroshot_vocab(vocab_dict)
lang_idx_dict = get_each_lang_idx(vocab_dict, zeroshot_vocab2int)

# Make hash_table
hash_key = tf.convert_to_tensor([key.encode('utf-8') for key in zeroshot_vocab2int.keys()], tf.string)
hash_value = tf.convert_to_tensor([zeroshot_vocab2int.get(key) for key in zeroshot_vocab2int.keys()], tf.int32)
table = tf.contrib.lookup.HashTable(tf.contrib.lookup.KeyValueTensorInitializer(hash_key, hash_value), zeroshot_vocab2int['<UNK>'])

# train / dev dataset iterator
train_iterator = get_dataset('train', table, table)
dev_iterator = get_dataset('dev', table, table)
eval_iterator = get_dataset('eval', table, table)

# print_vocab_info()



