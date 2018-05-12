# -*- coding: utf-8 -*-
#/usr/bin/python2

import os

class Hyperparams:

    # preprocessor.py - DownLoad & Preprocess & Save data to disk

    # data_path: dir for law data (download dir)
    data_path = '/media/disk1/public_milab/translation/DATA/OpenSubtitle2018'
    # save_path: dir for [training / development / vocabulary data] (preprocessed data will be save at this dir)
    save_path = '/media/disk1/public_milab/translation/transformer/transformer_base_1.0/train_data'
    # languages: Languages that will be used for training and development (for download)
    # you can check possible languages in "http://opus.nlpl.eu/OpenSubtitles2018.php"
    languages = ['KO', 'JA', 'EN']
    # resampling_size: # Maximum size of each parallel language data
    resampling_size = int(1e6)
    # dev_from / dev_t0 Development data ( From Language / To Language )
    dev_from = 'JA'
    dev_to = 'KO'
    # dev_size: Define Development data size
    dev_size = int(1e4)
    # max_len: Maximum length of tokenized data
    max_len = 100


    # data_load.py

    # vocab_path: dir contains vocab files
    vocab_path = os.path.join(save_path, 'vocab')
    # minimum_count: Vocab having lower frequency than minimum count will be discarded
    minimum_count = 10
    zeroshot_train_input = os.path.join(save_path, 'train', 'FROM')  # training data file (FROM)
    zeroshot_train_output = os.path.join(save_path, 'train', 'TO')  # training data file (TO)
    zeroshot_dev_input = os.path.join(save_path, 'dev', 'FROM')  # development data file (FROM)
    zeroshot_dev_output = os.path.join(save_path, 'dev', 'FROM')  # development data file (TO)

    batch_size = 256
    num_epochs = 100
    
    # train.py
    # train path : path for saving event / graph ...
    train_path = '/media/disk1/public_milab/translation/transformer/transformer_base_1.0/train/'
    num_units = 256
    num_blocks = 6  # number of encoder/decoder blocks
    num_heads = 8
    dropout_rate = 0.2
    warmup_step = 10000

    summary_every_n_step = 100
    save_every_n_step = 20
    evaluate_every_n_step = 20

hp = Hyperparams()

