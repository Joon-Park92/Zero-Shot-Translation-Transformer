# -*- coding: utf-8 -*-
# /usr/bin/python2

import os


class Hyperparams:
    # preprocessor.py - DownLoad & Preprocess & Save data to disk

    # data_path: dir for law data (download dir)
    # save_path: dir for [training / development / vocabulary data] (preprocessed data will be save at this dir)
    #   if you want to translate your own data, make "raw" dir in save_path and
    #       save two parellel data (two file names are abbreviation for each language)
    # languages: Languages that will be used for training and development (for download)
    #       you can check possible languages in "http://opus.nlpl.eu/OpenSubtitles2018.php"
    # FROM/TO: ( Orientation of Translation From Language / To Language )
    # dev_size: Define Development data size
    # max_len: Maximum length of tokenized data

    exp_path = '/media/disk1/public_milab/translation/zeroshot_exp/exp_zeroshot_4rd'

    download_path = '/media/disk1/public_milab/translation/DATA/MultiUN_data'
    extract_path = os.path.join(exp_path, 'raw')
    save_path = os.path.join(exp_path, 'preprocess')

    languages = ['FR', 'EN']  # use capital abbreviation
    FROM = 'FR'
    TO = 'EN'
    dev_size = int(1e4)
    max_len = 35
    min_len = 4

    # data_load.py

    # vocab_path: dir contains vocab files
    # zeroshot_*: preprocessed files
    # minimum_count: vocab having lower frequency than minimum count will be discarded

    vocab_path = os.path.join(save_path, 'vocab')
    train_input = os.path.join(save_path, 'train', 'FROM')  # training data file (FROM)
    train_output = os.path.join(save_path, 'train', 'TO')  # training data file (TO)
    dev_input = os.path.join(save_path, 'dev', 'FROM')  # development data file (FROM)
    dev_output = os.path.join(save_path, 'dev', 'TO')  # development data file (TO)

    minimum_count = 300
    batch_size = 256
    num_epochs = 100

    # train.py
    # train path : path for saving event / graph ...
    train_path = os.path.join(exp_path, 'train')
    num_units = 256
    num_blocks = 6  # number of encoder/decoder blocks
    num_heads = 8
    dropout_rate = 0.1
    warmup_step = 10000

    summary_every_n_step = 1000
    save_every_n_step = 2000
    evaluate_every_n_step = 3000


hp = Hyperparams()

