# -*- coding: utf-8 -*-
#/usr/bin/python2

import os

class Hyperparams:

    # preprocessor.py - DownLoad & Preprocess & Save data to disk

    DATASET = 'MultiUN'  # You can use 'OpenSubTitle2018' dataset if you change DATASET as 'OpenSubTitle2018'
    # data_path: dir for law data (download dir)
    data_path = '/media/disk1/public_milab/translation/DATA/MultiUN_data'
    # save_path: dir for [training / development / vocabulary data] (preprocessed data will be save at this dir)
    save_path = '/media/disk1/public_milab/translation/zeroshot_exp/exp_zeroshot_3rd/multiun_test'
    # languages: Languages that will be used for training and development (for download)
    # you can check possible languages in "http://opus.nlpl.eu/OpenSubtitles2018.php"
    languages = ['ES', 'EN', 'FR']
    # resampling_size: # Maximum size of each parallel language data
    resampling_size = int(5*1e6)
    # dev_from / dev_t0 Development data ( From Language / To Language )
    dev_from = 'FR'
    dev_to = 'EN'
    # dev_size: Define Development data size
    dev_size = int(5*1e5)
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
    zeroshot_dev_output = os.path.join(save_path, 'dev', 'TO')  # development data file (TO)

    batch_size = 256
    num_epochs = 100
    
    # train.py
    # train path : path for saving event / graph ...
    train_path = os.path.join(save_path, 'train')
    num_units = 256
    num_blocks = 6  # number of encoder/decoder blocks
    num_heads = 8
    dropout_rate = 0.2
    warmup_step = 10000

    summary_every_n_step = 300
    save_every_n_step = 3000
    evaluate_every_n_step = 1000

hp = Hyperparams()

