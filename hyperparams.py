# -*- coding: utf-8 -*-
#/usr/bin/python2

import os

class Hyperparams:

    # preprocessor.py
    data_path = '/media/disk1/public_milab/translation/DATA/OpenSubtitle2018'  # dir for law data (download dir)
    save_path = '/media/disk1/public_milab/translation/transformer/\
    transformer_base_1.0/train_data'  # dir for training / development / vocabulary data (preprocessed)

    # you can check possible languages in "http://opus.nlpl.eu/OpenSubtitles2018.php"
    languages = ['KO', 'JA', 'EN']  # Languages that will be used for training and development (for download)
    resampling_size = 1e6  # Maximum size of each parallel language data
    dev_from = 'JA'  # Development data ( From Language )
    dev_to = 'KO'  # Development data ( To Language )
    dev_size = 1e4  # Define Development data size

    # data_load.py
    vocab_path = os.path.join(save_path, 'vocab')
    
    zeroshot_train_input = '/media/disk1/public_milab/translation/transformer/zhkr_bible/data/train/zeroshot_input'
    zeroshot_train_output = '/media/disk1/public_milab/translation/transformer/zhkr_bible/data/train/zeroshot_target'
    zeroshot_dev_input = '/media/disk1/public_milab/translation/transformer/zhkr_bible/data/dev/zeroshot_input'
    zeroshot_dev_output = '/media/disk1/public_milab/translation/transformer/zhkr_bible/data/dev/zeroshot_target'
        
    # training
    batch_size = 256   # alias = N
    
    # model
    maxlen = 50 # Maximum number of words in a sentence. alias = T.                
    min_count = 0  # words whose occurred less than min_cnt are encoded as <UNK>.
    hidden_units = 256 # alias = C
    num_blocks = 3 # number of encoder/decoder blocks
    num_epochs = 100
    num_heads = 8
    dropout_rate = 0.2
    sinusoid = False  # If True, use sinusoid. If false, positional embedding.
        
    #config
    target_designated = True
    zeroshot_target_lang = 'KO'
    summary_every_n_step = 100
    save_every_n_step = 500
    evaluate_every_n_step = 300
    max_to_keep= 3
    logdir = '/media/disk1/public_milab/translation/transformer/zhkr_bible/log_kor_designate'
    # log directory => summary / ckpt / result text file 

hp = Hyperparams()
    
