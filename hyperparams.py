# -*- coding: utf-8 -*-
#/usr/bin/python2

class Hyperparams:
    '''Hyperparameters'''
    # data
    # vocab_path: containing each vocab of each languages ( gen by preprocess.py )
    vocab_path = '/media/disk1/public_milab/translation/transformer/zhkr_bible/data/vocab' 
    
    zeroshot_train_input = '/media/disk1/public_milab/translation/transformer/zhkr_bible/data/train/zeroshot_input'
    zeroshot_train_output = '/media/disk1/public_milab/translation/transformer/zhkr_bible/data/train/zeroshot_target'
    zeroshot_dev_input = '/media/disk1/public_milab/translation/transformer/zhkr_bible/data/dev/zeroshot_input'
    zeroshot_dev_output = '/media/disk1/public_milab/translation/transformer/zhkr_bible/data/dev/zeroshot_target'
        
    # training
    batch_size = 256   # alias = N
    lr = 0.0005 # learning rate. In paper, learning rate is adjusted to the global step.    
    
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

    
    
    
