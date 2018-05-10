# -*- coding: utf-8 -*-
#/usr/bin/python2

from __future__ import print_function
from __future__ import division
from hyperparams import Hyperparams as hp

import tensorflow as tf
import numpy as np
import os, codecs
from tqdm import tqdm

from model_layers import *
from utils import *


class InputController(object):
    """
    controller: tensorflow placehodler tensor
        if feed_dict : contoller == True => input tensor extracted from training dataset
        if feed_dict: contoller == False => input tensor extracted from dev dataset
    
    """
    def __init__ (self, train_iterator, dev_iterator):
        
        self._controller = tf.placeholder(tf.bool)
        self.train_iterator = train_iterator
        self.dev_iterator = dev_iterator
                
    def get_input(self):
        return(tf.cond(pred=self.controller,
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
    

class Transformer(object):
    
    def __init__(self, x, y, input_int2vocab, target_int2vocab, num_units, num_blocks, num_head, drop_rate, is_training):
        """
        Args:
            x : input 2-D tensor [ N, T ]
            y : target 2-D tensor [ N, T ]
            input_int2vocab : dictionary map integer to vocab in input space
            target_int2vocab : dictionary map integer to vocab in target space
            is_training : control_tensor(tf.bool type) used for switching training / validation data
        """
        self.graph = tf.Graph()

        self.x = x
        self.y = y
        self.decoder_inputs = tf.concat((tf.ones_like(self.y[:, :1]) * 2, self.y[:, :-1]), -1)

        self._num_units = num_units
        self._num_head = num_head
        self._drop_rate = drop_rate
        self._num_blocks = num_blocks

        self._input_int2vocab = input_int2vocab
        self._target_int2vocab = target_int2vocab
        self._is_training = is_training
        self._build_graph()
                    
    def _build_graph(self):

        with self.graph.as_default():
            with tf.name_scope('ENCODER'):
                # Input Embedding + Positional Encoding
                input_embedding = embedding(ids=self.x,
                                            vocab_size=len(self._input_int2vocab),
                                            embed_dim=self._num_units,
                                            zeropad=True,
                                            pos=True,
                                            scope='enc_embedding',
                                            reuse=False)

                input_embedding = tf.layers.dropout(inputs= input_embedding,
                                                    rate= self._drop_rate,
                                                    training=self._is_training)

                # Encoder Blocks
                for i in range(1, self._num_blocks + 1):
                    input_embedding = encoding_sublayer(input_embedding=input_embedding,
                                                        num_units=self._num_units,
                                                        num_heads=self._num_head,
                                                        drop_rate=self._drop_rate,
                                                        is_training=self._is_training,
                                                        scope='enc_block_{}'.format(i),
                                                        reuse=False)
            with tf.name_scope('DECODER'):
                output_embedding = embedding(ids=self.decoder_inputs,
                                             vocab_size=len(self._target_int2vocab),
                                             embed_dim=self._num_units,
                                             zeropad=True,
                                             pos=True,
                                             scope='dec_embedding',
                                             reuse=False)

                output_embedding = tf.layers.dropout(inputs=output_embedding,
                                                     rate=self._drop_rate,
                                                     training=self._is_training)

                # Decoding Blocks
                for i in range(1, self._num_blocks + 1):
                    output_embedding = decoding_sublayer(output_embedding=output_embedding,
                                                         input_embedding=input_embedding,
                                                         num_units=self._num_units,
                                                         num_heads=self._num_head,
                                                         drop_rate=self._drop_rate,
                                                         is_training=self._is_training,
                                                         scope='dec_block_{}'.format(i),
                                                         reuse=False)

        # Final linear projection
        self.logits = tf.layers.dense(inputs=output_embedding, units=len(self._target_int2vocab))
        self.preds = tf.to_int32(tf.argmax(self.logits, axis=-1))
        self.istarget = tf.to_float(tf.not_equal(self.y, 0)) ## Padding Problem
        self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.y))*self.istarget)/ (tf.reduce_sum(self.istarget))            
        tf.summary.scalar('acc', self.acc)

        # Loss
        self.y_smoothed = tf.cond(pred=self._is_training,
                                  true_fn= lambda: label_smoother(tf.one_hot(self.y, depth=len(self._target_int2vocab))),
                                  false_fn= lambda: tf.one_hot(self.y, depth=len(self._target_int2vocab)))
        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_smoothed)
        self.mean_loss = tf.reduce_sum(self.loss*self.istarget) / (tf.reduce_sum(self.istarget))

        # Training Scheme
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
        self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)

        # Summary 
        tf.summary.scalar('mean_loss', self.mean_loss)
        self.merged = tf.summary.merge_all()

        self.saver = tf.train.Saver()

    def save(self, sess):
        self.saver.save(sess, os.path.join(hp.logdir, ''), self.global_step)
        print("Graph is Saved")
    
    def load(self, sess):
        self.saver.restore(sess, tf.train.latest_checkpoint('/media/disk1/public_milab/translation/transformer/zhkr_bible/log/'))
        print("Graph is Loaded from ckpt")
        
    def evaluation(self, sess, dev_dataset_iterator):
        """
        Args:
            sess: tensorflow Session Object            
        """    
        self.load(sess)
        path = os.path.join(hp.logdir, 'result')
        if not os.path.isdir(path): os.mkdir(path)
        
        f_input = codecs.open(os.path.join(path, 'input.txt'),'w', 'utf-8')
        f_preds = codecs.open(os.path.join(path, 'pred.txt'),'w', 'utf-8')
        f_target = codecs.open(os.path.join(path, 'target.txt'), 'w', 'utf-8')

        reference = []
        translation = []

        while 1:
            try:                
                inputs, targets = sess.run(dev_dataset_iterator.get_next())            
                preds = np.zeros((hp.batch_size, hp.maxlen), np.int32)
                for j in tqdm(range(hp.maxlen)):                    
                    _preds = sess.run(self.logits, feed_dict={self.x: inputs, self.y: preds})
                    _preds = arg_max_target_lang(_preds)
                    preds[:, j] = _preds[:, j]

                for input_, pred, target in zip(inputs, preds, targets):
                    f_input.write((" ".join([self._input_int2vocab.get(idx) for idx in input_])).split('<PAD>')[0] + '\n')
                    f_preds.write((" ".join([self._target_int2vocab.get(idx) for idx in pred])).split('</S>')[0] + '\n')
                    f_target.write((" ".join([self._target_int2vocab.get(idx) for idx in target])).split('</S>')[0] + '\n')

                    reference.append((" ".join([self._target_int2vocab.get(idx) for idx in target])).split('</S>')[0])
                    translation.append((" ".join([self._target_int2vocab.get(idx) for idx in pred])).split('</S>')[0])

            except tf.errors.OutOfRangeError:
                break

        print('BLEU : {}'.format(compute_bleu(reference_corpus= reference, translation_corpus=translation)))
        f_input.close()
        f_preds.close()
        f_target.close()
