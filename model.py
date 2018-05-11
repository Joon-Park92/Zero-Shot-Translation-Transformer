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


class Transformer(object):

    def __init__(self, graph,
                 x,
                 y,
                 input_int2vocab,
                 target_int2vocab,
                 num_units,
                 num_blocks,
                 num_head,
                 drop_rate,
                 is_training,
                 save_dir,
                 warmup_step,
                 max_len,
                 config):
        """
        Args:
            x : input 2-D tensor [ N, T ]
            y : target 2-D tensor [ N, T ]
            input_int2vocab : dictionary map integer to vocab in input space
            target_int2vocab : dictionary map integer to vocab in target space
            is_training : control_tensor(tf.bool type) used for switching training / validation data
        """
        self.graph = graph

        self.x = x
        self.y = y
        # decoder inputs: <S> + shifted right
        self.decoder_inputs = tf.concat((tf.ones_like(self.y[:, :1]) * 2, self.y[:, :-1]), -1)

        self._num_units = num_units
        self._num_head = num_head
        self._drop_rate = drop_rate
        self._num_blocks = num_blocks
        self._warmup_step = warmup_step
        self._max_len = max_len

        self._input_int2vocab = input_int2vocab
        self._target_int2vocab = target_int2vocab
        self._is_training = is_training
        self._build_graph()
        self._save_dir = save_dir

    def __call__(self, inputs, decoder_inputs, drop_rate, is_training):
        return self._transformer_layer(inputs=inputs,
                                       decoder_inputs=decoder_inputs,
                                       drop_rate=drop_rate,
                                       is_training=is_training)

    def _transformer_layer(self,
                           inputs,
                           decoder_inputs,
                           drop_rate,
                           is_training,
                           scope='Transformer',
                           reuse=False):

        with tf.variable_scope(name_or_scope=scope, reuse=reuse):
            with tf.name_scope('ENCODER'):
                # Input Embedding + Positional Encoding
                input_embedding = embedding(ids=inputs,
                                            vocab_size=len(self._input_int2vocab),
                                            embed_dim=self._num_units,
                                            zeropad=True,
                                            pos=True,
                                            scope='enc_embedding',
                                            reuse=False)

                input_embedding = tf.layers.dropout(inputs=input_embedding,
                                                    rate=drop_rate,
                                                    training=is_training)

                # Encoder Blocks
                for i in range(1, self._num_blocks + 1):
                    input_embedding = encoding_sublayer(input_embedding=input_embedding,
                                                        num_units=self._num_units,
                                                        num_heads=self._num_head,
                                                        drop_rate=drop_rate,
                                                        is_training=is_training,
                                                        scope='enc_block_{}'.format(i),
                                                        reuse=False)
            with tf.name_scope('DECODER'):
                output_embedding = embedding(ids=decoder_inputs,
                                             vocab_size=len(self._target_int2vocab),
                                             embed_dim=self._num_units,
                                             zeropad=True,
                                             pos=True,
                                             scope='dec_embedding',
                                             reuse=False)

                output_embedding = tf.layers.dropout(inputs=output_embedding,
                                                     rate=drop_rate,
                                                     training=is_training)

                # Decoding Blocks
                for i in range(1, self._num_blocks + 1):
                    output_embedding = decoding_sublayer(output_embedding=output_embedding,
                                                         input_embedding=input_embedding,
                                                         num_units=self._num_units,
                                                         num_heads=self._num_head,
                                                         drop_rate=drop_rate,
                                                         is_training=is_training,
                                                         scope='dec_block_{}'.format(i),
                                                         reuse=False)

            # Final linear projection
            with tf.name_scope('FINAL_DENSE'):
                logits = tf.layers.dense(inputs=output_embedding, units=len(self._target_int2vocab))

        return logits

    def _build_graph(self):

        with self.graph.as_default():
            self.logits = self._transformer_layer(inputs=self.x,
                                                  decoder_inputs=self.decoder_inputs,
                                                  drop_rate=self._drop_rate,
                                                  is_training=self._is_training)

            self.preds = tf.to_int32(tf.argmax(self.logits, axis=-1))

            # Accuracy: Remove <PAD> Characters
            is_target = tf.to_float(tf.not_equal(self.y, 0))
            total_num = tf.reduce_sum(is_target)
            correct_num = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.y)) * is_target)
            self.acc = total_num / correct_num
            tf.summary.scalar('accuracy', self.acc)

            # Loss: Remove <PAD> Characters
            self.y_smoothed = tf.cond(pred=self._is_training,
                                      true_fn=lambda: label_smoother(tf.one_hot(self.y, depth=len(self._target_int2vocab))),
                                      false_fn=lambda: tf.one_hot(self.y, depth=len(self._target_int2vocab)))
            self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_smoothed)
            self.mean_loss = tf.reduce_sum(self.loss*is_target, axis=-1) / (tf.reduce_sum(is_target))

            # Training Scheme
            self.global_step = tf.get_variable(name='global_step',
                                               shape=[],
                                               dtype=tf.int32,
                                               initializer=tf.constant_initializer(value=1, dtype=tf.int32),
                                               trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=warmup_learning_rate(d_model=self._num_units,
                                                                                       step_num=self.global_step,
                                                                                       warmup_step=self._warmup_step),
                                                    beta1=0.9, beta2=0.98, epsilon=1e-9)
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

    def translate(self, sess, enc_outputs, is_training):
        """Greedy decoding translation
        Args:
            sess:
            enc_outputs:
            is_training:

        Return:
            translated tensor ( decode auto-regressively )
        """

        self.load(sess)
        max_len = self._max_len
        func = lambda x1, x2: self.__call__(inputs=x1,
                                            decoder_inputs=x2,
                                            drop_rate=self._drop_rate,
                                            is_training=is_training)

        def _cond(i, *args):
            return tf.less(i, max_len)

        def _body(i, enc_outputs, decoder_inputs):
            i = tf.add(i, 1)
            logits = func(enc_outputs, decoder_inputs)
            next_decoder_inputs = tf.to_int32(tf.argmax(logits, axis=-1))
            # Masking (causality)
            mask = tf.zeros_like(next_decoder_inputs)
            next_decoder_inputs = tf.concat([next_decoder_inputs[:, :, :i], mask[:, :, i:]], axis=-1)

            return i, enc_outputs, next_decoder_inputs

        # TODO: CHECK THIS PART
        init_dec_input = None
        loop_vars = [tf.constant(0), enc_outputs, init_dec_input]
        _, _, translated_tensor = tf.while_loop(_cond, _body, loop_vars)

        return translated_tensor

    def evaluate(self, sess, dev_dataset_iterator):
        """
        Translate development data and save translated data
        Args:
            sess: tensorflow Session Object
            iterator: tf.Dataset.iterator Object
        """
        path = os.path.join(hp.logdir, 'result')
        if not os.path.isdir(path): os.mkdir(path)

        f_result = codecs.open(os.path.join(path, 'result.txt'), 'w', 'utf-8')
        f_input = codecs.open(os.path.join(path, 'input.txt'), 'w', 'utf-8')
        f_preds = codecs.open(os.path.join(path, 'pred.txt'), 'w', 'utf-8')
        f_target = codecs.open(os.path.join(path, 'target.txt'), 'w', 'utf-8')

        reference = []
        translation = []

        while 1:
            try:
                inputs, targets = sess.run(dev_dataset_iterator.get_next())
                predictions = sess.run(self.translate(sess=sess, enc_outputs=inputs, is_training=False))

                for input_, pred, target in zip(inputs, predictions, targets):
                    input_string = (" ".join([self._input_int2vocab.get(idx) for idx in input_])).split('<PAD>')[0] + '\n'
                    output_string = (" ".join([self._target_int2vocab.get(idx) for idx in pred])).split('</S>')[0] + '\n'
                    target_string = (" ".join([self._target_int2vocab.get(idx) for idx in target])).split('</S>')[0] + '\n'

                    f_input.write(input_string)
                    f_preds.write(output_string)
                    f_target.write(target_string)

                    reference.append((" ".join([self._target_int2vocab.get(idx) for idx in target])).split('</S>')[0])
                    translation.append((" ".join([self._target_int2vocab.get(idx) for idx in pred])).split('</S>')[0])

            except tf.errors.OutOfRangeError:
                break

        f_result.close()
        f_input.close()
        f_preds.close()
        f_target.close()

        print("DECODING PROCESS FINISH...")
        print('BLEU score: {}'.format(compute_bleu(reference_corpus=reference, translation_corpus=translation)))




