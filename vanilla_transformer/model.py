# -*- coding: utf-8 -*-
# /usr/bin/python2

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

    def __init__(self,
                 x,
                 y,
                 input_int2vocab,
                 target_int2vocab,
                 num_units,
                 num_blocks,
                 num_heads,
                 drop_rate,
                 is_training,
                 warmup_step,
                 max_len):
        """
        Args:
            x : input 2-D tensor [ N, T ]
            y : target 2-D tensor [ N, T ]
            input_int2vocab : dictionary map integer to vocab in input space
            target_int2vocab : dictionary map integer to vocab in target space
            is_training : control_tensor(tf.bool type) used for switching training / validation data
        """
        assert num_units % num_heads == 0, "num_units must can be divided by num_head"

        self.x = x
        self.y = y
        # decoder inputs: <S> + shifted right
        self.decoder_inputs = tf.concat((tf.ones_like(self.y[:, :1]) * 2, self.y[:, :-1]), -1)

        self._num_units = num_units
        self._num_heads = num_heads
        self._drop_rate = drop_rate
        self._num_blocks = num_blocks
        self._warmup_step = warmup_step
        self._max_len = max_len

        self.input_int2vocab = input_int2vocab
        self.target_int2vocab = target_int2vocab
        self._is_training = is_training
        self._build_graph()

    def __call__(self, inputs, decoder_inputs, drop_rate, is_training):
        return self._transformer_layer(inputs=inputs,
                                       decoder_inputs=decoder_inputs,
                                       drop_rate=drop_rate,
                                       is_training=is_training,
                                       reuse=tf.AUTO_REUSE)

    def _transformer_layer(self,
                           inputs,
                           decoder_inputs,
                           drop_rate,
                           is_training,
                           scope='Transformer_body',
                           reuse=tf.AUTO_REUSE):

        with tf.variable_scope(name_or_scope=scope, reuse=reuse):
            with tf.name_scope('ENCODER'):
                # Input Embedding + Positional Encoding
                input_embedding = embedding(ids=inputs,
                                            vocab_size=len(self.input_int2vocab),
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
                                                        num_heads=self._num_heads,
                                                        drop_rate=drop_rate,
                                                        is_training=is_training,
                                                        scope='enc_block_{}'.format(i),
                                                        reuse=False)
            with tf.name_scope('DECODER'):
                output_embedding = embedding(ids=decoder_inputs,
                                             vocab_size=len(self.target_int2vocab),
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
                                                         num_heads=self._num_heads,
                                                         drop_rate=drop_rate,
                                                         is_training=is_training,
                                                         scope='dec_block_{}'.format(i),
                                                         reuse=False)

            # Final linear projection
            with tf.name_scope('FINAL_DENSE'):
                logits = tf.layers.dense(inputs=output_embedding, units=len(self.target_int2vocab))

        return logits

    def _build_graph(self):
        with tf.name_scope('TRANSFORMER'):
            self.logits = self._transformer_layer(inputs=self.x,
                                                  decoder_inputs=self.decoder_inputs,
                                                  drop_rate=self._drop_rate,
                                                  is_training=self._is_training)

            with tf.name_scope('Loss'):
                self.preds = tf.to_int32(tf.argmax(self.logits, axis=-1))
                # Accuracy: Remove <PAD> Characters
                is_target = tf.to_float(tf.not_equal(self.y, 0))
                total_num = tf.reduce_sum(is_target)
                correct_num = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.y)) * is_target)
                self.acc = correct_num / total_num

                # Loss: Remove <PAD> Characters
                self.y_smoothed = tf.cond(pred=self._is_training,
                                          true_fn=lambda: label_smoother(
                                              tf.one_hot(self.y, depth=len(self.target_int2vocab))),
                                          false_fn=lambda: tf.one_hot(self.y, depth=len(self.target_int2vocab)))
                self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_smoothed)
                self.mean_loss = tf.reduce_sum(self.loss * is_target) / (tf.reduce_sum(is_target))

            with tf.name_scope('Training_Scheme'):
                self.global_step = tf.get_variable(name='global_step',
                                                   shape=[],
                                                   dtype=tf.int32,
                                                   initializer=tf.constant_initializer(value=1, dtype=tf.int32),
                                                   trainable=False)
                self.learning_rate = warmup_learning_rate(d_model=self._num_units,
                                                          step_num=self.global_step,
                                                          warmup_step=self._warmup_step)
                # for batch normalization update
                self.update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                        beta1=0.9, beta2=0.98, epsilon=1e-9)

                with tf.control_dependencies(self.update_op):
                    self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)

            with tf.name_scope('Summary'):
                tf.summary.scalar('accuracy', self.acc)
                tf.summary.scalar('mean_loss', self.mean_loss)
                tf.summary.scalar('learning_rate', self.learning_rate)

            self.merged = tf.summary.merge_all()
            self.saver = tf.train.Saver()
        print("Model is built...")

    def save(self, sess, save_path):
        self.saver.save(sess=sess, save_path=save_path, global_step=self.global_step)

    def load(self, sess, save_path):
        self.saver.restore(sess=sess, save_path=tf.train.latest_checkpoint(checkpoint_dir=save_path))

    def translate(self, enc_inputs, is_training):
        """Greedy decoding translation
        Args:
            enc_inputs:
            is_training:

        Return:
            translated tensor ( decode auto-regressively )
        """
        max_len = self._max_len
        func = lambda x1, x2: self.__call__(inputs=x1,
                                            decoder_inputs=x2,
                                            drop_rate=self._drop_rate,
                                            is_training=is_training)

        def _cond(i, *args):
            return tf.less(i, max_len)

        def _body(i, enc_in, dec_in):
            i = tf.add(i, 1)
            logits = func(enc_in, dec_in)
            next_decoder_inputs = tf.to_int32(tf.argmax(logits, axis=-1))
            # Masking (causality)
            mask = tf.zeros_like(next_decoder_inputs)
            next_decoder_inputs = tf.concat([next_decoder_inputs[:, :(i + 1)], mask[:, (i + 1):]], axis=-1)
            next_decoder_inputs.set_shape([None, max_len])
            return i, enc_in, next_decoder_inputs

        init_dec_input = np.zeros(shape=[max_len])
        init_dec_input[0] = 2  # <S>
        init_dec_input = tf.zeros_like(enc_inputs, dtype=tf.int32) \
                         + tf.convert_to_tensor(init_dec_input, dtype=tf.int32)

        loop_vars = [tf.constant(1), enc_inputs, init_dec_input]
        _, _, translated_tensor = tf.while_loop(cond=_cond,
                                                body=_body,
                                                loop_vars=loop_vars,
                                                back_prop=is_training)

        return translated_tensor

    def evaluate(self, sess, dev_dataset_iterator):
        """
        Translate development data and save translated data
        Args:
            sess: tensorflow Session Object
            dev_dataset_iterator: tf.Dataset.iterator Object
        """
        train_path = add_hp_to_train_path(train_path=hp.train_path)
        path = os.path.join(train_path, 'result')
        if not os.path.isdir(path): os.mkdir(path)

        f_result = codecs.open(os.path.join(path, 'result.txt'), 'w', 'utf-8')
        f_input = codecs.open(os.path.join(path, 'input.txt'), 'w', 'utf-8')
        f_output = codecs.open(os.path.join(path, 'output.txt'), 'w', 'utf-8')
        f_target = codecs.open(os.path.join(path, 'target.txt'), 'w', 'utf-8')

        reference = []
        translation = []

        while 1:
            try:
                inputs, targets = dev_dataset_iterator.get_next()
                predictions = self.translate(enc_inputs=inputs, is_training=False)

                inputs, targets, predictions = sess.run([inputs, targets, predictions])

                for input_, pred, target in zip(inputs, predictions, targets):
                    input_string = (" ".join([self.input_int2vocab.get(idx) for idx in input_])).split('<PAD>')[0]
                    output_string = (" ".join([self.target_int2vocab.get(idx) for idx in pred])).split('</S>')[0]
                    target_string = (" ".join([self.target_int2vocab.get(idx) for idx in target])).split('</S>')[0]

                    bleu = compute_bleu(reference_corpus=target_string, translation_corpus=target_string)
                    result_string = "".join(['INPUTS: ' + input_string + '\n',
                                             'OUTPUT: ' + output_string + '\n',
                                             'TARGET: ' + target_string + '\n',
                                             'BLEU  : {}'.format(bleu) + '\n'])
                    print(result_string + '\n')

                    f_input.write(input_string + '\n')
                    f_output.write(output_string + '\n')
                    f_target.write(target_string + '\n')
                    f_result.write(result_string)

                    if len(target_string.split()) >= 4 and len(output_string.split()) >= 4:
                        reference.append(target_string)
                        translation.append(output_string)

            except tf.errors.OutOfRangeError:
                break

        f_result.close()
        f_input.close()
        f_output.close()
        f_target.close()

        print("DECODING PROCESS FINISH...")
        print('BLEU score: {}'.format(compute_bleu(reference_corpus=reference, translation_corpus=translation)))

    # def translate(self, enc_inputs, is_training):
    #     """Greedy decoding translation
    #     Args:
    #         enc_inputs:
    #         is_training:
    #
    #     Return:
    #         translated tensor ( decode auto-regressively )
    #     """
    #     max_len = self._max_len
    #     func = lambda x1, x2: self.__call__(inputs=x1,
    #                                         decoder_inputs=x2,
    #                                         drop_rate=self._drop_rate,
    #                                         is_training=is_training)
    #
    #     init_dec_input = np.zeros(shape=[max_len])
    #     init_dec_input[0] = 2  # <S>
    #     init_dec_input = tf.zeros_like(enc_inputs, dtype=tf.int32) \
    #                      + tf.convert_to_tensor(init_dec_input, dtype=tf.int32)
    #
    #     outputs = tf.TensorArray(dtype=tf.int32, size=max_len, clear_after_read=True)
    #     outputs = outputs.write(0, init_dec_input)
    #
    #     def _cond(i, *args):
    #         return tf.less(i, max_len - 1)
    #
    #     def _body(i, enc_in, outputs):
    #         dec_in = outputs.read(i)
    #         logits = func(enc_in, dec_in)
    #         next_decoder_inputs = tf.to_int32(tf.argmax(logits, axis=-1))
    #
    #         i = tf.add(i, 1)
    #         dec_in = tf.concat([dec_in[:, :i], tf.expand_dims(next_decoder_inputs[:, i], axis=-1)], axis=-1)
    #         dec_in = tf.concat([dec_in[:, :(i + 1)], tf.zeros_like(enc_in, tf.int32)[:, (i + 1):]], axis=-1)
    #         outputs = outputs.write(i, dec_in)
    #
    #         return i, enc_in, outputs
    #
    #     loop_vars = [tf.constant(0), enc_inputs, outputs]
    #     _, _, outputs = tf.while_loop(cond=_cond,
    #                                   body=_body,
    #                                   loop_vars=loop_vars,
    #                                   back_prop=is_training)
    #
    #     translated_tensor = outputs.read(max_len - 1)
    #
    #     return translated_tensor