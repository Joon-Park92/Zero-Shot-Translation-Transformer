# -*- coding: utf-8 -*-
#/usr/bin/python2

import tensorflow as tf
import os

from data_load import ZeroShotVocabMaker, TFDataSetMaker
from model import Transformer
from hyperparams import hp
from utils import *


class Trainer(object):

    def __init__(self, sess, model, maker, is_training):
        self.sess = sess
        self.model = model
        self.maker = maker
        self.is_training = is_training

    def train(self):
        print("Training Starts....")

        sess = self.sess
        Model = self.model
        Maker = self.maker

        # train_path are variable depending on hyperparams
        train_path = add_hp_to_train_path(train_path=hp.train_path)
        dev_path = os.path.join(train_path, 'dev')

        if not os.path.isdir(hp.train_path): os.mkdir(hp.train_path)
        if not os.path.isdir(hp.train_path): os.mkdir(train_path)
        if not os.path.isdir(hp.train_path): os.mkdir(dev_path)

        train_writer = tf.summary.FileWriter(logdir=train_path, graph=sess.graph)
        dev_writer = tf.summary.FileWriter(logdir=dev_path)

        sess.run(Maker.get_init_ops())

        if tf.train.latest_checkpoint(train_path) is not None:
            Model.load(sess=sess, save_path=train_path)

        step = sess.run(Model.global_step, feed_dict={Maker.controller: False})

        while True:
            try:
                if step % hp.summary_every_n_step == 0:
                    _, loss, step, merged = sess.run([Model.train_op, Model.mean_loss, Model.global_step, Model.merged],
                                                     feed_dict={Maker.controller: True,
                                                                self.is_training: True})
                    train_writer.add_summary(merged, step)
                    print('TRAINING LOSS: {} / STEP: {}'.format(loss, step))

                    # Show examples
                    input_sentence, target_sentence, output_sentence = sess.run(
                        [input_tensor, target_tensor, Model.preds],
                        feed_dict={is_training: True, Maker.controller: True})

                    input_sentence = input_sentence[0]
                    target_sentence = target_sentence[0]
                    output_sentence = output_sentence[0]

                    print("INPUT: " +
                          " ".join([Maker.zeroshot_int2voca.get(i) for i in input_sentence]).split('<PAD>')[0])
                    print("OUTPUT: " + " ".join([Maker.zeroshot_int2voca.get(i) for i in output_sentence]))
                    print("TARGET: " +
                          " ".join([Maker.zeroshot_int2voca.get(i) for i in target_sentence]).split('<PAD>')[0] + '\n')

                else:
                    _, step = sess.run([Model.train_op, Model.global_step],
                                       feed_dict={Maker.controller: True,
                                                  self.is_training: True})

                if step % hp.save_every_n_step == 1:
                    Model.save(sess=sess, save_path=os.path.join(train_path, get_hp() + "_step"))
                    print("Model is saved - step : {}".format(step))

                if step % hp.evaluate_every_n_step == 1:
                    loss, step, merged = sess.run([Model.mean_loss, Model.global_step, Model.merged],
                                                  feed_dict={Maker.controller: False,
                                                             self.is_training: False})
                    dev_writer.add_summary(merged, step)
                    print('DEVELOP LOSS: {} / STEP: {}'.format(loss, step))

                    # Show examples
                    input_sentence, target_sentence, output_sentence = sess.run(
                        [input_tensor, target_tensor, Model.preds],
                        feed_dict={is_training: False, Maker.controller: False})

                    input_sentence = input_sentence[0]
                    target_sentence = target_sentence[0]
                    output_sentence = output_sentence[0]

                    print("INPUT: " +
                          " ".join([Maker.zeroshot_int2voca.get(i) for i in input_sentence]).split('<PAD>')[0])
                    print("OUTPUT: " + " ".join([Maker.zeroshot_int2voca.get(i) for i in output_sentence]))
                    print("TARGET: " +
                          " ".join([Maker.zeroshot_int2voca.get(i) for i in target_sentence]).split('<PAD>')[0] + '\n')

            except tf.errors.OutOfRangeError:
                print("Training is over...!")
                train_writer.close()
                dev_writer.close()
                break


if __name__ == '__main__':

    g = tf.Graph()

    with g.as_default():
        with tf.Session() as sess:
            with tf.name_scope('DataGenerate'):
                VocabMaker = ZeroShotVocabMaker(vocab_path=hp.vocab_path, minimum_count=hp.minimum_count)
                DataSetMaker = TFDataSetMaker(zeroshot_voca2int=VocabMaker.zeroshot_voca2int,
                                              zeroshot_int2voca=VocabMaker.zeroshot_int2voca)

                input_tensor, target_tensor = DataSetMaker.get_input_tensor()

            is_training = tf.placeholder(tf.bool)  # control training / eval mode
            Model = Transformer(x=input_tensor,
                                y=target_tensor,
                                input_int2vocab=VocabMaker.zeroshot_int2voca,
                                target_int2vocab=VocabMaker.zeroshot_int2voca,
                                num_units=hp.num_units,
                                num_blocks=hp.num_blocks,
                                num_heads=hp.num_heads,
                                drop_rate=hp.dropout_rate,
                                is_training=is_training,
                                warmup_step=hp.warmup_step,
                                max_len=hp.max_len)

            with tf.name_scope('Trainer_scope'):
                trainer = Trainer(sess=sess, model=Model, maker=DataSetMaker, is_training=is_training)
                trainer.train()
