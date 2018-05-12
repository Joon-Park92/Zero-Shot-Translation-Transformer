# -*- coding: utf-8 -*-
#/usr/bin/python2

import tensorflow as tf
import os

from data_load import ZeroShotVocabMaker, TFDataSetMaker
from model import Transformer
from hyperparams import hp


class Trainer(object):

    def __init__(self, sess, model, maker, is_training):
        self.sess = sess
        self.model = model
        self.maker = maker
        self.is_training = is_training

    def train(self):

        sess = self.sess
        Model = self.model
        Maker = self.maker

        hparams = "units_{}_head_{}_block_{}".format(hp.num_units, hp.num_heads, hp.num_blocks)
        train_path = os.path.join(hp.train_path, hparams)
        dev_path = os.path.join(train_path, 'dev')

        if not os.path.isdir(hp.train_path): os.mkdir(hp.train_path)
        if not os.path.isdir(hp.train_path): os.mkdir(train_path)
        if not os.path.isdir(hp.train_path): os.mkdir(dev_path)

        train_writer = tf.summary.FileWriter(logdir=train_path, graph=sess.graph)
        dev_writer = tf.summary.FileWriter(logdir=dev_path)

        sess.run(Maker.get_init_ops())

        if tf.train.latest_checkpoint(train_path) is not None:
            Model.load(sess=sess, save_path=tf.train.latest_checkpoint(checkpoint_dir=train_path))

        step = sess.run(Model.global_step, feed_dict={Maker.controller: False})

        while True:
            try:
                if step % hp.summary_every_n_step == 0:
                    _, step, merged = sess.run([Model.train_op, Model.global_step, Model.merged],
                                               feed_dict={Maker.controller: True,
                                                          self.is_training: True})
                    train_writer.add_summary(merged, step)

                else:
                    _, step = sess.run([Model.train_op, Model.global_step],
                                       feed_dict={Maker.controller: True,
                                                  self.is_training: True})

                if step % hp.save_every_n_step == 1:
                    Model.save(sess=sess, save_path=os.path.join(train_path, "_step: {}".format(step)))
                    print("Model is saved - step : {}".format(step))

                if step % hp.evaluate_every_n_step == 1:
                    _, step, merged = sess.run([Model.train_op, Model.global_step, Model.merged],
                                               feed_dict={Maker.controller: False,
                                                          self.is_training: False})
                    dev_writer.add_summary(merged, step)

            except tf.errors.OutOfRangeError:
                print("Training is over...!")
                train_writer.close()
                dev_writer.close()
                break


if __name__ == '__main__':

    g = tf.Graph()

    with g.as_default():
        with tf.Session() as sess:
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

            trainer = Trainer(sess=sess, model=Model, maker=DataSetMaker, is_training=is_training)
            trainer.train()
