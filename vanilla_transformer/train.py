# -*- coding: utf-8 -*-
#/usr/bin/python2


from data.data_load import VocabMaker, TFDataSetMaker
from models.model import Transformer
from utils.utils import *


class Trainer(object):

    def __init__(self,
                 sess,
                 model,
                 summary_every_n_step,
                 save_every_n_step,
                 evaluate_every_n_step,
                 save_path,
                 is_training):

        self.sess = sess
        self.model = model
        self.summary_step = summary_every_n_step
        self.save_step = save_every_n_step
        self.eval_step = evaluate_every_n_step
        self.save_path = save_path
        self.is_training = is_training

        self.train_writer = tf.summary.FileWriter(logdir=save_path, graph=graph)
        self.dev_writer = tf.summary.FileWriter(logdir=os.path.join(save_path, 'dev'))

    def _train_step(self, summary=False):

        if summary:
            _, loss, step, merged = sess.run([self.model.train_op,
                                              self.model.mean_loss,
                                              self.model.global_step,
                                              self.model.merged],
                                             feed_dict={self.is_training: True})
            self.train_writer.add_summary(merged, step)
            print('TRAINING LOSS: {} / STEP: {}'.format(loss, step))

        else:
            _, step = sess.run([self.model.train_op, self.model.global_step], feed_dict={self.is_training: True})

    def _eval_step(self):
        loss, step, merged = sess.run([self.model.mean_loss,
                                       self.model.global_step,
                                       self.model.merged],
                                      feed_dict={self.is_training: False})
        self.dev_writer.add_summary(merged, step)
        print('DEVELOP LOSS: {} / STEP: {}'.format(loss, step))

    def train(self):

        train_path = add_hp_to_train_path(train_path=hp.train_path)

        if tf.train.latest_checkpoint(train_path) is not None:
            Model.load(sess=sess, save_path=train_path)

        step = sess.run(self.model.global_step, feed_dict={self.is_training: False})

        while True:
            try:
                if step % self.summary_step == 0:
                    self._train_step(summary=True)
                    self._show_examples(mode='train')

                else:
                    self._train_step(summary=False)

                if step % self.save_step == 1:
                    self.model.save(sess=sess, save_path=os.path.join(train_path, get_hp() + "_step.ckpt"))
                    print("Model is saved - step : {}".format(step))

                if step % self.eval_step == 1:
                    self._eval_step()
                    self._show_examples(mode='dev')

            except tf.errors.OutOfRangeError:
                print("Training is over...!")
                self.train_writer.close()
                self.dev_writer.close()
                break

    def _show_examples(self, mode='train'):

        if mode == 'train':
            # Show examples
            input_sentence, target_sentence, output_sentence, dec_inputs, _ = sess.run(
                [self.model.x, self.model.y, self.model.preds, self.model.decoder_inputs, self.model.train_op],
                feed_dict={self.is_training: True})

        elif mode == 'dev':
            # Show examples
            input_sentence, target_sentence, output_sentence, dec_inputs = sess.run(
                [self.model.x, self.model.y, self.model.preds, self.model.decoder_inputs],
                feed_dict={self.is_training: False})

        input_sentence = input_sentence[0]
        target_sentence = target_sentence[0]
        output_sentence = output_sentence[0]
        dec_inputs = dec_inputs[0]

        print("INPUT: "
              + " ".join([self.model.input_int2vocab.get(i) for i in input_sentence]).split('<PAD>')[0])
        print("DEC_IN: " + " ".join([self.model.target_int2vocab.get(i) for i in dec_inputs]))
        print("OUTPUT: " + " ".join([self.model.target_int2vocab.get(i) for i in output_sentence]))
        print("TARGET: "
              + " ".join([self.model.target_int2vocab.get(i) for i in target_sentence]).split('<PAD>')[0]
              + '\n')


if __name__ == '__main__':

    g = tf.Graph()
    with g.as_default():
        with tf.Session() as sess:
            with tf.name_scope('DataGenerate'):

                is_training = tf.placeholder(tf.bool)  # control training / eval mode

                vocamaker = VocabMaker(vocab_path=hp.vocab_path,
                                       minimum_count=hp.minimum_count,
                                       lang_from=hp.FROM,
                                       lang_to=hp.TO)

                datasetmaker = TFDataSetMaker(from_voca2int=vocamaker.from_voca2int,
                                              from_int2voca=vocamaker.from_int2voca,
                                              to_voca2int=vocamaker.to_voca2int,
                                              to_int2voca=vocamaker.to_int2voca,
                                              is_training=is_training)

                input_tensor, target_tensor = datasetmaker.get_input_tensor()

            Model = Transformer(x=input_tensor,
                                y=target_tensor,
                                input_int2vocab=vocamaker.from_int2voca,
                                target_int2vocab=vocamaker.to_int2voca,
                                num_units=hp.num_units,
                                num_blocks=hp.num_blocks,
                                num_heads=hp.num_heads,
                                drop_rate=hp.dropout_rate,
                                is_training=is_training,
                                warmup_step=hp.warmup_step,
                                max_len=hp.max_len)

            with tf.name_scope('Trainer_scope'):
                print("TRAINING START....")
                sess.run(datasetmaker.get_init_ops())
                trainer = Trainer(sess=sess,
                                  model=Model,
                                  summary_every_n_step=hp.summary_every_n_step,
                                  save_every_n_step=hp.save_every_n_step,
                                  evaluate_every_n_step=hp.evaluate_every_n_step,
                                  save_path=add_hp_to_train_path(hp.train_path),
                                  is_training=is_training)
                trainer.train()
