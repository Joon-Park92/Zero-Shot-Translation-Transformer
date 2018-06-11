import tensorflow as tf
import os


class Logger(object):

    def __init__(self, sess, logdir, graph):

        if not os.path.isdir(os.mkdir(os.path.join(logdir, 'dev'))):
            os.mkdir(os.path.join(logdir, 'dev'))

        self.train_writer = tf.summary.FileWriter(logdir=logdir, graph=graph)
        self.dev_writer = tf.summary.FileWriter(logdir=os.path.join(logdir, 'dev'))

    def summarize(self):
        pass

