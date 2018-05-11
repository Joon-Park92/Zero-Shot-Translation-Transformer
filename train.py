from data_load import load_vocabs, load_data_set
from model import *


class Trainer(object):
    def __init__(self, sess, model, config):
        pass

    def train_epoch(self, epoch):
        pass


if __name__ == '__main__':
    
    zeroshot_vocab2int, zeroshot_int2vocab, lang_idx_dict = load_vocabs()
    train_iterator, dev_iterator = load_data_set()

    input_controller = InputController(train_iterator, dev_iterator)
    x, y = input_controller.get_input()

    Model = Transformer(x, y, zeroshot_int2vocab, zeroshot_int2vocab, input_controller.controller)

    with tf.Session() as sess:

        train_writer = tf.summary.FileWriter(hp.logdir, sess.graph)
        if not os.path.isdir(os.path.join(hp.logdir, 'dev')) : os.mkdir(os.path.join(hp.logdir, 'dev'))
        dev_writer = tf.summary.FileWriter(logdir=os.path.join(hp.logdir, 'dev'))

        sess.run([tf.global_variables_initializer(), train_iterator.initializer, dev_iterator.initializer, tf.tables_initializer()])            
        if tf.train.latest_checkpoint(hp.logdir) is not None:
            Model.load(sess)
            print("Model is loaded from {}".format(hp.logdir))

        step = sess.run(Model.global_step, feed_dict={input_controller.controller:False})

        while True:
            try:                        
                if step % hp.summary_every_n_step == 0 :
                    _, step, merged = sess.run([Model.train_op, Model.global_step, Model.merged], feed_dict={input_controller.controller: True})
                    train_writer.add_summary(merged, step)

                else:
                    _, step = sess.run([Model.train_op, Model.global_step], feed_dict={input_controller.controller: True})

                if step % hp.save_every_n_step == 1 :
                    Model.save(sess)  
                    print("Model is saved - step : {}".format(step))

                if step % hp.evaluate_every_n_step == 1:
                    _, step, merged = sess.run([Model.train_op, Model.global_step, Model.merged], feed_dict={input_controller.controller: False})
                    dev_writer.add_summary(merged, step)

            except tf.errors.OutOfRangeError:
                print("Training is over...!")
                train_writer.close()
                dev_writer.close()
                break