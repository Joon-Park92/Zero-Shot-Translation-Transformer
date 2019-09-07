from data_load import load_vocabs, eval_iterator
from model import *

def main():
    zeroshot_vocab2int, zeroshot_int2vocab, lang_idx_dict = load_vocabs()
    x = tf.placeholder(tf.int32, shape=[hp.batch_size, hp.maxlen] )
    y = tf.placeholder(tf.int32, shape=[hp.batch_size, hp.maxlen] )
    is_train = tf.constant(False, tf.bool, name='is_train')
    model = Transformer(x, y, zeroshot_int2vocab, zeroshot_int2vocab, is_train)

    with tf.Session() as sess:

        sess.run([
            tf.global_variables_initializer(),
            eval_iterator.initializer,
            tf.tables_initializer()])

        model.evaluate(sess, eval_iterator)

if __name__ =='__main__':
    main()
