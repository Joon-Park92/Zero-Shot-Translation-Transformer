# -*- coding: utf-8 -*-
#/usr/bin/python2

from train import *

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
            
            sess.run(datasetmaker.get_init_ops())
            Model.load(sess=sess, save_path=add_hp_to_train_path(train_path=hp.train_path))
            print("Model Loaded")
            Model.evaluate(sess=sess,
                           dev_dataset_iterator = datasetmaker.dev_iterator)
            