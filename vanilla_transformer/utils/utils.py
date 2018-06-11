import tensorflow as tf
from nltk.translate import bleu_score
from hyperparams import hp
import os

# Model Build Util
def label_smoother(one_hot, epsilon=0.1):
    depth = one_hot.shape.as_list()[-1]
    return one_hot * (1-epsilon) + (1.0 - one_hot) * (epsilon / (depth-1))


def warmup_learning_rate(d_model, step_num, warmup_step):
    d_model = tf.cast(d_model, tf.float32)
    step_num = tf.cast(step_num, tf.float32)
    warmup_step = tf.cast(warmup_step, tf.float32)

    compare_1 = tf.pow(step_num, -0.5)
    compare_2 = step_num * tf.pow(warmup_step, -1.5)
    compare = tf.cond(pred=tf.greater_equal(compare_1, compare_2),
                      true_fn=lambda: compare_2,
                      false_fn=lambda: compare_1)

    learning_rate = tf.pow(d_model, -0.5) * compare

    return learning_rate

# Training Utils

def make_dir(path):
    os.makedirs(path=path,)


# Metric Utils
def compute_bleu(reference_corpus, translation_corpus):

    if len(reference_corpus.split()) < 4 or len(translation_corpus.split()) < 4:
        return 0.0

    return bleu_score.sentence_bleu(references=reference_corpus.split(),
                                    hypothesis=translation_corpus.split())


def get_hp():
    hparams = "{}{}_batch_{}_mincnt_{}_maxlen_{}_units_{}_head_{}_block_{}".format(hp.FROM,
                                                                                   hp.TO,
                                                                                   hp.batch_size,
                                                                                   hp.minimum_count,
                                                                                   hp.max_len,
                                                                                   hp.num_units,
                                                                                   hp.num_heads,
                                                                                   hp.num_blocks)
    return hparams


def add_hp_to_train_path(train_path):
    return os.path.join(train_path, get_hp())
