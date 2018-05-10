import tensorflow as tf


def label_smoother(one_hot, epsilon=0.1):
    depth = one_hot.shape.as_list()[-1]
    return one_hot * (1-epsilon) + (1.0 - one_hot) * (epsilon / (depth-1))


def warmup_learning_rate(d_model, step_num, warmup_step):
    d_model = tf.cast(d_model, tf.float32)
    step_num = tf.cast(step_num, tf.float32)
    warmup_step = tf.cast(warmup_step, tf.float32)

    compare_1 = tf.sqrt(step_num)
    compare_2 = step_num * tf.pow(warmup_step, -1.5)
    compare = tf.cond(pred=tf.greater_equal(compare_1, compare_2),
                      true_fn=lambda: compare_2,
                      false_fn=lambda: compare_2)

    learning_rate = tf.pow(d_model, -0.5) * compare

    return learning_rate


def compute_bleu(reference_corpus, translation_corpus):
    return NotImplementedError