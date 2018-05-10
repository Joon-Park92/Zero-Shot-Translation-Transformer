import tensorflow as tf

def label_smoother(one_hot, epsilon=0.1):
    return