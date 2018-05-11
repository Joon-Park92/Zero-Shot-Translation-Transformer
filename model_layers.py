from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import numpy as np


def embedding(ids,
              vocab_size,
              embed_dim,
              zeropad=True,
              pos=True,
              scope='embedding',
              reuse=None):

    with tf.variable_scope(name_or_scope=scope, reuse=reuse):
        embed_table = tf.get_variable('embed_table', shape=[vocab_size, embed_dim], dtype=tf.float32)

        # Padding Character(<PAD>) to Zero-vector
        if zeropad:
            zero_embed = tf.convert_to_tensor(np.zeros(shape=[1, embed_dim]), dtype=tf.float32)
            embed_table = tf.concat([zero_embed, embed_table[1:, :]], axis=0)

        embeded_tensor = tf.nn.embedding_lookup(params=embed_table, ids=ids)

        # Positional Encoding
        if pos:
            seq_length = embeded_tensor.shape.as_list()[1]
            d_dim = embeded_tensor.shape.as_list()[2]

            def _pos_helper(pos_, i, d_dim_):
                if (i % 2) == 0:
                    return np.sin(pos_ / np.power(1e4, (2*i) / d_dim_))
                else:
                    return np.cos(pos_ / np.power(1e4, (2*i) / d_dim_))

            lookup_array = np.array([[_pos_helper(pos, i, d_dim) for i in range(1, d_dim+1)]
                                     for pos in range(1, seq_length+1)], np.float32)
            lookup_array = tf.convert_to_tensor(lookup_array, dtype=tf.float32)
            pos_table = tf.convert_to_tensor(lookup_array, dtype=tf.float32, name='pos_table')

            ids = np.arange(seq_length)
            pos_embed = tf.nn.embedding_lookup(params=pos_table, ids=ids)
            pos_embed = embeded_tensor + pos_embed

            # Masking (<PAD> to Zero-vector)
            mask = tf.sign(tf.abs(tf.reduce_sum(embeded_tensor, axis=-1)))
            mask = tf.tile(tf.expand_dims(mask, -1), [1, 1, tf.shape(pos_embed)[-1]])
            pos_embed *= mask
            return pos_embed

    return embeded_tensor


def multi_head_attention(queries,
                         keys,
                         num_units,
                         num_heads,
                         causality=False,
                         scope='multi_head_attention',
                         reuse=False):
    # Check dimension
    assert queries.shape[-1] == keys.shape[-1]

    with tf.variable_scope(scope, reuse=reuse):
        # Projection
        q_proj = tf.layers.dense(queries, units=num_units, activation=tf.nn.relu)
        k_proj = tf.layers.dense(keys, units=num_units, activation=tf.nn.relu)
        v_proj = tf.layers.dense(keys, units=num_units, activation=tf.nn.relu)

        # Split Head & Temporarily concatenate to the first dimension
        q_proj = tf.concat(tf.split(q_proj, num_or_size_splits=num_heads, axis=-1), axis=0)
        k_proj = tf.concat(tf.split(k_proj, num_or_size_splits=num_heads, axis=-1), axis=0)
        v_proj = tf.concat(tf.split(v_proj, num_or_size_splits=num_heads, axis=-1), axis=0)

        # Attention
        k_trans = tf.transpose(k_proj, perm=[0, 2, 1])
        k_dim = k_proj.shape.as_list()[-1]

        # Scale
        weights = tf.matmul(q_proj, k_trans) / tf.sqrt(tf.constant(k_dim, dtype=tf.float32,shape=[]))

        # Masking ( Causality )
        if causality:
            """
            Decoder : We also modify the self-attention sub-layer in the decoder stack 
            to prevent positions from attending to subsequent positions.
            """
            diag_vals = tf.ones_like(weights[0, :, :])
            tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense()
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(weights)[0], 1, 1])
            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            weights = tf.where(tf.equal(masks, 0), paddings, weights)

        weights = tf.nn.softmax(weights, axis=-1)
        attention = tf.matmul(weights, v_proj)

        # Restore splitted tensor
        attention = tf.concat(tf.split(attention, num_heads, axis=0), axis=-1)
        output = tf.layers.dense(attention, units=num_units, activation=tf.nn.relu)

        return output


def encoding_sublayer(input_embedding,
                      num_units,
                      num_heads,
                      drop_rate,
                      is_training,
                      scope=None,
                      reuse=False):

    with tf.variable_scope(scope, reuse=reuse):
        embeded_tensor = input_embedding

        # SubLayer_1 (Multi-Head Attention)
        with tf.variable_scope('sublayer_1'):
            residual = embeded_tensor
            embeded_tensor = multi_head_attention(queries=embeded_tensor,
                                                  keys=embeded_tensor,
                                                  num_units=num_units,
                                                  num_heads=num_heads,
                                                  causality=False)
            embeded_tensor = tf.layers.dropout(embeded_tensor, rate=drop_rate, training=is_training)
            # Add & Norm
            embeded_tensor = embeded_tensor + residual
            embeded_tensor = tf.layers.batch_normalization(embeded_tensor, axis=-1, training=is_training)

        # SubLayer_2 (Feed Forward)
        with tf.variable_scope('sublayer_2'):
            residual = embeded_tensor
            embeded_tensor = tf.layers.dense(embeded_tensor, units=num_units*4, activation=tf.nn.relu)
            embeded_tensor = tf.layers.dense(embeded_tensor, units=num_units)
            embeded_tensor = tf.layers.dropout(embeded_tensor, rate=drop_rate, training=is_training)
            # Add & Norm
            embeded_tensor = embeded_tensor + residual
            embeded_tensor = tf.layers.batch_normalization(embeded_tensor, axis=-1, training=is_training)

        return embeded_tensor


def decoding_sublayer(output_embedding,
                      input_embedding,
                      num_units,
                      num_heads,
                      drop_rate,
                      is_training,
                      scope=None,
                      reuse=False):

    with tf.variable_scope(scope, reuse=reuse):
        # SubLayer_1 (Masked Multi-Head Attention)
        with tf.variable_scope('sublayer_1'):
            residual = output_embedding
            output_embedding = multi_head_attention(queries=output_embedding,
                                                    keys=output_embedding,
                                                    num_units=num_units,
                                                    num_heads=num_heads,
                                                    causality=True)
            output_embedding = tf.layers.dropout(output_embedding, rate=drop_rate, training=is_training)
            # Add & Norm
            output_embedding = output_embedding + residual
            output_embedding = tf.layers.batch_normalization(output_embedding, axis=-1, training=is_training)

        # SubLayer_2 (Multi-Head Attention)
        with tf.variable_scope('sublayer_2'):
            residual = output_embedding
            embeded_tensor = multi_head_attention(queries=output_embedding,
                                                  keys=input_embedding,
                                                  num_units=num_units,
                                                  num_heads=num_heads,
                                                  causality=False)
            embeded_tensor = tf.layers.dropout(embeded_tensor, rate=drop_rate, training=is_training)
            # Add & Norm
            embeded_tensor = embeded_tensor + residual
            embeded_tensor = tf.layers.batch_normalization(embeded_tensor, axis=-1, training=is_training)

        # SubLayer_3 (Feed Forward)
        with tf.variable_scope('sublayer_3'):
            residual = embeded_tensor
            embeded_tensor = tf.layers.dense(embeded_tensor, units=num_units*4, activation=tf.nn.relu)
            embeded_tensor = tf.layers.dense(embeded_tensor, units=num_units)
            embeded_tensor = tf.layers.dropout(embeded_tensor, rate=drop_rate, training=is_training)
            embeded_tensor = embeded_tensor + residual
            embeded_tensor = tf.layers.batch_normalization(embeded_tensor, axis=-1, training=is_training)

        return embeded_tensor






