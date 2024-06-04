import tensorflow as tf
import numpy as np


def repeat_kv(x: tf.Tensor, n_rep: int) -> tf.Tensor:
    """tf.repeat(x, repeats=n_rep, axis=2)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return tf.reshape(
        tf.tile(tf.expand_dims(x, axis=3), multiples=[1, 1, 1, n_rep, 1]),
        shape=[bs, slen, n_kv_heads * n_rep, head_dim]
    )


def rotate(inputs, d_model, max_position):
  positions = tf.range(0, max_position, dtype=tf.float32)
  theta = 10000.0 ** (-2 * positions / d_model)

  sin_coeff = np.zeros_like(inputs)

  sin_coeff[1::2] = inputs[1::2]
  sin_coeff[0::2] = -inputs[0::2]

  sin_matrix = np.zeros((d_model, d_model))
  cos_matrix = np.zeros((d_model, d_model))

  sin_matrix[:, 0::2] = np.sin(theta * positions)[0]
  sin_matrix[:, 1::2] = np.sin(theta * positions)[0]

  cos_matrix[0::2] = np.cos(theta * positions)[0]
  cos_matrix[1::2] = np.cos(theta * positions)[0]

  output = tf.matmul(inputs, cos_matrix) + tf.matmul(sin_coeff, sin_matrix)
  return tf.cast(output, tf.float32)


class GroupedMultiQueryAttention(tf.keras.layers.Layer):
  def __init__(self, num_heads, d_model, max_batch_size, max_seq_length,
               gpu_count=1, num_kv_heads=None):
    super(GroupedMultiQueryAttention, self).__init__()
    self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
    self.gpu_count = gpu_count
    self.local_num_heads = num_heads // gpu_count
    self.local_kv_heads = self.num_kv_heads // gpu_count
    self.n_rep = num_heads // self.num_kv_heads
    self.d_model = d_model
    self.depth = d_model // num_heads

    self.query_dense = tf.keras.layers.Dense(d_model, use_bias=False)
    self.key_dense = tf.keras.layers.Dense(self.depth * self.num_kv_heads, use_bias=False)
    self.value_dense = tf.keras.layers.Dense(self.depth * self.num_kv_heads, use_bias=False)

    self.final = tf.keras.layers.Dense(d_model, use_bias=False)

    self.k_cache = np.zeros((max_batch_size, max_seq_length, self.local_kv_heads, self.depth))
    self.v_cache = np.zeros((max_batch_size, max_seq_length, self.local_kv_heads, self.depth))


  def call(self, inputs, start_pos):
    q, k, v, mask = inputs
    q = self.query_dense(q)
    k = self.key_dense(k)
    v = self.value_dense(v)
    batch_size, seq_len, _ = q.shape

    q = tf.reshape(q, (batch_size, seq_len, self.local_num_heads, self.depth))
    k = tf.reshape(k, (batch_size, seq_len, self.num_kv_heads, self.depth))
    v = tf.reshape(v, (batch_size, seq_len, self.num_kv_heads, self.depth))

    q = rotate(q, self.depth, seq_len)
    k = rotate(k, self.depth, seq_len)

    self.k_cache[:batch_size, start_pos : start_pos + seq_len] = k
    self.v_cache[:batch_size, start_pos : start_pos + seq_len] = v

    k = repeat_kv(k, self.n_rep)
    v = repeat_kv(v, self.n_rep)

    q = tf.transpose(q, perm=[0, 2, 1, 3])
    k = tf.transpose(k, perm=[0, 2, 1, 3])
    v = tf.transpose(v, perm=[0, 2, 1, 3])

    matmul_qk = tf.matmul(q, k, transpose_b=True)

    if mask is not None:
      matmul_qk += mask

    output = tf.nn.softmax(matmul_qk, axis=-1)
    output = tf.matmul(output, v)
    output = tf.transpose(output, perm=[0, 2, 1, 3])
    output = tf.reshape(output, (batch_size, seq_len, self.d_model))
    return self.final(output)