import re
import string
import numpy as np
from collections import defaultdict
import tensorflow as tf
from tensorflow.keras import layers




def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation),
                                  '')


def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


def process_data(train_ds, batch_size=200):
    # get the train data
    data_batch = []
    label_batch = []
    for element in train_ds.as_numpy_iterator():
      data_batch.append(element[0])
      label_batch.append(element[1])
      # print(element)
      break

    train_data_use = np.asarray(data_batch).squeeze()#[0].reshape(1, 250)
    train_labels_use = np.asarray(label_batch).reshape(batch_size,1)

    return train_data_use, train_labels_use



def txt_to_emb(train_data_use, max_features=10000, embedding_dim = 250):
    # make embedding layer
    emb_layer = layers.Embedding(max_features + 1, embedding_dim,
                                 embeddings_initializer=tf.keras.initializers.RandomUniform(minval=0.0, maxval=1., seed=None))  # tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.1))

    emb_train_data_use = emb_layer(train_data_use)

    return emb_train_data_use



def set_emb_layer(max_features=10000, embedding_dim = 250):
    # make embedding layer
    emb_layer = layers.Embedding(max_features + 1, embedding_dim, embeddings_initializer=tf.keras.initializers.RandomUniform(minval=0.0, maxval=1., seed=None))  # tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.1))

    # learn the inversion
    inv = defaultdict(lambda: -1)
    inv[0.0] = 0.0

    red = lambda x: str(
        f"{int(np.sum(x) * (10 ** 5))}")  # in order to avoid problems with the floating point accuracy, we round. #hash over sums to be replaced ...
    # this might make us lose some individual mappings (but below we see that it is only very few.)

    for t in range(max_features):
        emb = emb_layer(np.array([t]))
        inv[red(emb)] = t

    return emb_layer, inv, red



def vectorizer(batch_size=20, max_features=10000, sequence_length=250):

    raw_train_ds = tf.keras.utils.text_dataset_from_directory('aclImdb/train', batch_size=batch_size)

    vectorize_layer = layers.TextVectorization(
        standardize=custom_standardization,
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length)

    vectorize_layer.adapt(raw_train_ds.map(lambda x, y: x))

    def vectorize_text(text, label):
        text = tf.expand_dims(text, -1)
        return vectorize_layer(text), label

    return vectorize_layer, vectorize_text
