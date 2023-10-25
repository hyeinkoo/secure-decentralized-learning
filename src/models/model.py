import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, MaxPooling2D, Conv2D, Activation, Dropout, Flatten, Input, Dense, Softmax, Input, Embedding
from collections import defaultdict
import numpy as np
from data.preprocess import set_emb_layer


class ImgModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = Flatten()
        self.d1 = Dense(1000, input_shape=(32 * 32 * 3,), activation='relu',
                        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.5))
        self.d2 = Dense(3000, activation='relu')
        self.d3 = Dense(3000, activation='relu')
        self.d4 = Dense(2000, activation='relu')
        self.d5 = Dense(1000, activation='relu')
        self.d6 = Dense(10)


    def call(self, x, training=True):
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        x = self.d5(x)
        x = self.d6(x)
        return x


class BaseImgModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = Flatten()
        self.d1 = Dense(1000, input_shape=(32 * 32 * 3,), activation='relu',
                        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.5))
        self.d2 = Dense(10)


    def call(self, x, training=True):
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return x




class TextModel(tf.keras.Model):
    def __init__(self, max_features=10000, embedding_dim = 250):
        super().__init__()
        # self.emb_layer, self.inv, self.red = set_emb_layer(max_features, embedding_dim)
        self.emb_layer = Embedding(max_features + 1, embedding_dim, embeddings_initializer=tf.keras.initializers.RandomUniform(minval=0.0, maxval=1., seed=None))
        self.flatten = Flatten()
        self.d1 = Dense(1000, input_shape=(62500,), activation='relu')
#         self.d2 = Dense(3000)
        self.d3 = Dense(1)#, activation='sigmoid')

    def call(self, x):
        x = self.emb_layer(x)
        x = self.flatten(x)
        x = self.d1(x)
#         x = self.d2(x)
        x = self.d3(x)
        return x


def set_emb_layer(max_features=10000, embedding_dim = 250):
    # make embedding layer
    emb_layer = Embedding(max_features + 1, embedding_dim, embeddings_initializer=tf.keras.initializers.RandomUniform(minval=0.0, maxval=1., seed=None))  # tf.keras.initializers.RandomNormal(mean=0.5, stddev=0.1))

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


def emb_inv_dic(emb_layer, max_features=10000):
    # learn the inversion
    inv = defaultdict(lambda: -1)
    inv[0.0] = 0.0

    red = lambda x: str(
        f"{int(np.sum(x) * (10 ** 5))}")  # in order to avoid problems with the floating point accuracy, we round. #hash over sums to be replaced ...
    # this might make us lose some individual mappings (but below we see that it is only very few.)

    for t in range(max_features):
        emb = emb_layer(np.array([t]))
        inv[red(emb)] = t

    return inv, red