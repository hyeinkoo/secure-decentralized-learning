import numpy as np
import tensorflow as tf
from models.model import *
from typing import List
from utils.params import load_parameters


class Client:

    def __init__(self, params):
        # params = load_parameters()

        self.id = np.random.randint(2 ** 30)

        self.dataset_name = None
        self.data = None # batched data
        self.data_len = None
        self.gradients = None
        self.batch_size = None
        self.dropout_rate = params['dropout_rate']
        self.num_mix = params['rgm_params']['n_mix'][0]

        self.model = None
        self.emb_layer = None

        self.peer_ids = []
        self.peer_dic = {}
        self.secret = None
        self.n_mixed = 0
        self.mix = True
        self.send = True
        self.dropout = None

        self.current_batch_size = None
        self.loss = None
        self.loss_fn = None
        self.optimizer = tf.keras.optimizers.Adam()
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        # self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        # self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


    def setup(self, data, batch_size, dataset="cifar10", model="fc-nn", emb_layer=None):
        self.dataset_name = dataset
        if self.dataset_name=="cifar10":
            self.data = data
            if model == "fc-nn":
                self.model = ImgModel()
            else: # model == "baseimg"
                self.model = BaseImgModel()
            self.model.build(input_shape=(1, 32, 32, 3))
            self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        else:
            self.data = data
            self.emb_layer = emb_layer
            self.model = TextModel()
            # self.model(np.zeros((1, 62500)))
            self.model.build(input_shape=(1, 250))#62500))
            self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            self.train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
        self.data_len = len(self.data)


    def send_proposal(self, g, p):
        self.secret = np.random.randint(2 ** 30)
        key = pow(g, self.secret, p)
        return self.id, key


    def receive_proposal(self, peer_id, peer_key):
        if peer_id in self.peer_dic.keys():  # exchanged
            return False
        else:
            self.peer_dic[peer_id] = {"key": peer_key}
            return True


    def generate_mask(self, peer_id, p, frac=0): # frac: fraction of gradient to send
        seed = pow(self.peer_dic[peer_id]["key"], self.secret, p)
#         np.random.seed(seed)
#         if frac == 0:
#             mask = np.random.uniform(size=self.gradients[0].shape) < np.random.uniform()
#         else:
#             mask = np.random.uniform(size=self.gradients[0].shape) < frac
#         self.peer_dic[peer_id]["secret"] = self.secret
#         self.peer_dic[peer_id]["mask"] = mask
        ####
        self.peer_dic[peer_id]["secret"] = self.secret
        self.peer_dic[peer_id]["mask"] = []
        for grad in self.gradients:
            np.random.seed(seed)
            if frac == 0:
                mask = np.random.uniform(size=grad.shape) < np.random.uniform()
            else:
                mask = np.random.uniform(size=grad.shape) < frac
            self.peer_dic[peer_id]["mask"].append(mask)


    def send_gradients(self, peer_id, scale):
        masks = self.peer_dic[peer_id]["mask"]
        gradients_to_send = []
        for i, mask in enumerate(masks):
            v = np.random.uniform(scale[0], scale[1], size=np.sum(mask))
            gradient = tf.boolean_mask(self.gradients[i], mask) * v
            self.gradients[i] = tf.tensor_scatter_nd_sub(self.gradients[i], tf.where(mask), gradient)
            gradients_to_send.append(gradient)
        return gradients_to_send


    def receive_gradients(self, peer_id, gradients):
        self.peer_dic[peer_id]["gradients"] = gradients


    def update_gradients(self, peer_id):
        masks = self.peer_dic[peer_id]["mask"]
        gradients = self.peer_dic[peer_id]["gradients"]
        for i, (mask, gradient) in enumerate(zip(masks, gradients)):
            self.gradients[i] = tf.tensor_scatter_nd_add(self.gradients[i], tf.where(mask), gradient)
        self.peer_dic[peer_id] = None


    def reset_loss(self):
        self.train_loss.reset_states()
        self.train_accuracy.reset_states()
        # self.test_loss.reset_states()
        # self.test_accuracy.reset_states()


    def train_batch(self, batch_idx):
        x, y = self.data[batch_idx]
#         if self.emb_layer:
#             x = self.emb_layer(x)
        self.current_batch_size = len(y)
        with tf.GradientTape() as tape:
            pred = self.model(x)
            loss = self.loss_fn(y, pred)
        self.gradients = tape.gradient(loss, self.model.trainable_variables)
        self.validate_gradient()
        self.loss = loss
        self.train_loss(loss)
        self.train_accuracy(y, pred)


    def validate_gradient(self):
        if tf.reduce_all(abs(tf.convert_to_tensor(self.gradients[0])) < 1e-99):
            self.mix = False


    def set_status(self, rgm):
        self.dropout = np.random.uniform() < self.dropout_rate
        if rgm and self.n_mixed < self.num_mix:
            self.send = False


    def reset_status(self):
        self.peer_ids = []
        self.peer_dic = {}
        self.secret = None
        self.n_mixed = 0
        self.mix = True
        self.send = True
        self.dropout = None


    def train_step(self):
        self.optimizer.apply_gradients(zip(self.gradients, self.model.trainable_variables))




def create_clients(data: list, num_clients: int, batch_size: int, dataset: str, model: str, params, emb_layer=None):

    clients = []
    for i in range(num_clients):
        client = Client(params)
        client.setup(data[i], batch_size, dataset, model, emb_layer)
        clients.append(client)

    return clients
