import tensorflow as tf
import numpy as np
import datetime
from models.model import *
from rgm.rgm import random_gradient_mixing
from attack.initializer import adversarial_initialization
from attack.eval import rescale_gradients
from utils.plot import plot_images
from data.preprocess import set_emb_layer, vectorizer


class Server:

    def __init__(self):

        self.data = None
        self.val_dataloader = None
        self.test_dataloader = None
        self.model = None

        self.vectorize_layer = None
        self.vectorize_text = None
        self.emb_layer = None

        self.loss_fn = None
        self.optimizer = tf.keras.optimizers.Adam()

        self.train_loss = 0.
        self.train_accuracy = 0.
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    def setup(self, val_dataset, test_dataset, batch_size, adversarial_init, dataset="cifar10", model="fc-nn"):
        self._data_setup(val_dataset, test_dataset, dataset, batch_size)
        self._model_setup(model, adversarial_init, dataset)

    def _data_setup(self, val_dataset, test_dataset, dataset, batch_size=20,
                    max_features=10000, sequence_length=250, embedding_dim=250):
        self.val_dataloader = val_dataset
        self.test_dataloader = test_dataset
        if dataset == "imdb":
            self.vectorize_layer, self.vectorize_text = vectorizer(batch_size, max_features, sequence_length)
            self.emb_layer, self.inv, self.red = set_emb_layer(max_features, embedding_dim)

    def _model_setup(self, model, adversarial_init, dataset,
                     sequence_length=250, embedding_dim=250):
        if dataset == "cifar10":
            if model == "fc-nn":
                self.model = ImgModel()
            else: # model == "baseimg"
                self.model = BaseImgModel()
            # self.model(np.zeros((1, 32, 32, 3)))
            self.model.build(input_shape=(1, 32, 32, 3))
            if adversarial_init:
                adversarial_initialization(self.model)
            self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        else:
            self.model = TextModel()
            self.model.build(input_shape=(1, sequence_length)) # sequence_length, embedding_dim))
            if adversarial_init:
                adversarial_initialization(self.model, hyperparameter={'down_scale_factor': 0.99},
                                           features=sequence_length * embedding_dim)
            self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
            self.val_accuracy = tf.keras.metrics.BinaryAccuracy(name='val_accuracy')
            self.test_accuracy = tf.keras.metrics.BinaryAccuracy(name='test_accuracy')

    def reset_loss(self):
        self.train_loss = 0
        self.train_accuracy = 0
        self.val_loss.reset_states()
        self.val_accuracy.reset_states()

    def update_metrics(self, clients=None, weights=None, num_iter=None, epoch_end=False):
        if epoch_end:
            self.train_loss /= num_iter
            self.train_accuracy /= num_iter
        else:
            self.train_loss += sum(np.array([c.train_loss.result() for c in clients]) * weights)
            self.train_accuracy += sum(np.array([c.train_accuracy.result() for c in clients]) * weights)

    def aggregate_weights(self, clients):
        c_weights = np.array([c.current_batch_size for c in clients])
        c_weights = c_weights / sum(c_weights)

#         new_gradients = [np.zeros_like(g) for g in clients[0].gradients]
#         for j, client in enumerate(clients):
#             for k, gradient in enumerate(client.gradients):
#                 new_gradients[k] += gradient * c_weights[j]
                
        new_gradients = [np.zeros_like(tf.convert_to_tensor(g)) for g in clients[0].gradients]
        for j, client in enumerate(clients):
            for k, gradient in enumerate(client.gradients):
                new_gradients[k] += tf.convert_to_tensor(gradient) * c_weights[j]

        return new_gradients, c_weights

    def check_connection(self, clients):
        # for c in clients:
        #     print(f"send: {c.send},\t dropout: {c.dropout},\t connection: {c.send and not c.dropout}")
        surviving_clients = [c for c in clients if c.send and not c.dropout]
        # print(len(surviving_clients))
        return surviving_clients

    def train_step(self, surviving_clients):
        new_gradients, client_weights = self.aggregate_weights(surviving_clients)
        self.optimizer.apply_gradients(zip(new_gradients, self.model.trainable_variables))
        self.update_metrics(clients=surviving_clients, weights=client_weights)

    def create_log_writer(self, run):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/' + run + '/' + current_time + '/train'
        val_log_dir = 'logs/' + run + '/' + current_time + '/val'
        test_log_dir = 'logs/' + run + '/' + current_time + '/test'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        val_summary_writer = tf.summary.create_file_writer(val_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)
        return train_summary_writer, val_summary_writer, test_summary_writer

    def train(self, clients, epochs=1, num_iter=10, run=None, rgm=False, rgm_params={}, log=True):

        if log:
            train_summary_writer, val_summary_writer, test_summary_writer = self.create_log_writer(run)

        for epoch in range(epochs):

            self.reset_loss()
            for client in clients:
                client.reset_loss()

            for i in range(num_iter):
                for client in clients:
                    client.reset_status()
                    client.model.set_weights(self.model.get_weights())
                    client.train_batch(batch_idx=i)

                if rgm:
                    random_gradient_mixing(clients, **rgm_params)
                for client in clients:
                    client.set_status(rgm)

                surviving_clients = self.check_connection(clients)  # clients not dropped out & sent gradients
                if len(surviving_clients) == 0:
                    continue

                # update model
                self.train_step(surviving_clients)

            self.update_metrics(num_iter=num_iter, epoch_end=True)
            self.validation()

            if log:
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', self.train_loss, step=epoch)
                    tf.summary.scalar('accuracy', self.train_accuracy, step=epoch)
                with val_summary_writer.as_default():
                    tf.summary.scalar('loss', self.val_loss.result(), step=epoch)
                    tf.summary.scalar('accuracy', self.val_accuracy.result(), step=epoch)

            template = 'Epoch {}, Loss: {}, Accuracy: {}, Val Loss: {}, Val Accuracy: {}'
            print(template.format(epoch + 1,
                                  self.train_loss,
                                  self.train_accuracy * 100,
                                  self.val_loss.result(),
                                  self.val_accuracy.result() * 100))

        self.test()
        if log:
            with test_summary_writer.as_default():
                tf.summary.scalar('loss', self.test_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', self.test_accuracy.result(), step=epoch)
        print(f'\nTest Loss: {self.test_loss.result()}, Test Accuracy: {self.test_accuracy.result() * 100}\n')

    def validation(self):
        for x, y in self.val_dataloader:
#             if self.emb_layer:
#                 x = self.emb_layer(x)
            pred = self.model(x, training=False)
            loss = self.loss_fn(y, pred)
            self.val_loss(loss)
            self.val_accuracy(y, pred)

    def test(self):
        for x, y in self.test_dataloader:
#             if self.emb_layer:
#                 x = self.emb_layer(x)
            pred = self.model(x, training=False)
            loss = self.loss_fn(y, pred)
            self.test_loss(loss)
            self.test_accuracy(y, pred)

    def attack(self, clients, run):
        for c, client in enumerate(clients):
            reconstruction = rescale_gradients(client.gradients[0], client.gradients[1])
            plot_images(reconstruction, save=True, run=run + f"_c-{c}")
