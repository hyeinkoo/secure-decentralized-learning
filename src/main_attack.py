import os
import sys
import time
import tensorflow as tf
import numpy as np
import argparse
from copy import deepcopy
import pickle

from data.dataloader import *
from data.preprocess import vectorizer
from components.server import Server
from components.client import Client, create_clients
from utils.params import load_parameters, print_args, update_parameters
from utils.plot import plot_images_2
from models.model import *
from models.gradient import compute_gradients
from attack.initializer import *
from attack.eval import *
from rgm.rgm import *



class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


if __name__ == "__main__":

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    params = load_parameters()
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', default=params['dataset'])
    parser.add_argument('--model',   default=params['model'])

    parser.add_argument('--rgm', action='store_true')
    parser.add_argument('--adversarial_init', action='store_true')

    parser.add_argument('-n', '--num_clients', type=int, default=params['num_clients'])
    parser.add_argument('-d', '--dropout',     type=float, default=params['dropout_rate'])

    # parser.add_argument('-e', '--num_epochs', type=int, default=params['num_epochs'])
    parser.add_argument('-b', '--batch_size', type=int, default=params['batch_size'])

    parser.add_argument('-m', '--n_mix',  type=int, default=params['rgm_params']['n_mix'], nargs='+')
    parser.add_argument('-s', '--scale', type=int, default=params['rgm_params']['scale'], nargs='+')
    parser.add_argument('-f', '--frac',  type=float, default=params['rgm_params']['frac'])

    # parser.add_argument('--plot', action='store_true')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--print_result', action='store_true')

    # parser.add_argument('--no_log', action='store_true')

    parser.set_defaults(rgm=True,
                        adversarial_init=True,
                        # plot=True,
                        save=True,
                        print_result=False)

    args = parser.parse_args()
    params = update_parameters(params, args)
    l2_distance = [0.0, 1e-08, 1e-07, 1e-06, 1e-05, 1e-04, 1e-03, 1e-02, 1e-01, 1, 10]
    # print_args(args)

    with HiddenPrints():

        # print("Loading data and a model...")
        if args.dataset == "cifar10":
            train_data, _, _ = load_cifar_10(args.num_clients, args.batch_size)
            model = ImgModel()
            model.build(input_shape=(1, 32, 32, 3))
            if args.adversarial_init:
                adversarial_initialization(model)
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            emb_layer = None

        if args.dataset == "imdb":
            set_acl_imdb(args.num_clients)  # , args.batch_size)
            vectorize_layer, vectorize_text = vectorizer(args.batch_size, params["max_features"],
                                                         params["sequence_length"])
            emb_layer, inv, red = set_emb_layer(params["max_features"], params["embedding_dim"])
            vocab = vectorize_layer.get_vocabulary()
            train_data, _, _ = load_acl_imdb(vectorize_text, args.num_clients, args.batch_size)
            reconstruction_data = {"vectorize_layer": vectorize_layer,
                                   "inv": inv, "red": red,
                                   "sequence_length": params["sequence_length"]}

            model = TextModel()
            model.build(input_shape=(1, params["sequence_length"], params["embedding_dim"]))
            adversarial_initialization(model, hyperparameter={'down_scale_factor': 0.99},
                                       features=params["sequence_length"] * params["embedding_dim"])
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        # print("Setting clients...\n")

        dir_path = f"output/{args.dataset}"
        subdir_path = f"{dir_path}/n{args.num_clients}-b{args.batch_size}"
        run = f"{subdir_path}/m{args.n_mix[0]}-{args.n_mix[1]}_s{args.scale[0]}-{args.scale[1]}_f{args.frac}"

        mkdir(dir_path)
        mkdir(subdir_path)
        mkdir(run)
        if args.save:
            for c in range(args.num_clients):
                mkdir(f"{run}/c{c}")
        # if not os.path.exists(dir_name):
        #     os.mkdir(dir_name)
        # if not os.path.exists(os.path.join(dir_name, subdir_name)):
        #     os.mkdir(os.path.join(dir_name, subdir_name))
        # if not os.path.exists(os.path.join(dir_name, run)):
        #     os.mkdir(os.path.join(dir_name, run))
        # if args.save and not os.path.exists(os.path.join(dir_name, run, "c0")):
        #     for c in range(args.num_clients):
        #         os.mkdir(os.path.join(dir_name, run, f"c{c}"))

        # clients = create_clients(train_data, args.num_clients, args.batch_size, dataset=args.dataset, params=params)
        clients = create_clients(train_data, args.num_clients, args.batch_size, args.dataset, args.model, params)
        # `params` is not important

        num_iter = max([client.data_len for client in clients])

        keys = ["shape", "minmax_l2", "recall", "precision"]
        values = [[], []] + [dict([(l2, []) for l2 in l2_distance]) for _ in range(2)]
        rec_dic = dict([(key, value) for key, value in zip(keys, deepcopy(values))])
        rgm_dic = dict([(key, value) for key, value in zip(keys, deepcopy(values))])

        num_mixing_list = []
        for batch_idx in range(num_iter):

            # batch_idx = 0
            for c in range(args.num_clients):
                clients[c].reset_status()
                x, y = clients[c].data[batch_idx]
                if args.dataset == "imdb":
                    x = emb_layer(x)
                clients[c].gradients = compute_gradients(model, loss, x, y)

            reconstructions = []
            closest_samples = []
            for c, client in enumerate(clients):
                reconst = rescale_gradients(client.gradients[0], client.gradients[1])
                eval_dic, closest_sample = evaluate_attack(client, batch_idx, l2_distance, args.print_result, emb_layer)
                rec_dic = eval_dic_update(rec_dic, eval_dic)
                reconstructions.append(reconst)
                closest_samples.append(closest_sample)

            if args.rgm:
                random_gradient_mixing(clients, params["rgm_params"]["n_mix"],
                        params["rgm_params"]["scale"], params["rgm_params"]["frac"])
            num_mixing = [clients[i].n_mixed for i in range(len(clients))]
            num_mixing_list.append(num_mixing)

            # for c, (client, reconst, closest_sample) in enumerate(zip(clients, reconstructions, closest_samples)):
            #     reconst_rgm = rescale_gradients(client.gradients[0], client.gradients[1])
            #     eval_dic, closest_rgm = evaluate_attack(client, batch_idx, l2_distance, args.print_result, emb_layer)
            #     rgm_dic = eval_dic_update(rgm_dic, eval_dic)
            #     if args.dataset == 'cifar10':
            #         plot_images_2([client.data[batch_idx][0], reconst, reconst_rgm], save=args.save, run=f"{run}/c{c}/b{batch_idx}")
            #     else:
            #         org_emb = client.data[batch_idx][0][0].numpy().astype(int)
            #         org_sentence = ' '.join([vocab[token] for token in org_emb])
            #         text_reconstruction(org_sentence, [reconst, reconst_rgm], [closest_sample, closest_rgm],
            #                             reconstruction_data, save=args.save, run=f"{run}/c{c}/b{batch_idx}",
            #                             print_result=args.print_result)
            #

            ###
            for c, (client, reconst, closest_sample) in enumerate(zip(clients, reconstructions, closest_samples)):
                reconst_rgm = rescale_gradients(client.gradients[0], client.gradients[1])
                eval_dic, closest_rgm = evaluate_attack(client, batch_idx, l2_distance, args.print_result, emb_layer)
                rgm_dic = eval_dic_update(rgm_dic, eval_dic)
                run_name = f"{run}/c{c}/b%03.d" % batch_idx
                if args.dataset == 'cifar10':
                    pass
                    #                    plot_images_2([client.data[batch_idx][0], reconst, reconst_rgm], save=args.save, run=run_name)
                else:
                    org_emb = client.data[batch_idx][0][0].numpy().astype(int)
                    org_sentence = ' '.join([vocab[token] for token in org_emb])
                    text_reconstruction(org_sentence, [reconst, reconst_rgm], [closest_sample, closest_rgm],
                                        reconstruction_data, save=args.save, run=run_name,
                                        print_result=args.print_result)


        summarize(rec_dic, rgm_dic, l2_distance, run, args.save)
        if args.save:
            with open(os.path.join(run, "num_mixing.pkl"), 'wb') as f:
                pickle.dump(num_mixing_list, f)
