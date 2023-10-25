import tensorflow as tf
import numpy as np
import argparse
import time

from data.dataloader import *
from data.preprocess import vectorizer
from components.server import Server
from components.client import Client, create_clients
from utils.params import load_parameters, print_args, update_parameters



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

    parser.add_argument('-e', '--num_epochs', type=int, default=params['num_epochs'])
    parser.add_argument('-b', '--batch_size', type=int, default=params['batch_size'])

    parser.add_argument('-m', '--n_mix',  type=int, default=params['rgm_params']['n_mix'], nargs='+')
    parser.add_argument('-s', '--scale', type=int, default=params['rgm_params']['scale'], nargs='+')
    parser.add_argument('-f', '--frac',  type=float, default=params['rgm_params']['frac'])

    parser.add_argument('--no_log', action='store_true')

    parser.set_defaults(rgm=False,
                        adversarial_init=params['adversarial_init'],
                        no_log=False)

    args = parser.parse_args()
    params = update_parameters(params, args)
    print_args(args)

    print("Loading data...")
    if args.dataset == "cifar10":
        train_data, val_data, test_data = load_cifar_10(args.num_clients, args.batch_size)
    if args.dataset == "imdb":
        set_acl_imdb(args.num_clients)
        vectorize_layer, vectorize_text = vectorizer(args.batch_size, params["max_features"],
                                                     params["sequence_length"])
        emb_layer, inv, red = set_emb_layer(params["max_features"], params["embedding_dim"])
        vocab = vectorize_layer.get_vocabulary()
        train_data, val_data, test_data = load_acl_imdb(vectorize_text, args.num_clients, args.batch_size)


    print("Setting server and clients...\n")
    server = Server()
    server.setup(val_data, test_data, args.batch_size, args.adversarial_init, args.dataset, args.model)

    clients = create_clients(train_data, args.num_clients, args.batch_size, args.dataset, args.model, params, server.emb_layer)

    num_iter = max([client.data_len for client in clients])

    run = f"{args.dataset}/{args.model}/adv_init-{args.adversarial_init}/rgm-{args.rgm}/n{args.num_clients}_d{args.dropout}_e{args.num_epochs}_b{args.batch_size}"
    if args.rgm:
        mix_str = '-'.join([str(x) for x in args.n_mix])
        scale_str = '-'.join([str(x) for x in args.scale])
        run = run + f"_m{mix_str}_s{scale_str}_f{args.frac}"

    start_time = time.time()
    server.train(clients, args.num_epochs, num_iter, run=run, rgm=args.rgm, rgm_params=params['rgm_params'], log=not args.no_log)
    end_time = time.time()

    print(f'Execution time: {round((end_time - start_time)/60, 2)} minutes')

