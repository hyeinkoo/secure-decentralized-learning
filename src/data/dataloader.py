import os
import shutil
import tensorflow as tf
from tensorflow.keras import layers
from data.preprocess import *
from utils.params import load_parameters



def load_cifar_10(num_clients=5, batch_size=20):
    """Loads CIFAR10-Dataset and preprocesses it."""
    train, test = tf.keras.datasets.cifar10.load_data()

    train_data, train_labels = train
    test_data, test_labels = test

    # form of normalization used until 10.11.2020
    # mean = np.mean(train_data, axis=(0, 1, 2, 3))
    # std = np.std(train_data, axis=(0, 1, 2, 3))
    # train_data = (train_data - mean) / (std + 1e-7)
    # test_data = (test_data - mean) / (std + 1e-7)
    train_labels = train_labels.flatten()  # added for skript 51
    test_labels = test_labels.flatten()
    train_data, test_data = train_data / 255.0, test_data / 255.0

    split_idx = 40000
    val_data = train_data[split_idx:]
    val_labels = train_labels[split_idx:]
    train_data = train_data[:split_idx]
    train_labels = train_labels[:split_idx]

    data_size = int(len(train_labels) / num_clients)

    train_data_list = []
    for i in range(num_clients):
        x = train_data[i*data_size : (i+1)*data_size]
        y = train_labels[i*data_size : (i+1)*data_size]

        train_dataloader = tf.data.Dataset.from_tensor_slices((x,y))
        train_dataloader = train_dataloader.shuffle(data_size).batch(batch_size)
        # train_dataloader = train_dataloader.batch(batch_size)
        train_dataset = [(x, y) for x, y in train_dataloader]
        train_data_list.append(train_dataset)

    val_dataloader = tf.data.Dataset.from_tensor_slices((val_data, val_labels))
    val_dataloader = val_dataloader.shuffle(len(val_labels)).batch(batch_size)

    test_dataloader = tf.data.Dataset.from_tensor_slices((test_data, test_labels))
    test_dataloader = test_dataloader.shuffle(len(test_labels)).batch(batch_size)

    return train_data_list, val_dataloader, test_dataloader



def set_acl_imdb(num_clients=10):

    if os.path.exists("aclImdb"):
        dataset_dir = "aclImdb"
        train_dir = dataset_dir + "/train"
    else:
        url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
        dataset = tf.keras.utils.get_file("aclImdb_v1", url,
                                          untar=True, cache_dir='.',
                                          cache_subdir='')
        dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
        train_dir = os.path.join(dataset_dir, 'train')
        remove_dir = os.path.join(train_dir, 'unsup')
        shutil.rmtree(remove_dir)

    train_dir_new = os.path.join(dataset_dir, f"train_split_{num_clients}")
    if not os.path.exists(train_dir_new):
        os.mkdir(train_dir_new)

        path_src = [os.path.join(train_dir, label) for label in ['pos', 'neg']]
        data_size = int(len(os.listdir(path_src[0])) / num_clients)

        file_list = [sorted(os.listdir(path)) for path in path_src]
        for i in range(num_clients):
            client_path = os.path.join(train_dir_new, f'c{i}')
            path_dst = [os.path.join(client_path, label) for label in ['pos', 'neg']]
            for dir in [client_path, *path_dst]:
                if not os.path.exists(dir):
                    os.mkdir(dir)

            for files, path_s, path_d in zip(file_list, path_src, path_dst):
                if os.listdir(path_d) == []:
                    files = files[i * data_size: (i + 1) * data_size]
                    for f in files:
                        shutil.copy(os.path.join(path_s, f), path_d)
    val_dir = os.path.join(dataset_dir, "val")
    test_dir = os.path.join(dataset_dir, "test")
    if not os.path.exists(val_dir):
        os.mkdir(val_dir)
        path_src = [os.path.join(test_dir, label) for label in ['pos', 'neg']]
        data_size = int(len(os.listdir(path_src[0])) / 2)

        for label in ['pos', 'neg']:
            src_dir = os.path.join(test_dir, label)
            files = sorted(os.listdir(src_dir))
            dst_dir = os.path.join(val_dir, label)

            if not os.path.exists(dst_dir):
                os.mkdir(dst_dir)
            for file in files[:data_size]:
                shutil.move(os.path.join(src_dir, file), dst_dir)



def load_acl_imdb(vectorize_text, num_clients=5, batch_size=20):

    dataset_dir = 'aclImdb'
    train_dir = os.path.join(dataset_dir, 'train')

    train_data_list = []
    train_dir_new = os.path.join(dataset_dir, f"train_split_{num_clients}")
    for i in range(num_clients):
        client_path = os.path.join(train_dir_new, f'c{i}')
        train_dataset_raw = tf.keras.utils.text_dataset_from_directory(client_path, batch_size=batch_size)#, seed=42)
        train_data_list.append([(x, y) for x, y in train_dataset_raw.map(vectorize_text)])

    test_dataset_raw = tf.keras.utils.text_dataset_from_directory(
        'aclImdb/test', batch_size=batch_size)
    test_dataset = [(x, y) for x, y in test_dataset_raw.map(vectorize_text)]

    val_dataset_raw = tf.keras.utils.text_dataset_from_directory(
        'aclImdb/val', batch_size=batch_size)
    val_dataset = [(x, y) for x, y in val_dataset_raw.map(vectorize_text)]

    return train_data_list, val_dataset, test_dataset
