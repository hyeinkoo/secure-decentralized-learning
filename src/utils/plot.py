import matplotlib.pyplot as plt
import numpy as np
import logging


def plot_images(img_data, save=False, run=None):
    num_row = 2
    num_col = 10
    num_samples = num_row * num_col
    shape = (32, 32, 3)

    # turn off warning messages
    logger = logging.getLogger()
    old_level = logger.level
    logger.setLevel(100)

    fig, axes = plt.subplots(num_row, num_col, figsize=(1. * num_col, 1. * num_row))
    for i, ax in enumerate(axes.flat):
        ax.set_axis_off()
        ax.imshow(np.array(img_data[i]).reshape(shape))
    logger.setLevel(old_level)

    if save:
        # fig.suptitle("n-10_d-30_a-T_e-3_i-2_c-7", y=1.05)
        plt.savefig(f'output/{run}.png', bbox_inches='tight')


def plot_images_2(images, save=False, run=None, show=False):
    original, reconst, reconst_rgm = images

    num_row = 2 * 3
    num_col = 10
    num_samples = num_row * num_col
    shape = (32, 32, 3)

    # turn off warning messages
    logger = logging.getLogger()
    old_level = logger.level
    logger.setLevel(100)

    fig, axes = plt.subplots(num_row, num_col, figsize=(1. * num_col, 1. * num_row))
    for i, ax in enumerate(axes.flat):
        ax.set_axis_off()
        if i < 20:
            img_idx = i
            img = original
        elif i < 40:
            img_idx = i - 20
            img = reconst
        else:
            img_idx = i - 40
            img = reconst_rgm
        try:
            ax.imshow(np.array(img[img_idx]).reshape(shape))
        except:
            ax.imshow(np.zeros(shape))
    logger.setLevel(old_level)

    if save:
        # fig.suptitle("n-10_d-30_a-T_e-3_i-2_c-7", y=1.05)
        plt.savefig(f'{run}.png', bbox_inches='tight')

    if not show:
        plt.close()

