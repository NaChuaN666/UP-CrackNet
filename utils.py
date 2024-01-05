import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
import cv2 as cv
from numpy.linalg import norm

# For logger
def to_np(x):
    return x.data.cpu().numpy()


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


# De-normalization
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


# Plot losses
def plot_loss(d_losses, g_losses, num_epochs, save=False, save_dir='results/', show=False):
    fig, ax = plt.subplots()
    ax.set_xlim(0, num_epochs)
    ax.set_ylim(0, max(np.max(g_losses), np.max(d_losses))*1.1)
    plt.xlabel('# of Epochs')
    plt.ylabel('Loss values')
    plt.plot(d_losses, label='Discriminator')
    plt.plot(g_losses, label='Generator')
    plt.legend()

    # save figure
    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_fn = save_dir + 'Loss_values_epoch_{:d}'.format(num_epochs) + '.png'
        plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()


def plot_test_result(input, target, gen_image, epoch, training=True, save=False, save_dir='results/', show=False, fig_size=(5, 5)):
    if not training:
        fig_size = (input.size(2) * 3 / 100, input.size(3)/100)
    # print(input.shape)
    # print("fig_size is {}".format(fig_size))
    # exit()

    fig, axes = plt.subplots(1, 3, figsize=fig_size)
    imgs = [input, gen_image, target]
    for ax, img in zip(axes.flatten(), imgs):
        ax.axis('off')
        # ax.set_adjustable('box-forced')
        # Scale to 0-255
        img = (((img[0] - img[0].min()) * 255) / (img[0].max() - img[0].min())).numpy().transpose(1, 2, 0).astype(np.uint8)
        # print(img)
        ax.imshow(img, cmap=None, aspect='equal')
    plt.subplots_adjust(wspace=0, hspace=0)

    # print(imgs[0])
    # print(imgs[1])
    # print(imgs[2])

    if training:
        title = 'Epoch {0}'.format(epoch + 1)
        fig.text(0.5, 0.04, title, ha='center')

    # save figure
    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        if training:
            save_fn = save_dir + 'Result_epoch_{:d}'.format(epoch+1) + '.png'
        else:
            save_fn = save_dir + 'Test_result_{:d}'.format(epoch+1) + '.png'
            fig.subplots_adjust(bottom=0)
            fig.subplots_adjust(top=1)
            fig.subplots_adjust(right=1)
            fig.subplots_adjust(left=0)
        plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


def save_error_maps_gray(input_name, label, input_ori, target, gen_image,
                                  epoch, save=False, save_dir='output_results/'):

    name = input_name.split(".")[0]

    gen_image_show = (((gen_image[0] - gen_image[0].min()) * 255) / (
                 gen_image[0].max() - gen_image[0].min())).numpy().transpose(1, 2, 0).astype(np.uint8)
    input_image_show = (((input_ori[0] - input_ori[0].min()) * 255) / (
                 input_ori[0].max() - input_ori[0].min())).numpy().transpose(1, 2, 0).astype(np.uint8)

    input = input_ori[0].numpy().transpose(1, 2, 0)
    input_gray = cv.cvtColor(input, cv.COLOR_BGR2GRAY)
    input_gray = (input_gray - input_gray.min()) / (input_gray.max() - input_gray.min())

    gen_image = gen_image[0].numpy().transpose(1, 2, 0)
    gen_image_gray = cv.cvtColor(gen_image, cv.COLOR_BGR2GRAY)
    gen_image_gray = (gen_image_gray - gen_image_gray.min()) / (gen_image_gray.max() - gen_image_gray.min())

    target = (((target[0] - target[0].min()) * 255) / (target[0].max() - target[0].min())).numpy().transpose(1, 2,
                                                                                                             0)
    label = (((label[0] - label[0].min()) * 255) / (label[0].max() - label[0].min())).numpy().transpose(1, 2, 0)
    label = cv.cvtColor(label, cv.COLOR_BGR2GRAY)

    error_map = np.absolute(input_gray - gen_image_gray)
   
    error_map = (error_map - np.min(error_map)) / (np.max(error_map) - np.min(error_map))
    error_map = (error_map * 255).astype(np.uint8)

    input_gray = (input_gray * 255).astype(np.uint8)
    gen_image_gray = (gen_image_gray * 255).astype(np.uint8)

    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        cv.imwrite(save_dir + '{:s}'.format(name) + '.png', error_map)


# Make gif
def make_gif(dataset, num_epochs, save_dir='results/'):
    gen_image_plots = []
    for epoch in range(num_epochs):
        # plot for generating gif
        save_fn = save_dir + 'Result_epoch_{:d}'.format(epoch + 1) + '.png'
        gen_image_plots.append(imageio.imread(save_fn))

    imageio.mimsave(save_dir + dataset + '_pix2pix_epochs_{:d}'.format(num_epochs) + '.gif', gen_image_plots, fps=5)

