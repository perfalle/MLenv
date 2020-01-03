import torch
import numpy as np
from itertools import product
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

def autoencoder_image_sample(dataloader, encoder, decoder, classnames, n_samples_per_class=21):
    #   one     two     three   four
    #   O|L|R   O|L|R   O|L|R   O|L|R
    #   O|L|R   O|L|R   O|L|R   O|L|R
    #   O|L|R   O|L|R   O|L|R   O|L|R
    #   O|L|R   O|L|R   O|L|R   O|L|R
    #   O|L|R   O|L|R   O|L|R   O|L|R
    #   O|L|R   O|L|R   O|L|R   O|L|R
    #  -> Original|Latent|Reconstr.

    tiles = {}
    n_classes = len(classnames)
    for c in range(n_classes):
        # print(type(c), c)
        tiles[c] = []

    enough = 0

    for inputs, labels in dataloader:
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        codes = encoder(inputs)
        outputs = decoder(codes)
        for i, label in enumerate(labels):
            if enough == 10:
                break
            label = label.item()
            # print(type(label), label)
            if len(tiles[label]) < n_samples_per_class:
                tiles[label].append((inputs[i], codes[i], outputs[i]))
                if len(tiles[label]) == n_samples_per_class:
                    enough+=1
                    break
    # for i in tiles:
    #     print(len(tiles[i]))
    assert(all(map(lambda k: len(tiles[k]) == n_samples_per_class , tiles)))


    fig_reconstruction_samples = plt.figure(figsize=(50,50), frameon=False, constrained_layout=False)

    # gridspec inside gridspec
    outer_grid = fig_reconstruction_samples.add_gridspec(n_samples_per_class, n_classes, wspace=0.2, hspace=-0.9)

    for i in range(n_samples_per_class):
        for j in range(n_classes):
            olr_triple = tiles[j][i]
            original_image_chw = olr_triple[0].cpu().detach().numpy()
            latent = olr_triple[1].cpu().detach().numpy().reshape(olr_triple[1].numel())
            a = int(np.ceil(np.sqrt(len(latent) / 2)))
            latent_image = np.zeros(2*a*a)
            latent_image[:len(latent)] = latent
            latent_image = latent_image.reshape((2*a,a))
            reconstr_image_chw = olr_triple[2].cpu().detach().numpy()
            original_image = np.transpose(original_image_chw, (1,2,0))
            # latent_image = np.random.random((8, 4, 3))#olr_triple[1].item()
            reconstr_image = np.transpose(reconstr_image_chw, (1,2,0))
            width_ratios=[original_image.shape[1] / original_image.shape[0],
                          latent_image.shape[1] / latent_image.shape[0],
                          reconstr_image.shape[1] / reconstr_image.shape[0]]
            olr_grid = outer_grid[i, j].subgridspec(1, 3, wspace=0.1, hspace=0.0, width_ratios=width_ratios)
            original_ax = fig_reconstruction_samples.add_subplot(olr_grid[0,0])
            latent_ax = fig_reconstruction_samples.add_subplot(olr_grid[0,1])
            reconstr_ax = fig_reconstruction_samples.add_subplot(olr_grid[0,2])
            original_ax.imshow(original_image)
            latent_ax.imshow(latent_image)
            reconstr_ax.imshow(reconstr_image)
            original_ax.set_xticks([])
            original_ax.set_yticks([])
            latent_ax.set_xticks([])
            latent_ax.set_yticks([])
            reconstr_ax.set_xticks([])
            reconstr_ax.set_yticks([])
            fig_reconstruction_samples.add_subplot(original_ax)
            fig_reconstruction_samples.add_subplot(latent_ax)
            fig_reconstruction_samples.add_subplot(reconstr_ax)

    all_axes = fig_reconstruction_samples.get_axes()

    # show only the outside spines
    for ax in all_axes:
        for sp in ax.spines.values():
            sp.set_visible(False)
        if ax.is_first_row():
            ax.spines['top'].set_visible(True)
        if ax.is_last_row():
            ax.spines['bottom'].set_visible(True)
        if ax.is_first_col():
            ax.spines['left'].set_visible(True)
        if ax.is_last_col():
            ax.spines['right'].set_visible(True)

    # remove white border
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)

    fig_reconstruction_samples.canvas.draw()
    image = np.fromstring(fig_reconstruction_samples.canvas.tostring_rgb(), dtype='uint8')
    width, height = fig_reconstruction_samples.get_size_inches() * fig_reconstruction_samples.get_dpi()
    image = image.reshape((int(width), int(height), 3))
    # image = image.reshape((3, int(width), int(height)))
    # image = np.transpose(image, (2,0,1))

    return image
