from randomwalker.RandomWalkerModule import RandomWalker
import torch
from datapreprocessing.target_sparse_sampling import SparseMaskTransform
import matplotlib.pyplot as plt
import numpy as np
from randomwalker.randomwalker_loss_utils import NHomogeneousBatchLoss
from unet.unet import UNet
import os

if not os.path.exists('results'):
    os.makedirs('results')


def make_summary_plot(it, raw, output, net_output, seeds, target, subsampling_ratio):
    """
    This function create and save a summary figure
    """
    f, axarr = plt.subplots(2, 2, figsize=(8, 9.5))
    f.suptitle("RW summary, Iteration: " + repr(it))

    axarr[0, 0].set_title("Ground Truth Image")
    axarr[0, 0].imshow(raw[0].detach().numpy(), cmap="gray")
    axarr[0, 0].imshow(target[0, 0].detach().numpy(), alpha=0.6, vmin=-3, cmap="prism_r")
    seeds_listx, seeds_listy = np.where(seeds[0].data != 0)
    axarr[0, 0].scatter(seeds_listy,
                        seeds_listx, c="r")
    axarr[0, 0].axis("off")

    axarr[0, 1].set_title("LRW output (white seed)")
    axarr[0, 1].imshow(raw[0].detach().numpy(), cmap="gray")
    axarr[0, 1].imshow(np.argmax(output[0][0].detach().numpy(), 0), alpha=0.6, vmin=-3, cmap="prism_r")
    axarr[0, 1].axis("off")

    axarr[1, 0].set_title("Vertical Diffusivities")
    axarr[1, 0].imshow(net_output[0, 0].detach().numpy(), cmap="gray")
    axarr[1, 0].axis("off")

    axarr[1, 1].set_title("Horizontal Diffusivities")
    axarr[1, 1].imshow(net_output[0, 1].detach().numpy(), cmap="gray")
    axarr[1, 1].axis("off")

    plt.tight_layout()
    if not os.path.exists(f"results/{subsampling_ratio}/"):
        os.makedirs(f"results/{subsampling_ratio}/")
    plt.savefig(f"./results/{subsampling_ratio}/{it}.png")
    plt.close()


if __name__ == '__main__':
    # Init parameters
    batch_size = 1
    iterations = 50
    size = (128, 128)
    datadir = "data/"

    subsampling_ratios = [1.0, 0.5, 0.1, 0.05, 0.01]
    for subsampling_ratio in subsampling_ratios:
        # Init the UNet
        unet = UNet(1, 32, 2)

        # Init the random walker modules
        rw = RandomWalker(1000, max_backprop=False)

        # Load data and init
        raw = torch.load(datadir + "raw.pytorch")
        target = torch.load(datadir + "target.pytorch")
        seeds = torch.load(datadir + "seeds.pytorch")

        # Init optimizer
        optimizer = torch.optim.Adam(unet.parameters(), lr=0.01)

        # Loss has to been wrapped in order to work with random walker algorithm
        loss = NHomogeneousBatchLoss(torch.nn.NLLLoss)

        # Main overfit loop
        for it in range(iterations + 1):
            optimizer.zero_grad()

            diffusivities = unet(raw.unsqueeze(0))

            # Diffusivities must be positive
            net_output = torch.sigmoid(diffusivities)

            # Random walker
            output = rw(net_output, seeds)

            # Define sparse mask for the loss
            # mask_x, mask_y = torch.bernoulli(subsampling_ratio * torch.ones(128, 128)).nonzero(as_tuple=True)
            mask_x, mask_y = SparseMaskTransform(subsampling_ratio = subsampling_ratio)(target.squeeze()).nonzero(as_tuple=True)

            # Loss and diffusivities update
            output_log = [torch.log(o)[:, :, mask_x, mask_y] for o in output]

            l = loss(output_log, target[:, :, mask_x, mask_y])
            l.backward(retain_graph=True)
            optimizer.step()

            # Summary
            if it % 5 == 0:
                print("Iteration: ", it, " Loss: ", l.item())
                make_summary_plot(it, raw, output, net_output, seeds, target, subsampling_ratio)
