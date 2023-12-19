from randomwalker.RandomWalkerModule import RandomWalker
import torch
from datapreprocessing.target_sparse_sampling import SparseMaskTransform
import matplotlib.pyplot as plt
import numpy as np
from randomwalker.randomwalker_loss_utils import NHomogeneousBatchLoss
from unet.unet import UNet
from utils.evaluation import compute_iou
import time
import os

if not os.path.exists('results'):
    os.makedirs('results')


def make_summary_plot(experiment_name, it, raw, output, net_output, seeds, target, mask, subsampling_ratio):
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

    mask_x, mask_y = np.where(mask != 0)
    axarr[0, 0].scatter(mask_y,
                        mask_x, c="b", s=0.5, marker='o')
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
    if not os.path.exists(f"results/{experiment_name}/{subsampling_ratio}/"):
        os.makedirs(f"results/{experiment_name}/{subsampling_ratio}/")
    plt.savefig(f"./results/{experiment_name}/{subsampling_ratio}/{it}.png")
    plt.close()

def sample_seeds(seeds_per_region, target, masked_target, mask_x, mask_y, num_classes):
    seeds = torch.zeros_like(target.squeeze())
    if seeds_per_region == 1:
        seed_indices = np.array([
            np.random.choice(np.where(masked_target == i)[0]) for i in range(num_classes)
        ])
    else:
        num_seeds = [
            min(len(np.where(masked_target == i)[0]), seeds_per_region) for i in range(num_classes)
        ]
        seed_indices = np.concatenate([
            np.random.choice(np.where(masked_target == i)[0], num_seeds[i], replace=False)
            for i in range(num_classes)
        ])
    target_seeds = target.squeeze()[mask_x[seed_indices], mask_y[seed_indices]] + 1
    seeds[mask_x[seed_indices], mask_y[seed_indices]] = target_seeds
    seeds = seeds.unsqueeze(0)
    return seeds


if __name__ == '__main__':
    # Experiment parameters
    random_seeding = False
    seeds_per_region = 1
    accumulate_iterations = 1
    pruned_gradients = False
    experiment_name = "fixed-seeding"
    if not os.path.exists(f"results/{experiment_name}"):
        os.makedirs(f"results/{experiment_name}")
    log_file = open(f"results/{experiment_name}/log.txt", "a+")
    print(experiment_name, file=log_file)

    # Init parameters
    batch_size = 1
    iterations = 50
    size = (128, 128)
    datadir = "data/"

    # Load data and init
    raw = torch.load(datadir + "raw.pytorch")
    target = torch.load(datadir + "target.pytorch")
    seeds = torch.load(datadir + "seeds.pytorch")

    print(raw.shape, target.shape)

    num_classes = len(np.unique(target))

    subsampling_ratios = [0.01, 0.1, 0.5]
    for subsampling_ratio in subsampling_ratios:
        print(f"\nSubsampling ratio: {subsampling_ratio}", file=log_file)
        # Init the UNet
        unet = UNet(1, 32, 2)

        # Init the random walker modules
        rw = RandomWalker(1000, max_backprop=pruned_gradients)

        # Init optimizer
        optimizer = torch.optim.Adam(unet.parameters(), lr=0.01)

        # Loss has to been wrapped in order to work with random walker algorithm
        loss = NHomogeneousBatchLoss(torch.nn.NLLLoss)

        # Define sparse mask for the loss
        valid_mask = False
        while not valid_mask:
            valid_mask = True
            mask = SparseMaskTransform(subsampling_ratio=subsampling_ratio)(target.squeeze())
            mask_x, mask_y = mask.nonzero(as_tuple=True)
            masked_target = target.squeeze()[mask_x, mask_y]

            for i in range(num_classes):
                labels_in_region = len(np.where(masked_target == i)[0])
                if labels_in_region == 0:
                    valid_mask = False

        # Main overfit loop
        total_time = 0.0
        num_it = 0
        for it in range(iterations + 1):
            t1 = time.time()
            optimizer.zero_grad()

            diffusivities = unet(raw.unsqueeze(0))

            # Diffusivities must be positive
            net_output = torch.sigmoid(diffusivities)

            avg_loss = torch.tensor(0.0)
            for k in range(accumulate_iterations):
                # Sample random seeds
                if random_seeding:
                    seeds = sample_seeds(seeds_per_region, target, masked_target, mask_x, mask_y, num_classes)

                # Random walker
                output = rw(net_output, seeds)

                # Loss and diffusivities update
                output_log = [torch.log(o)[:, :, mask_x, mask_y] for o in output]

                l = loss(output_log, target[:, :, mask_x, mask_y])
                avg_loss += l

            avg_loss /= accumulate_iterations
            avg_loss.backward(retain_graph=True)
            optimizer.step()

            t2 = time.time()
            total_time += t2-t1
            num_it += 1

            # Summary
            if it % 5 == 0:
                pred = torch.argmax(output[0], dim=1)
                iou_score = compute_iou(pred, target[0], num_classes)
                avg_time = total_time / num_it
                print(f"Iteration {it}  Time/iteration(s): {avg_time}  Loss: {avg_loss.item()}  mIoU: {iou_score}",
                      file=log_file)
                make_summary_plot(experiment_name, it, raw, output, net_output, seeds, target, mask, subsampling_ratio)
                total_time = 0.0
                num_it = 0
        print("\n")
    os.system(f"say 'Finished running experiment {experiment_name}'")
