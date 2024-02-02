import os
import numpy as np
import matplotlib.pyplot as plt


def plot_ground_truth(ax, raw, target, seeds, mask=None):
    ax.imshow(raw[0].detach().numpy(), cmap="gray")
    ax.imshow(target[0, 0].detach().numpy(), alpha=0.6, vmin=-3, cmap="prism_r")
    seeds_listx, seeds_listy = np.where(seeds[0].data != 0)
    ax.scatter(seeds_listy, seeds_listx, c="r")
    if mask is not None:
        mask_x, mask_y = np.where(mask != 0)
        ax.scatter(mask_y, mask_x, c="b", s=0.5, marker='o')
    ax.axis("off")


def plot_predictions(ax, raw, pred):
    ax.imshow(raw[0].detach().numpy(), cmap="gray")
    ax.imshow(pred[0], alpha=0.6, vmin=-3, cmap="prism_r")
    ax.axis("off")


def plot_vertical_diffusivities(ax, diffusivities):
    ax.imshow(diffusivities[0, 0].detach().numpy(), cmap="gray")
    ax.axis("off")


def plot_horizontal_diffusivities(ax, diffusivities):
    ax.imshow(diffusivities[0, 1].detach().numpy(), cmap="gray")
    ax.axis("off")


def get_summary_fig(raw, target, seeds, diffusivities, pred, mask=None, figsize=(8, 9.5)):
    f, axarr = plt.subplots(2, 2, figsize=(8, 9.5))

    # Plot ground truth image + mask + seeds
    axarr[0, 0].set_title("Ground Truth + Mask + Seeds")
    plot_ground_truth(axarr[0, 0], raw, target, seeds, mask)

    # Plot pred segmentation
    axarr[0, 1].set_title("SLRW Segmentation Output")
    plot_predictions(axarr[0, 1], raw, pred)

    # Plot vertical diffusivities
    axarr[1, 0].set_title("Vertical Diffusivities")
    plot_vertical_diffusivities(axarr[1, 0], diffusivities)

    # Plot horizontal diffusivities
    axarr[1, 1].set_title("Horizontal Diffusivities")
    plot_horizontal_diffusivities(axarr[1, 1], diffusivities)

    return f, axarr


def save_summary_plot(raw, target, seeds, diffusivities, pred, mask, subsampling_ratio,
                      it, iou_score, save_path):
    """
    This function creates and saves a summary figure
    """
    f, axarr = get_summary_fig(raw, target, seeds, diffusivities, pred, mask)
    f.suptitle(f"SLRW summary - Subsampling ratio: {subsampling_ratio} - Iteration: {it} - mIoU: {iou_score}")

    plt.tight_layout()
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(f"{save_path}/{it}.png")
    plt.close()
