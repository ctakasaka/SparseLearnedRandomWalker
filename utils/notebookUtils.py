import numpy as np
import torch
import matplotlib.pyplot as plt

def make_summary_plot(it, raw, output, net_output, seeds, target, mask, subsampling_ratio, epoch_index):
        """
        This function create and save a summary figure
        """
        f, axarr = plt.subplots(2, 2, figsize=(8, 9.5))
        f.suptitle("RW summary, Iteration: " + repr(it))

        axarr[0, 0].set_title("Ground Truth Image")
        axarr[0, 0].imshow(raw[0].detach().numpy(), cmap="gray")
        axarr[0, 0].imshow(target[0].detach().numpy(), alpha=0.6, vmin=-3, cmap="prism_r")
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
        plt.show()
        # if not os.path.exists(f"results-full/{subsampling_ratio}/epoch-{epoch_index}/"):
        #     os.makedirs(f"results-full/{subsampling_ratio}/epoch-{epoch_index}/")
        # plt.savefig(f"./results-full/{subsampling_ratio}/epoch-{epoch_index}/{it}.png")
        # plt.close()

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