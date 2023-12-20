import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms

def make_summary_plot(it, raw, output, net_output, seeds, target, miou, model_subsampling_ratio):
        """
        This function create and save a summary figure
        """
        f, axarr = plt.subplots(2, 2, figsize=(8, 9.5))
        f.suptitle(f"Model Pretrained on {model_subsampling_ratio} Subsampling")

        axarr[0, 0].set_title("Ground Truth Image")
        axarr[0, 0].imshow(raw[0].detach().numpy(), cmap="gray")
        axarr[0, 0].imshow(target[0].detach().numpy(), alpha=0.6, vmin=-3, cmap="prism_r")
        seeds_listx, seeds_listy = np.where(seeds[0].data != 0)
        axarr[0, 0].scatter(seeds_listy,
                            seeds_listx, c="r")
        axarr[0, 0].axis("off")

        axarr[0, 1].set_title(f"SLRW Prediction\nMean IoU: {round(miou, 4)}")
        axarr[0, 1].imshow(raw[0].detach().numpy(), cmap="gray")
        axarr[0, 1].imshow(np.argmax(output[0][0].detach().numpy(), 0), alpha=0.6, vmin=-3, cmap="prism_r")
        axarr[0, 1].axis("off")

        axarr[1, 0].set_title("Vertical Diffusivities (UNet)")
        axarr[1, 0].imshow(net_output[0, 0].detach().numpy(), cmap="gray")
        axarr[1, 0].axis("off")

        axarr[1, 1].set_title("Horizontal Diffusivities (UNet)")
        axarr[1, 1].imshow(net_output[0, 1].detach().numpy(), cmap="gray")
        axarr[1, 1].axis("off")

        plt.tight_layout()
        plt.show()
        # if not os.path.exists(f"results-full/{subsampling_ratio}/epoch-{epoch_index}/"):
        #     os.makedirs(f"results-full/{subsampling_ratio}/epoch-{epoch_index}/")
        # plt.savefig(f"./results-full/{subsampling_ratio}/epoch-{epoch_index}/{it}.png")
        # plt.close()

def sample_seeds(seeds_per_region, target, num_classes):
        
        target_temp = target.squeeze()
        flat_target = target_temp.reshape((-1))
        full_mask = torch.ones_like(target_temp)
        mask_x, mask_y = full_mask.nonzero(as_tuple=True)

        seeds = torch.zeros_like(target_temp)
        if seeds_per_region == 1:
            seed_indices = np.array([
                np.random.choice(np.where(flat_target == i)[0]) for i in range(num_classes)
            ])
        else:
            num_seeds = [
                min(len(np.where(flat_target == i)[0]), seeds_per_region) for i in range(num_classes)
            ]
            seed_indices = np.concatenate([
                np.random.choice(np.where(flat_target == i)[0], num_seeds[i], replace=False)
                for i in range(num_classes)
            ])
        target_seeds = target.squeeze()[mask_x[seed_indices], mask_y[seed_indices]] + 1
        seeds[mask_x[seed_indices], mask_y[seed_indices]] = target_seeds
        seeds = seeds.unsqueeze(0)
        return seeds

def bsdfsample_seeds(seeds_per_region, target, masked_target, mask_x, mask_y, num_classes):
        seeds = torch.zeros_like(target.squeeze())
        if seeds_per_region == 1:
            seed_indices = np.array([
                np.random.choice(np.where(target == i)[0]) for i in range(num_classes)
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

def generate_transforms(image_size=128):
        
        if image_size < 128:
            print(f"Image size of {image_size} too small. Setting to 128x128.")
            image_size = 128
        elif image_size > 512:
            print(f"Image size of {image_size} too large. Setting to 512x512.")
            image_size = 512
        elif image_size % 2 != 0:
            image_size -= 1

        raw_transforms = transforms.Compose([
            transforms.Normalize(mean=[0.5], std=[0.5]),
            transforms.FiveCrop(size=(image_size, image_size)),
        ])
        target_transforms = transforms.Compose([
            transforms.FiveCrop(size=(image_size, image_size)),
        ])

        return raw_transforms, target_transforms