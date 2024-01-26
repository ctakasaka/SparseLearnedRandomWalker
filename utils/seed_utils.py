import random
import numpy as np
import torch


def set_seeds(seed):
    """
    Set seeds for random, numpy, and PyTorch libraries.

    Parameters:
    - seed (int): Seed value for reproducibility.
    """
    # Set seed for the random library
    random.seed(seed)

    # Set seed for the numpy library
    np.random.seed(seed)

    # Set seed for the PyTorch library
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using GPU


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


def get_all_seeds(target, mask_x, mask_y):
    seeds = torch.zeros_like(target.squeeze())
    target_seeds = target.squeeze()[mask_x, mask_y] + 1
    seeds[mask_x, mask_y] = target_seeds
    seeds = seeds.unsqueeze(0)
    return seeds
