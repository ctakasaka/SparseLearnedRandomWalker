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