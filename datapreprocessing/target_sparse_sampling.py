import torch
import random

class SparseMaskTransform(object):
    def __init__(self, subsampling_ratio=0.1):
        self.subsampling_ratio = subsampling_ratio
        

    def __call__(self, target_mask):

        # Convert the target mask to a PyTorch tensor if it's not already
        if not isinstance(target_mask, torch.Tensor):
            target_mask = torch.tensor(target_mask, dtype=torch.int)

        # Flatten the mask
        flat_mask = target_mask.ravel()

        # Create a sparse mask by subsampling
        num_elements = flat_mask.shape[0]
        num_samples = int(self.subsampling_ratio * num_elements)
        sampled_indices = random.sample(range(num_elements), num_samples)
        sparse_target = torch.zeros_like(flat_mask)
        sparse_target[sampled_indices] = 1

        return sparse_target.view(target_mask.size())


