import torch
import random
import numpy as np

class SparseMaskTransform(object):
    def __init__(self, subsampling_ratio=0.1):
        self.subsampling_ratio = subsampling_ratio
        

    def __call__(self, target_mask):

        # Convert the target mask to a PyTorch tensor if it's not already
        if not isinstance(target_mask, torch.Tensor):
            target_mask = torch.tensor(target_mask, dtype=torch.int)

        # Flatten the mask
        flat_mask = target_mask.ravel().numpy()
         # Create a sparse mask by subsampling
        sparse_target = np.zeros_like(flat_mask)
        unique_classes = np.unique(flat_mask)
        num_elements = flat_mask.shape[0]
        for class_label in unique_classes:
            class_indices = np.where(flat_mask == class_label)[0]
            if len(class_indices) > 0:
                seed_index = random.choice(class_indices)
                sparse_target[seed_index] = 1
        num_samples = int(self.subsampling_ratio * num_elements) - len(unique_classes)
        
        sampled_indices = np.random.choice(np.arange(sparse_target.shape[0]), 
                               replace=False, 
                               size=num_samples)
        sparse_target[sampled_indices] = 1

        # fix mask shapes & cast to tensor
        return torch.from_numpy((sparse_target.reshape(target_mask.shape)).astype(np.int64)).to(torch.int8)
