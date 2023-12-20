import numpy as np
import torch
from torchvision import transforms
from unet.unet import UNet
from randomwalker.RandomWalkerModule import RandomWalker
from data.cremiDataloading import CremiSegmentationDataset
from datapreprocessing.target_sparse_sampling import SparseMaskTransform

from randomwalker.randomwalker_loss_utils import NHomogeneousBatchLoss
from utils.evaluation import compute_iou
from typing import Dict
import matplotlib.pyplot as plt

from notebookUtils import make_summary_plot, sample_seeds

subsampling_ratio = 0.01
seeds_per_region = 5

model_path = "checkpoints/models/last_model_20231219_165700_9"

model = UNet(1, 32, 3)
model.load_state_dict(torch.load(model_path))
model.eval()
rw = RandomWalker(1000, max_backprop=True)

raw_transforms = transforms.Compose([
    transforms.Normalize(mean=[0.5], std=[0.5]),
    transforms.FiveCrop(size=(512, 512)),
])
target_transforms = transforms.Compose([
    transforms.FiveCrop(size=(512, 512)),
])

# load a test observation
test_dataset = CremiSegmentationDataset("data/sample_A_20160501.hdf", transform=raw_transforms, target_transform=target_transforms, testing=True)
test_raw, test_segmentation, test_mask = test_dataset[93]
num_classes = len(np.unique(test_segmentation.squeeze()))

diffusivities = model(test_raw.unsqueeze(0))
# Diffusivities must be positive
net_output = torch.sigmoid(diffusivities)

mask = SparseMaskTransform(subsampling_ratio)(test_segmentation.squeeze())
mask_x, mask_y = mask.nonzero(as_tuple=True)
masked_targets = test_segmentation.squeeze()[mask_x, mask_y]

seeds = sample_seeds(seeds_per_region, test_segmentation, masked_targets, mask_x, mask_y, num_classes)
output = rw(net_output, seeds)

make_summary_plot(0, test_raw, output, net_output, seeds, test_segmentation, mask, subsampling_ratio, 0)