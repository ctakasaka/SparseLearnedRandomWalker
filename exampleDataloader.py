import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from data.cremiDataloading import CremiSegmentationDataset

import matplotlib.pyplot as plt
from exampleCREMI import make_summary_plot, sample_seeds


target = torch.load("data/" + "target.pytorch")
print(target.unique())

# transforms
raw_transforms = transforms.Compose([
    transforms.Normalize(mean=[0.5], std=[0.5]),
    transforms.FiveCrop(size=(256, 256)),
])
target_transforms = transforms.Compose([
    transforms.FiveCrop(size=(256, 256)),
])

# loading in dataset
train_dataset = CremiSegmentationDataset("data/sample_A_20160501.hdf", transform=raw_transforms, target_transform=target_transforms, subsampling_ratio=0.1, split="test")

# setting up dataloader
train_dataloader = DataLoader(train_dataset, batch_size=25, shuffle=True)

batch = next(iter(train_dataloader))
batch_subseg = (batch[1] * batch[2])

# sanity check
figure = plt.figure(figsize=(8, 8))
cols, rows = 2, 2
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_dataloader), size=(1,)).item()

    img = batch[0][sample_idx]
    segmentation = batch[1][sample_idx]
    # mask the ground truth

    figure.add_subplot(rows, cols, i)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
    plt.imshow(segmentation.squeeze(), alpha=0.4, vmin=-3, cmap="prism_r")
    ## can see the subsampled mask with this line
    # plt.imshow(mask.squeeze(), alpha=0.6, vmin=-3, cmap="prism_r")
plt.show()