import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from data.cremi_dataloading import CremiSegmentationDataset

import matplotlib.pyplot as plt

# loading in dataset
train_dataset = CremiSegmentationDataset("data/sample_A_20160501.hdf")

# setting up dataloader (should give a length of 5, 125 / 25)
# shuffle should be okay, since the segmentation & raw are tupled now
train_dataloader = DataLoader(train_dataset, batch_size=25, shuffle=True)

print(f"Dataloader should have a length of 125 / 25 = 5: {len(train_dataloader)}")

# sanity check
figure = plt.figure(figsize=(8, 8))
cols, rows = 2, 2
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_dataset), size=(1,)).item()
    img, segmentation = train_dataset[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
    plt.imshow(segmentation.squeeze(), alpha=0.4, vmin=-3, cmap="prism_r")
plt.show()