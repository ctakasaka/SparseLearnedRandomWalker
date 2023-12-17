import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from data.cremi_dataloading import CremiSegmentationDataset

import matplotlib.pyplot as plt

# transforms
raw_transforms = transforms.Compose([
    # transforms.ToTensor(),
    transforms.CenterCrop(size=(256, 256)),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])
target_transforms = transforms.Compose([
    # transforms.ToTensor(),
    transforms.CenterCrop(size=(256, 256)),
])

# loading in dataset
train_dataset = CremiSegmentationDataset("data/sample_A_20160501.hdf", transform=raw_transforms, target_transform=target_transforms)

# setting up dataloader (should give a length of 5, 125 / 25)
# shuffle should be okay, since the segmentation & raw are tupled now
train_dataloader = DataLoader(train_dataset, batch_size=25, shuffle=True)

print(f"Dataloader should have a length of 125 / 25: {len(train_dataloader)}")

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

# printing size of raw data tensor from dataloader batch
print(next(iter(train_dataloader))[0].shape)