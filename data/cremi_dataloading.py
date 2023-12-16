from cremi.io import CremiFile

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# raw & segmentation combined dataset object
class CremiSegmentationDataset(Dataset):

  def __init__(self, cremi_location, transform=None, target_transform=None):
    cremi_file = CremiFile(cremi_location, "r")
    self.raw = torch.from_numpy(np.array(cremi_file.read_raw().data)).to(torch.float64)
    self.seg = torch.from_numpy(np.array(cremi_file.read_neuron_ids().data).astype(np.int64))

    self.transform = transform
    self.target_transform = target_transform

  def __len__(self):
    return len(self.raw.data)
  
  def __getitem__(self, idx):

    raw_neurons = self.raw[idx].unsqueeze(0)
    seg_neurons = self.seg[idx].unsqueeze(0)

    if self.transform:
      raw_neurons = self.transform(raw_neurons)
    if self.target_transform:
      seg_neurons = self.target_transform(seg_neurons)

    return raw_neurons, seg_neurons
  
