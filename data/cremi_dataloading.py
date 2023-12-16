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
    # self.raw = cremi_file.read_raw().data
    # self.seg = cremi_file.read_neuron_ids().data

    self.transform = transform
    self.target_transform = target_transform

  def __len__(self):
    return len(self.raw.data)
  
  def __getitem__(self, idx):

    # may need to cast to tensor earlier, unsure as of now
    # raw_neurons = torch.from_numpy(self.raw.data[idx]).to(torch.float64).unsqueeze(0)
    # seg_neurons = torch.from_numpy(self.seg.data[idx].astype(np.int64)).unsqueeze(0)
    raw_neurons = self.raw[idx].unsqueeze(0)
    seg_neurons = self.seg[idx].unsqueeze(0)

    if self.transform:
      raw_neurons = self.transform(raw_neurons)
    if self.target_transform:
      seg_neurons = self.target_transform(seg_neurons)

    return raw_neurons, seg_neurons
  
