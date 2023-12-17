from cremi.io import CremiFile
from .data_util import generateSparseMasks

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# raw & segmentation combined dataset object
class CremiSegmentationDataset(Dataset):

  def __init__(self, cremi_location, transform=None, target_transform=None, subsampling_ratio=0.1):
    cremi_file = CremiFile(cremi_location, "r")
    self.raw = torch.from_numpy(np.array(cremi_file.read_raw().data)).to(torch.float64)
    # must be numpy array to allow translation to byte-string
    self.seg = np.array(cremi_file.read_neuron_ids().data).astype(np.int64)
    # maybe temporary, maybe forever
    self.mask = generateSparseMasks(self.seg, subsampling_ratio)
    # now cast segmentation truths to tensor
    self.seg = torch.from_numpy(self.seg)

    self.transform = transform
    self.target_transform = target_transform

  def __len__(self):
    return len(self.raw.data)
  
  def __getitem__(self, idx):

    raw_neurons = self.raw[idx].unsqueeze(0)
    seg_neurons = self.seg[idx].unsqueeze(0)
    sample_mask = self.mask[idx].unsqueeze(0)

    if self.transform:
      raw_neurons = self.transform(raw_neurons)
    if self.target_transform:
      seg_neurons = self.target_transform(seg_neurons)
      sample_mask = self.target_transform(sample_mask)

    return raw_neurons, seg_neurons, sample_mask
  
