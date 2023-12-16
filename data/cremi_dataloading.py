from cremi.io import CremiFile

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt

# raw & segmentation combined dataset object
class CremiSegmentationDataset(Dataset):

  def __init__(self, cremi_location, transform=None, target_transform=None):
    cremi_file = CremiFile(cremi_location, "r")
    self.raw = cremi_file.read_raw()
    self.seg = cremi_file.read_neuron_ids()
    self.transform = transform
    self.target_transform = target_transform

  def __len__(self):
    return len(self.raw.data)
  
  def __getitem__(self, idx):
    return self.raw.data[idx], self.seg.data[idx]