import h5py
from .dataUtil import generate_sparse_masks, Volume

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# raw & segmentation combined dataset object
class CremiSegmentationDataset(Dataset):

  def __init__(self, cremi_location, transform=None, target_transform=None, subsampling_ratio=0.1, testing=False):

    self.testing = testing
    self.transform = transform
    self.target_transform = target_transform

    cremi_hdf = h5py.File(cremi_location, "r")
    raw_dataset = Volume(cremi_hdf["/volumes/raw"])
    self.raw = torch.from_numpy(np.array(raw_dataset.data)).unsqueeze(1).to(torch.float)

    seg_dataset = Volume(cremi_hdf["/volumes/labels/neuron_ids"])
    self.seg = np.array(seg_dataset.data).astype(np.int64)
    self.seg = torch.from_numpy(self.seg)

    crop_idx = 4
    if not self.testing:
      # rng = np.random.default_rng()
      # crop_idx = rng.integers(4, size=1)[0]
      crop_idx = 2

    if self.transform:
      self.raw = self.transform(self.raw)
      self.raw = self.raw[crop_idx]
    if self.target_transform:
      self.seg  = self.target_transform(self.seg)
      self.seg  = self.seg[crop_idx]

    self.seg = self.seg.numpy().astype(np.int64)
    # maybe temporary, maybe forever
    self.mask = generate_sparse_masks(self.seg, subsampling_ratio).unsqueeze(1)
    # now cast segmentation truths to tensor
    self.seg = torch.from_numpy(self.seg)

    # re-value segmentation targets for efficiency
    for idx in range(self.seg.shape[0]):
        _, inverse_indices = torch.unique(self.seg[idx], sorted=True, return_inverse=True)
        self.seg[idx] = inverse_indices
      
    self.seg = self.seg.unsqueeze(1)

    # close hdf file
    cremi_hdf.close()

  def __len__(self):
    return len(self.raw.data)
  
  def __getitem__(self, idx):

    raw_neurons = self.raw[idx]
    seg_neurons = self.seg[idx]
    sample_mask = self.mask[idx]

    return raw_neurons, seg_neurons, sample_mask
  
