import numpy as np
import torch
import random
import xxhash
import pickle

def generateSparseMasks(gt_segmentation, subsampling_ratio=0.1):

  # flatten each observation in the dataset
  data_shape = gt_segmentation.shape
  flat_mask = gt_segmentation.reshape((data_shape[0], -1))

  # calculate the required number of gt pixels
  num_elements = flat_mask.shape[1]
  num_samples = int(subsampling_ratio * num_elements)

  # sample pixels for each observation
  # yikes, for-loop here to allow resetting of random seed each time...
  sparse_target = np.zeros_like(flat_mask)
  for idx,segmentation in enumerate(flat_mask):
    # get (sufficiently) unique hash for observation
    obs_hash = xxhash.xxh3_64(segmentation.tobytes()).hexdigest().upper()
    # update random seed to get same pixel mask every time
    random.seed(obs_hash)
    # sample and create the mask
    sampled_indices = random.sample(range(num_elements), num_samples)
    sparse_target[idx][sampled_indices] = 1

  # fix mask shapes & cast to tensor
  return torch.from_numpy((sparse_target.reshape(data_shape)).astype(np.int64)).to(torch.int8)