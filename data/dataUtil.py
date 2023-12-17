import numpy as np
import torch
import random
import xxhash

class Volume:

    def __init__(self, data, resolution = (1.0, 1.0, 1.0), offset = (0.0, 0.0, 0.0), comment = ""):
        self.data = data
        self.resolution = resolution
        self.offset = offset
        self.comment = comment

    def __getitem__(self, location):
        """Get the closest value of this volume to the given location. The 
        location is in world units, relative to the volumes offset.

        This method takes into account the resolution of the volume. An 
        IndexError exception is raised if the location is not contained in this 
        volume.

        To access the raw pixel values, use the `data` attribute.
        """

        i = tuple([ round(location[d]/self.resolution[d]) for d in range(len(location)) ])

        if min(i) >= 0:
            try:
                return self.data[i]
            except IndexError as e:
                raise IndexError("location " + str(location) + " does not lie inside volume: " + str(e))

        raise IndexError("location " + str(location) + " does not lie inside volume")


def generate_sparse_masks(gt_segmentation, subsampling_ratio=0.1):

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
    # can generate an arbitrary int here
    np_seed = random.randrange(0, 1024, 1)
    np.random.seed(np_seed)
    # sample and create the mask
    sampled_indices = np.random.choice(np.arange(sparse_target.shape[1]), 
                               replace=False, 
                               size=int(sparse_target.shape[1] * subsampling_ratio))
    sparse_target[idx][sampled_indices] = 1

  # fix mask shapes & cast to tensor
  return torch.from_numpy((sparse_target.reshape(data_shape)).astype(np.int64)).to(torch.int8)