from cremi.io import CremiFile
from data_util import generateSparseMasks

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import random
import xxhash

def array_id(a, *, include_dtype = False, include_shape = False):
    data = bytes()
    if include_dtype:
        data += str(a.dtype).encode('ascii')
    data += b','
    if include_shape:
        data += str(a.shape).encode('ascii')
    data += b','
    data += a.tobytes()

    return xxhash.xxh3_64(data).hexdigest().upper()

cremi_file = CremiFile("sample_A_20160501.hdf", "r")
segmentation = cremi_file.read_neuron_ids()

blah = np.array(cremi_file.read_neuron_ids().data).astype(np.int64)
print(blah.shape[0])
masks = generateSparseMasks(blah, 0.1)
print(masks.shape)