import numpy as np
import torch
from unet.unet import UNet
from randomwalker.RandomWalkerModule import RandomWalker
from data.cremiDataloading import CremiSegmentationDataset
from datapreprocessing.target_sparse_sampling import SparseMaskTransform
from utils.notebookUtils import make_summary_plot, sample_seeds, generate_transforms
from utils.evaluation import compute_iou


image_size = 128         # note that the image shall be square (and preferably even)
seeds_per_region = 5     # number of seeds provided 

raw_transforms, target_transforms = generate_transforms(image_size=image_size)

model_path = "checkpoints/models/last_model_20231219_165700_9"

model = UNet(1, 32, 3)
model.load_state_dict(torch.load(model_path))
model.eval()
rw = RandomWalker(1000, max_backprop=True)

# load a test observation
test_dataset = CremiSegmentationDataset("data/sample_A_20160501.hdf", transform=raw_transforms, target_transform=target_transforms, testing=True)
test_raw, test_segmentation, test_mask = test_dataset[93]

# determine number of segments in the image
num_classes = len(np.unique(test_segmentation.squeeze()))

# sample some amount of seeds from each segment
seeds = sample_seeds(seeds_per_region, test_segmentation, num_classes)

# compute UNet output
net_output = model(test_raw.unsqueeze(0))
diffusivities = torch.sigmoid(net_output)
# RW given the diffusivities from UNet
output = rw(diffusivities, seeds)

# compute fit metrics (mIoU)
pred_masks = torch.argmax(output[0], dim=1)
iou_score = compute_iou(pred_masks.detach().cpu(), test_segmentation.detach().cpu(), num_classes)

make_summary_plot(0, test_raw, output, diffusivities, seeds, test_segmentation, iou_score, model_subsampling_ratio=0.01)