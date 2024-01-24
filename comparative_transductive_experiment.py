import os
import yaml
from argparse import Namespace
import numpy as np
import matplotlib.pyplot as plt

from train import get_base_parser
from transductive_experiment import run_transductive_experiment
from torchvision import transforms
from utils.seed_utils import sample_seeds
from data.cremi_dataloader import CremiSegmentationDataset
from utils.plotting_utils import (plot_ground_truth, plot_horizontal_diffusivities,
                                  plot_vertical_diffusivities, plot_predictions)


def parse_args():
    parser = get_base_parser(description='Train and compare on a single neuronal image')
    parser.add_argument('--image-index', dest='img_idx', type=int, default=0)
    args = parser.parse_args()

    with open(args.config, "r") as stream:
        config_params = yaml.load(stream, Loader=yaml.FullLoader)
        args_dict = vars(args)
        args_dict.update(config_params)
        args = Namespace(**args_dict)

    return args


if __name__ == '__main__':
    args = parse_args()

    save_path = f"comparative-image-{args.img_idx}-seeds-{args.seeds_per_region}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    log_file = open(f"{save_path}/log.txt", "a+")

    # Init parameters
    batch_size = 1
    iterations = args.max_epochs
    size = (args.resolution, args.resolution)
    datadir = "data/"

    # Load data and init
    raw_transforms = transforms.Compose([
        transforms.Normalize(mean=[0.5], std=[0.5]),
        transforms.CenterCrop(size=size),
    ])
    target_transforms = transforms.Compose([
        transforms.CenterCrop(size=size),
    ])

    # loading in dataset
    train_dataset = CremiSegmentationDataset("data/sample_A_20160501.hdf", transform=raw_transforms,
                                             target_transform=target_transforms,
                                             subsampling_ratio=args.subsampling_ratio, split="train")
    raw, target, mask = train_dataset[args.img_idx]
    target = target.unsqueeze(0)
    num_classes = len(np.unique(target))

    # Define sparse mask for the loss
    mask = mask.squeeze()
    mask_x, mask_y = mask.nonzero(as_tuple=True)
    masked_targets = target.squeeze()[mask_x, mask_y]

    # Generate seeds
    seeds = sample_seeds(args.seeds_per_region, target, masked_targets, mask_x, mask_y, num_classes)

    # Create figures for comparative plots
    f_hdiff, axarr_hdiff = plt.subplots(3, 1 + iterations // 5, figsize=(32, 10))
    f_hdiff.suptitle("RW summary - horizontal diffusivities")
    axarr_hdiff[0, 0].set_title("Ground Truth Image")
    axarr_hdiff[1, 0].set_title("Untrained model")
    axarr_hdiff[2, 0].set_title("Pretrained model")

    f_vdiff, axarr_vdiff = plt.subplots(3, 1 + iterations // 5, figsize=(32, 10))
    f_vdiff.suptitle("RW summary - vertical diffusivities")
    axarr_vdiff[0, 0].set_title("Ground Truth Image")
    axarr_vdiff[1, 0].set_title("Untrained model")
    axarr_vdiff[2, 0].set_title("Pretrained model")

    f_pred, axarr_pred = plt.subplots(3, 1 + iterations // 5, figsize=(32, 10))
    f_pred.suptitle("RW summary - predictions")
    axarr_pred[0, 0].set_title("Ground Truth Image")
    axarr_pred[1, 0].set_title("Untrained model")
    axarr_pred[2, 0].set_title("Pretrained model")

    def comparative_summary(raw, seeds, mask, diffusivities, output, target, it, avg_time, l, iou_score,
                                model_path=False):
        print(
            f"Iteration {it}  Time/iteration(s): {avg_time}  Loss: {l.item()}  mIoU: {iou_score}",
            file=log_file)
        plot_ground_truth(axarr_hdiff[0, it // 5], raw, target, seeds, mask)
        plot_ground_truth(axarr_vdiff[0, it // 5], raw, target, seeds, mask)
        plot_ground_truth(axarr_pred[0, it // 5], raw, target, seeds, mask)

        plot_index = 1 if not model_path else 2
        plot_horizontal_diffusivities(axarr_hdiff[plot_index, it // 5], diffusivities)
        plot_vertical_diffusivities(axarr_vdiff[plot_index, it // 5], diffusivities)
        plot_predictions(axarr_pred[plot_index, it // 5], raw, output)

        if not os.path.exists(f"{save_path}"):
            os.makedirs(f"{save_path}")

        f_hdiff.savefig(f"{save_path}/h_diff.png", dpi=300)
        f_vdiff.savefig(f"{save_path}/v_diff.png", dpi=300)
        f_pred.savefig(f"{save_path}/pred.png", dpi=300)

    run_transductive_experiment(args, raw, seeds, mask_x, mask_y, num_classes, summary_callback=comparative_summary)
    run_transductive_experiment(args, raw, seeds, mask_x, mask_y, num_classes, summary_callback=comparative_summary,
                                model_path=f"checkpoints/models/best_model_subsample_{args.subsampling_ratio}")
    plt.tight_layout()

    if not os.path.exists(f"{save_path}"):
        os.makedirs(f"{save_path}")

    f_hdiff.savefig(f"{save_path}/h_diff.png", dpi=300)
    f_vdiff.savefig(f"{save_path}/v_diff.png", dpi=300)
    f_pred.savefig(f"{save_path}/pred.png", dpi=300)
