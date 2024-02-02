from randomwalker.RandomWalkerModule import RandomWalker
import torch
import numpy as np
from randomwalker.randomwalker_loss_utils import NHomogeneousBatchLoss
from unet.unet import UNet
from utils.evaluation_utils import compute_iou
import time
import os
import yaml
from argparse import Namespace

from tqdm import tqdm

from torchvision import transforms
from data.cremi_dataloader import CremiSegmentationDataset
from utils.seed_utils import sample_seeds, get_all_seeds
from utils.plotting_utils import save_summary_plot
from train import get_base_parser, EarlyStopper


def parse_args():
    parser = get_base_parser(description='Train on a single neuronal image')
    parser.add_argument('--all', type=bool, default=False)
    parser.add_argument('--image-index', dest='img_idx', type=int, default=0)
    parser.add_argument('--load', type=str, default=False, help='Load model from a .pth file')

    args = parser.parse_args()

    if args.config is not None:
        with open(args.config, "r") as stream:
            config_params = yaml.load(stream, Loader=yaml.FullLoader)
            args_dict = vars(args)
            args_dict.update(config_params)
            args = Namespace(**args_dict)

    return args


def run_baseline_experiment(raw, target, mask_x, mask_y, num_classes, save_path, threshold=0.6, summary_callback=None):
    # Init the random walker module (just for forward, backprop will not be used)
    rw = RandomWalker(0, max_backprop=False)

    threshold_diffusivities = raw[0] / torch.max(raw)
    threshold_diffusivities[threshold_diffusivities < threshold] = 1e-5
    threshold_diffusivities[threshold_diffusivities > threshold] = 1

    diffusivities = torch.stack([threshold_diffusivities, threshold_diffusivities]).unsqueeze(0)

    # Random walker
    seeds = get_all_seeds(target, mask_x, mask_y)
    output = rw(diffusivities, seeds)

    pred = torch.argmax(output[0], dim=1)
    iou_score = compute_iou(pred, target[0], num_classes)
    if summary_callback is not None:
        summary_callback(raw, seeds, mask, diffusivities, pred, target, 0, 0.0, 0.0, iou_score, save_path, False)
    return iou_score


def run_best_baseline_experiment(raw, target, mask_x, mask_y, num_classes, thresholds, save_path, summary_callback=None):
    iou_list = []
    for threshold in thresholds:
        iou = run_baseline_experiment(raw, target, mask_x, mask_y, num_classes, save_path, threshold=threshold)
        iou_list.append(iou)
    print(iou_list)
    best_iou = max(iou_list)
    best_threshold = thresholds[np.argmax(iou_list)]
    print(f"Best threshold: {best_threshold} - Best mIoU: {best_iou}")

    # Get summary plot for best mIoU threshold
    run_baseline_experiment(raw, target, mask_x, mask_y, num_classes, save_path, threshold=best_threshold,
                            summary_callback=summary_callback)
    return best_iou


def run_transductive_experiment(args, raw, target, seeds, mask_x, mask_y, num_classes, save_path, 
                                summary_callback=None, model_path=False):
    # Init the UNet
    unet = UNet(1, args.unet_channels, args.unet_blocks)
    if model_path:
        unet.load_state_dict(torch.load(model_path))

    # Init the random walker modules
    rw = RandomWalker(args.sampled_gradients, max_backprop=args.gradient_pruning)

    # Init optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=0.01, weight_decay=args.weight_decay)

    # Loss has to been wrapped in order to work with random walker algorithm
    loss_fn = NHomogeneousBatchLoss(torch.nn.NLLLoss)

    early_stopper = EarlyStopper(
        patience=args.patience,
        min_delta=args.min_delta
    )

    # Main overfit loop
    total_time = 0.0
    num_it = 0
    iou_score = -np.inf
    for it in tqdm(range(iterations + 1)):
        t1 = time.time()
        optimizer.zero_grad()

        net_output = unet(raw.unsqueeze(0))

        # Diffusivities must be positive
        diffusivities = torch.sigmoid(net_output)

        # Random walker
        output = rw(diffusivities, seeds)

        # Loss and diffusivities update
        output_log = [torch.log(o)[:, :, mask_x, mask_y] for o in output]

        l = loss_fn(output_log, target[:, :, mask_x, mask_y])
        if early_stopper.should_early_stop(-l.item()):
            print(f'Early stopping condition triggered at iteration {it}. Training terminated.')

            # Compute final predictions using all seeds
            all_seeds = get_all_seeds(target, mask_x, mask_y)
            output = rw(diffusivities, all_seeds)
            pred = torch.argmax(output[0], dim=1)
            iou_score = compute_iou(pred, target[0], num_classes)
            avg_time = total_time / num_it
            if summary_callback is not None:
                summary_callback(raw, all_seeds, mask, diffusivities, pred, target, it, avg_time, l, iou_score, save_path)
            return iou_score

        l.backward(retain_graph=True)
        optimizer.step()

        t2 = time.time()
        total_time += t2 - t1
        num_it += 1

    all_seeds = get_all_seeds(target, mask_x, mask_y)
    output = rw(diffusivities, all_seeds)
    pred = torch.argmax(output[0], dim=1)
    iou_score = compute_iou(pred, target[0], num_classes)
    if summary_callback is not None:
        avg_time = total_time / num_it
        summary_callback(raw, all_seeds, mask, diffusivities, pred, target, it, avg_time, l, iou_score, save_path)
    return iou_score


def transductive_summary(raw, seeds, mask, diffusivities, pred, target, it, avg_time, loss, iou_score, save_path,
                         model_path=False):
    print(f"Iteration {it}  Time/iteration(s): {avg_time}  Loss: {loss}  mIoU: {iou_score}",
          file=log_file)
    save_summary_plot(raw, target, seeds, diffusivities, pred, mask, args.subsampling_ratio,
                      it, iou_score, save_path)


if __name__ == '__main__':
    args = parse_args()

    save_path = f"{args.experiment_name}-{'all' if args.all else args.img_idx}-seeds-{args.seeds_per_region}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    log_file = open(f"{save_path}/log.txt", "a+")
    print(save_path, file=log_file)

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
    dataset = CremiSegmentationDataset("data/sample_A_20160501.hdf", transform=raw_transforms,
                                       target_transform=target_transforms,
                                       subsampling_ratio=args.subsampling_ratio, split="all")

    iou_list = []
    baseline_list = []
    for i in range(len(dataset)):
        if not args.all and i != args.img_idx:
            continue
        print(f"Starting transductive traning on image {i} / {len(dataset)}")
        raw, target, mask = dataset[i]
        target = target.unsqueeze(0)
        num_classes = len(np.unique(target))

        # Define sparse mask for the loss
        mask = mask.squeeze()
        mask_x, mask_y = mask.nonzero(as_tuple=True)
        masked_targets = target.squeeze()[mask_x, mask_y]

        # Generate seeds
        seeds = sample_seeds(args.seeds_per_region, target, masked_targets, mask_x, mask_y, num_classes)

        # Run transductive experiment
        iou = run_transductive_experiment(args, raw, target, seeds, mask_x, mask_y, num_classes,
                                            save_path=f"{save_path}/img_{i}",
                                            summary_callback=transductive_summary,
                                            model_path=args.load)
        baseline_iou = run_best_baseline_experiment(raw, target, mask_x, mask_y, num_classes,
                                                    [0.45, 0.5, 0.55, 0.6, 0.65],
                                                    save_path=f"{save_path}/baseline_img_{i}",
                                                    summary_callback=transductive_summary)
        iou_list.append(iou)
        baseline_list.append(baseline_iou)
        print(f"Image {i} - mIoU: {iou} - baseline mIoU: {baseline_iou}")
    print(f"List of mIoU per image: {iou_list}")
    print(f"Average of mIoU: {np.mean(iou_list)}")
    print(f"Std of mIoU: {np.std(iou_list)}")

    print(f"List of baseline mIoU per image: {baseline_list}")
    print(f"Average of baseline mIoU: {np.mean(baseline_list)}")
    print(f"Std of baseline mIoU: {np.std(baseline_list)}")
