import os
import logging
import argparse
import torch.nn as nn
import time
from tqdm import tqdm
from pathlib import Path
from torchinfo import summary
from torchvision import transforms
from torch.utils.data import DataLoader
from data.cremi_dataloader import CremiSegmentationDataset
from randomwalker.randomwalker_loss_utils import NHomogeneousBatchLoss
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np

import torch
from unet.unet import UNet
from randomwalker.RandomWalkerModule import RandomWalker

from utils.seed_utils import sample_seeds
from utils.evaluation_utils import compute_iou
from utils.plotting_utils import save_summary_plot
from typing import Dict


MODEL_SAVE_DIR = Path('checkpoints/models')


class EarlyStopper:
    """Early stopping based on validation performance."""
    def __init__(self, patience=1, min_delta=0):
        """Init method."""
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.max_iou = 0.

    def should_early_stop(self, iou: float) -> bool:
        """Determine if early stopping criterion is met."""
        if iou > self.max_iou:
            self.max_iou = iou
            self.counter = 0
        elif iou <= (self.max_iou - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class Trainer:
    """Trainer."""
    def __init__(
        self,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        test_dataloader: DataLoader,
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        options: Dict[str, any]
    ):
        """Init method."""
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.options = options

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = 'cpu'
        logging.info(f'Using device {self.device}')

        self.model.to(device=self.device)

        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.tb_writer = SummaryWriter(f'tensorboard/segmentation_trainer_{self.timestamp}')

        self.early_stopper = EarlyStopper(
            patience=options['patience'],
            min_delta=options['min_delta']
        )

        self.rw = RandomWalker(options.get('rw_num_grad', 1000),
                               max_backprop=options.get('rw_max_backprop', True))

    def _process_epoch(self, dataloader: DataLoader, phase: str, epoch_index: int):
        is_training = (phase == "train")
        subsampling_ratio = self.options['subsampling_ratio']
        seeds_per_region = self.options['seeds_per_region']
        diffusivity_threshold = self.options['diffusivity_threshold']
        self.model.train(is_training)
        total_loss = 0.0
        total_iou = 0.0

        logging.info(f"Starting {phase} step")
        total_time = 0.0
        save_path = f"results-{phase}/{subsampling_ratio}/epoch-{epoch_index}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        log_file = open(f"{save_path}/log.txt", "a+")
        print(f"Epoch {epoch_index}", file=log_file)
        
        with torch.set_grad_enabled(is_training):
            num_it = 0
            for it, batch in enumerate(tqdm(dataloader)):
                t1 = time.time()
                images, targets, masks = batch

                images = images.to(self.device)
                masks = masks.to(self.device)
                targets = targets.to(self.device)

                num_classes = len(np.unique(targets.squeeze()))

                self.optimizer.zero_grad()
                net_output = self.model(images)

                # Diffusivities must be positive
                diffusivities = torch.sigmoid(net_output)

                mask = masks.squeeze()
                mask_x, mask_y = mask.nonzero(as_tuple=True)
                masked_targets = targets.squeeze()[mask_x, mask_y]

                seeds = sample_seeds(seeds_per_region, targets, masked_targets, mask_x, mask_y, num_classes)

                if not is_training and diffusivity_threshold:
                    diffusivities = (diffusivities >= diffusivity_threshold).to(torch.float32) + 1e-5

                # Random walker
                output = self.rw(diffusivities, seeds)

                # Loss and diffusivities update
                output_log = [torch.log(o + 1e-10)[:, :, mask_x, mask_y] for o in output]
                loss = self.loss_fn(output_log, targets[:, :, mask_x, mask_y])

                if is_training:
                    with torch.autograd.set_detect_anomaly(True):
                        loss.backward(retain_graph=True)
                        self.optimizer.step()

                pred_masks = torch.argmax(output[0], dim=1)
                iou_score = compute_iou(pred_masks.detach().cpu(), targets[0].detach().cpu(), num_classes)
                total_loss += loss.item()
                total_iou += iou_score
                t2 = time.time()
                total_time += t2-t1
                num_it += 1
                if it % 10 == 0 or phase == "test":
                    avg_time = total_time / num_it
                    print(f"Iteration {it}  Time/iteration(s): {avg_time}  Loss: {loss.item()}  mIoU: {iou_score}",
                          file=log_file)
                    if phase != "test":
                        save_summary_plot(images[0], targets, seeds, diffusivities, output, mask, subsampling_ratio,
                                          it, iou_score, save_path)
                    else:
                        save_summary_plot(images[0], targets, seeds, diffusivities, output, None, subsampling_ratio,
                                          it, iou_score, save_path)
                    total_time = 0.0
                    num_it = 0

        avg_loss = total_loss / len(dataloader)
        avg_iou = total_iou / len(dataloader)
        return avg_loss, avg_iou

    def train(self):
        """Train the model with the provided options."""
        max_epochs = self.options['max_epochs']
        subsampling_ratio = self.options['subsampling_ratio']
        best_valid_iou = 0.

        epoch_index = 0
        for epoch in range(max_epochs):
            logging.info(f'Starting Epoch {epoch_index + 1}:')

            train_loss, train_iou = self._process_epoch(self.train_dataloader, "train", epoch_index)
            valid_loss, valid_iou = self._process_epoch(self.valid_dataloader, "valid", epoch_index)

            self.tb_writer.add_scalars('loss', {'train': train_loss, 'valid': valid_loss},
                                       epoch_index + 1)
            self.tb_writer.add_scalars('IoU', {'train': train_iou, 'valid': valid_iou},
                                       epoch_index + 1)
            self.tb_writer.flush()

            if valid_iou > best_valid_iou and epoch_index > 0:
                improvement = valid_iou - best_valid_iou
                best_valid_iou = valid_iou
                model_path = (MODEL_SAVE_DIR /
                              f'best_model_{self.timestamp}_{epoch_index}_subsample_{subsampling_ratio}.pt')
                torch.save(self.model.state_dict(), model_path)
                logging.info(
                    f'Best model saved at epoch {epoch_index + 1}.\
                    IoU: {best_valid_iou:.6f} (improved by {improvement:.6f})'
                )

            if self.early_stopper.should_early_stop(valid_iou):
                logging.info('Early stopping condition triggered. Training terminated.')
                break

            logging.info(f'Losses - train: {train_loss}, valid: {valid_loss}')
            logging.info(f'IoUs - train: {train_iou}, valid: {valid_iou}')
            epoch_index += 1

        # Save last model checkpoint
        model_path = MODEL_SAVE_DIR / f'last_model_{self.timestamp}_{epoch_index}'
        torch.save(self.model.state_dict(), model_path)

    def test(self):
        epoch_index = 0
        test_loss, test_iou = self._process_epoch(self.test_dataloader, "test", epoch_index)

        self.tb_writer.add_scalars('loss', {'test': test_loss}, epoch_index + 1)
        self.tb_writer.add_scalars('IoU', {'valid': test_iou}, epoch_index + 1)
        self.tb_writer.flush()

        logging.info(f'Losses - test: {test_loss}')
        logging.info(f'IoUs - test: {test_iou}')


def main(args):
    raw_transforms = transforms.Compose([
        transforms.Normalize(mean=[0.5], std=[0.5]),
        transforms.CenterCrop(size=(args.resolution, args.resolution)),
    ])
    target_transforms = transforms.Compose([
        transforms.CenterCrop(size=(args.resolution, args.resolution)),
    ])
    # Create datasets and dataloaders for training and validation
    train_dataset = CremiSegmentationDataset("data/sample_A_20160501.hdf",
                                             transform=raw_transforms, target_transform=target_transforms,
                                             subsampling_ratio=args.subsampling_ratio, split="train")

    valid_dataset = CremiSegmentationDataset("data/sample_A_20160501.hdf",
                                             transform=raw_transforms, target_transform=target_transforms,
                                             subsampling_ratio=args.subsampling_ratio, split="validation")

    test_dataset = CremiSegmentationDataset("data/sample_A_20160501.hdf",
                                            transform=raw_transforms, target_transform=target_transforms,
                                            subsampling_ratio=args.subsampling_ratio, split="test")

    loader_args = dict(batch_size=args.batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_dataloader = DataLoader(train_dataset, shuffle=True, **loader_args)
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, **loader_args)
    test_dataloader = DataLoader(test_dataset, shuffle=False, **loader_args)

    # Create model, load from state (is possible) and log model summary
    model = UNet(1, 32, 3)
    if args.load:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(args.load, map_location=device)
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    input_shape = train_dataset[0][0].shape
    logging.info("Model summary")
    logging.info(summary(model, input_size=(args.batch_size, *input_shape)))  # dimensions of padded images are fixed

    # Create optimizer and loss function
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    loss_fn = NHomogeneousBatchLoss(torch.nn.NLLLoss)

    # Create options dict
    options = dict(
        max_epochs=args.max_epochs if not args.test else 1,
        patience=args.patience,
        min_delta=args.min_delta,
        subsampling_ratio=args.subsampling_ratio,
        seeds_per_region=args.seeds_per_region,
        diffusivity_threshold=args.diffusivity_threshold,
        rw_num_grad=args.rw_num_grad,
        rw_max_backprop=args.rw_max_backprop
    )

    # Create checkpoints folder
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        train_dataloader,
        valid_dataloader,
        test_dataloader,
        model,
        loss_fn,
        optimizer,
        options
    )
    if not args.test:
        # Train model
        trainer.train()
    else:
        # Test model
        trainer.test()


def parse_args():
    parser = argparse.ArgumentParser(description='Train the segmentation model on images and target masks')
    parser.add_argument('--max-epochs', type=int, default=40, help='Maximum number of epochs')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--lr', dest='lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight-decay', dest='weight_decay', type=float, default=1e-2,
                        help='Weight decay')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    parser.add_argument('--min-delta', dest='min_delta', type=float, default=1e-3,
                        help='Early stopping min delta')
    parser.add_argument('--load', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--diffusivity-threshold', dest="diffusivity_threshold", type=float, default=False,
                        help='Diffusivity threshold')
    parser.add_argument('--rw-num-grad', dest="rw_num_grad", type=int, default=1000,
                        help='Number of sampled gradients for Random Walker backprop')
    parser.add_argument('--rw-max-backprop', dest="rw_max_backprop", type=bool, default=True,
                        help='Whether to use gradient pruning in Random Walker backprop')
    parser.add_argument('--subsampling-ratio', dest="subsampling_ratio", type=float, default=0.01,
                        help='Subsampling ratio')
    parser.add_argument('--seeds-per-region', dest="seeds_per_region", type=int, default=5,
                        help='Seeds per Region')
    parser.add_argument('--resolution', type=int, default=512,
                        help='Image resolution')
    parser.add_argument('--test', type=bool, action=argparse.BooleanOptionalAction, default=False,
                        help='Evaluates model on test dataset')
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    # set_seeds(0)
    args = parse_args()

    # Log the hyperparameters
    logging.info(f"Max epochs: {args.max_epochs}")
    logging.info(f"Batch size: {args.batch_size}")
    logging.info(f"Learning rate: {args.lr}")
    logging.info(f"Weight decay: {args.weight_decay}")
    logging.info(f"Patience: {args.patience}")
    logging.info(f"Min delta: {args.min_delta}")
    logging.info(f"Load model path: {args.load}")
    logging.info(f"Diffusivity threshold: {args.diffusivity_threshold}")
    logging.info(f"Number of sampled gradients: {args.rw_num_grad}")
    logging.info(f"Using gradient pruning: {args.rw_max_backprop}")
    logging.info(f"Subsampling Ratio: {args.subsampling_ratio}")
    logging.info(f"Seeds per Region: {args.seeds_per_region}")
    logging.info(f"Image resolution: {args.resolution}")
    logging.info(f"Test mode: {args.test}")
    main(args)
