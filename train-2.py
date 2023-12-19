import os
import logging
import argparse
import torch.nn as nn
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from pathlib import Path
from torchinfo import summary
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from data.cremiDataloading import CremiSegmentationDataset
from randomwalker.randomwalker_loss_utils import NHomogeneousBatchLoss
from datapreprocessing.target_sparse_sampling import SparseMaskTransform
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np

import torch
from unet.unet import UNet
from randomwalker.RandomWalkerModule import RandomWalker

# from data.segmentation_dataset import SegmentationDataset
from utils.evaluation import compute_iou
# from utils.seed_utils import set_seeds
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
        model: nn.Module,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        options: Dict[str, any]
    ):
        """Init method."""
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.options = options

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    def make_summary_plot(self, it, raw, output, net_output, seeds, target, mask, subsampling_ratio, epoch_index):
        """
        This function create and save a summary figure
        """
        f, axarr = plt.subplots(2, 2, figsize=(8, 9.5))
        f.suptitle("RW summary, Iteration: " + repr(it))

        axarr[0, 0].set_title("Ground Truth Image")
        axarr[0, 0].imshow(raw[0].detach().numpy(), cmap="gray")
        axarr[0, 0].imshow(target[0, 0].detach().numpy(), alpha=0.6, vmin=-3, cmap="prism_r")
        seeds_listx, seeds_listy = np.where(seeds[0].data != 0)
        axarr[0, 0].scatter(seeds_listy,
                            seeds_listx, c="r")
        axarr[0, 0].axis("off")

        mask_x, mask_y = np.where(mask != 0)
        axarr[0, 0].scatter(mask_y,
                            mask_x, c="b", s=0.5, marker='o')
        axarr[0, 0].axis("off")

        axarr[0, 1].set_title("LRW output (white seed)")
        axarr[0, 1].imshow(raw[0].detach().numpy(), cmap="gray")
        axarr[0, 1].imshow(np.argmax(output[0][0].detach().numpy(), 0), alpha=0.6, vmin=-3, cmap="prism_r")
        axarr[0, 1].axis("off")

        axarr[1, 0].set_title("Vertical Diffusivities")
        axarr[1, 0].imshow(net_output[0, 0].detach().numpy(), cmap="gray")
        axarr[1, 0].axis("off")

        axarr[1, 1].set_title("Horizontal Diffusivities")
        axarr[1, 1].imshow(net_output[0, 1].detach().numpy(), cmap="gray")
        axarr[1, 1].axis("off")

        plt.tight_layout()
        if not os.path.exists(f"results-full/{subsampling_ratio}/epoch-{epoch_index}/"):
            os.makedirs(f"results-full/{subsampling_ratio}/epoch-{epoch_index}/")
        plt.savefig(f"./results-full/{subsampling_ratio}/epoch-{epoch_index}/{it}.png")
        plt.close()

    
    def sample_seeds(self,seeds_per_region, target, masked_target, mask_x, mask_y, num_classes):
        seeds = torch.zeros_like(target.squeeze())
        if seeds_per_region == 1:
            seed_indices = np.array([
                np.random.choice(np.where(masked_target == i)[0]) for i in range(num_classes)
            ])
        else:
            num_seeds = [
                min(len(np.where(masked_target == i)[0]), seeds_per_region) for i in range(num_classes)
            ]
            seed_indices = np.concatenate([
                np.random.choice(np.where(masked_target == i)[0], num_seeds[i], replace=False)
                for i in range(num_classes)
            ])
        target_seeds = target.squeeze()[mask_x[seed_indices], mask_y[seed_indices]] + 1
        seeds[mask_x[seed_indices], mask_y[seed_indices]] = target_seeds
        seeds = seeds.unsqueeze(0)
        return seeds

    def _process_epoch(self, dataloader: DataLoader, is_training: bool, epoch_index: int):
        phase = "train" if is_training else "valid"
        subsampling_ratio = self.options['subsampling_ratio']
        seeds_per_region = self.options['seeds_per_region']
        self.model.train(is_training)
        total_loss = 0.0
        total_iou = 0.0
        avg_loss = torch.tensor(0.0)
        logging.info(f"Starting {phase} step")
        total_time = 0.0
        if not os.path.exists(f"results-full/{subsampling_ratio}/epoch-{epoch_index}"):
            os.makedirs(f"results-full/{subsampling_ratio}/epoch-{epoch_index}")
        log_file = open(f"results-full/{subsampling_ratio}/epoch-{epoch_index}/log.txt", "a+")
        print(f"Epoch {epoch_index}",file=log_file)
        with torch.set_grad_enabled(is_training):
            num_it = 0
            for it, batch in enumerate(tqdm(dataloader)):
                t1 = time.time()
                images, targets, masks = batch
                # images = images.to(self.device).to(dtype=torch.float)
                # masks = masks.to(self.device).to(dtype=torch.float)
                images = images.to(self.device)
                masks = masks.to(self.device)
                targets = targets.to(self.device)

                num_classes = len(np.unique(targets.squeeze()))

                self.optimizer.zero_grad()
                diffusivities = self.model(images)
                # Diffusivities must be positive
                net_output = torch.sigmoid(diffusivities)

                # valid_mask = False
                # while not valid_mask:
                #     valid_mask = True
                #     mask = SparseMaskTransform(subsampling_ratio)(targets.squeeze())
                #     mask_x, mask_y = mask.nonzero(as_tuple=True)
                #     masked_targets = targets.squeeze()[mask_x, mask_y]

                #     for i in range(num_classes):
                #         labels_in_region = len(np.where(masked_targets == i)[0])
                #         if labels_in_region == 0:
                #             valid_mask = False

                # seeds = self.sample_seeds(seeds_per_region, targets, masked_targets, mask_x, mask_y, num_classes)

                # valid_output = False
                # while not valid_output:
                #     try:
                #         # Random walker
                #         output = self.rw(net_output, seeds)
                #         valid_output = True
                #     except:
                #         print("Singular Laplacian. Resampling seeds!")
                #         seeds_copy = seeds
                #         seeds = self.sample_seeds(seeds_per_region, targets, masked_targets, mask_x, mask_y, num_classes)
                #         print(torch.all(torch.eq(seeds_copy, seeds)).item())
                
                mask = SparseMaskTransform(subsampling_ratio)(targets.squeeze())
                mask_x, mask_y = mask.nonzero(as_tuple=True)
                masked_targets = targets.squeeze()[mask_x, mask_y]

                seeds = self.sample_seeds(seeds_per_region, targets, masked_targets, mask_x, mask_y, num_classes)
                valid_output = False
                count = 0
                while not valid_output:
                    try:
                        # Random walker
                        output = self.rw(net_output, seeds)
                        valid_output = True
                    except:
                        print("Singular Laplacian. Resampling seeds!")
                        print(total_loss)
                        self.make_summary_plot(9000+count, images[0], output, net_output, seeds, targets, mask, subsampling_ratio,epoch_index)
                        seeds = self.sample_seeds(seeds_per_region, targets, masked_targets, mask_x, mask_y, num_classes)
                        count+=1

                # Loss and diffusivities update
                output_log = [torch.log(o)[:, :, mask_x, mask_y] for o in output]
                loss = self.loss_fn(output_log, targets[:, :, mask_x, mask_y])

                if is_training:
                    loss.backward(retain_graph=True)
                    self.optimizer.step()

                pred_masks = torch.argmax(output[0], dim=1)
                iou_score = compute_iou(pred_masks.detach().cpu(), targets[0].detach().cpu(), num_classes)
                total_loss += loss.item()
                total_iou += iou_score
                t2 = time.time()
                total_time += t2-t1
                num_it += 1
                if it % 1 == 0:
                    avg_time = total_time / num_it
                    print(f"Iteration {it}  Time/iteration(s): {avg_time}  Loss: {loss.item()}  mIoU: {iou_score}",file=log_file)
                    self.make_summary_plot(it, images[0], output, net_output, seeds, targets, mask, subsampling_ratio,epoch_index)
                    total_time = 0.0
                    num_it = 0

        avg_loss = total_loss / len(dataloader)
        avg_iou = total_iou / len(dataloader)
        return avg_loss, avg_iou

    def train(self):
        """Train the model with the provided options."""
        max_epochs = self.options['max_epochs']
        best_valid_iou = 0.

        epoch_index = 0
        for epoch in range(max_epochs):
            logging.info(f'Starting Epoch {epoch_index + 1}:')

            train_loss, train_iou = self._process_epoch(self.train_dataloader, True, epoch_index)
            valid_loss, valid_iou = self._process_epoch(self.valid_dataloader, False, epoch_index)

            self.tb_writer.add_scalars('loss', {'train': train_loss, 'valid': valid_loss}, epoch_index + 1)
            self.tb_writer.add_scalars('IoU', {'train': train_iou, 'valid': valid_iou}, epoch_index + 1)
            self.tb_writer.flush()

            if valid_iou > best_valid_iou and epoch_index > 0:
                improvement = valid_iou - best_valid_iou
                best_valid_iou = valid_iou
                model_path = MODEL_SAVE_DIR / f'best_model_{self.timestamp}_{epoch_index}'
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


def main(args):
    raw_transforms = transforms.Compose([
        transforms.Normalize(mean=[0.5], std=[0.5]),
        transforms.FiveCrop(size=(128, 128)),
    ])
    target_transforms = transforms.Compose([
        transforms.FiveCrop(size=(128, 128)),
    ])
    # Create datasets and dataloaders for training and validation
    train_img_dir = Path("./data/train_split/train/img")
    train_mask_dir = Path("./data/train_split/train/mask")
    train_dataset = CremiSegmentationDataset("data/sample_A_20160501.hdf", transform=raw_transforms, target_transform=target_transforms, subsampling_ratio = args.subsampling_ratio, testing=False)

    valid_img_dir = Path("./data/train_split/valid/img")
    valid_mask_dir = Path("./data/train_split/valid/mask")
    valid_dataset = CremiSegmentationDataset("data/sample_A_20160501.hdf", transform=raw_transforms, target_transform=target_transforms, subsampling_ratio = args.subsampling_ratio, testing=True)

    loader_args = dict(batch_size=args.batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_dataloader = DataLoader(train_dataset, shuffle=True, **loader_args)
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, **loader_args)

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
        max_epochs=args.max_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        subsampling_ratio = args.subsampling_ratio,
        seeds_per_region = args.seeds_per_region
    )

    # Create checkpoints folder
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    # log_file = open(f"results-full/log.txt", "a+")

    # Train model
    trainer = Trainer(
        train_dataloader,
        valid_dataloader,
        model,
        loss_fn,
        optimizer,
        options
    )
    trainer.train()


def parse_args():
    parser = argparse.ArgumentParser(description='Train the segmentation model on images and target masks')
    parser.add_argument('--max-epochs', type=int, default=40, help='Maximum number of epochs')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--lr', dest='lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight-decay', dest='weight_decay', type=float, default=1e-2, help='Weight decay')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    parser.add_argument('--min-delta', dest='min_delta', type=float, default=1e-3, help='Early stopping min delta')
    parser.add_argument('--load', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--subsampling-ratio', dest = "subsampling_ratio",type=float, default=0.1, help='Subsampling ratio')
    parser.add_argument('--seeds-per-region', dest = "seeds_per_region",type=int, default=1, help='Seeds per Region')
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
    logging.info(f"Subsampling Ratio: {args.subsampling_ratio}")
    logging.info(f"Seeds per Region: {args.seeds_per_region}")
    main(args)
