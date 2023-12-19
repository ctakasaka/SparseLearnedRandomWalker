import os
import logging
import argparse
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from torchinfo import summary
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

import torch
from unet.unet import UNet
from randomwalker.RandomWalkerModule import RandomWalker
from randomwalker.randomwalker_loss_utils import NHomogeneousBatchLoss

# from .data.segmentation_dataset import SegmentationDataset
from utils.evaluation import compute_iou
from utils.seed_utils import set_seeds
from typing import Dict

from data.cremiDataloading import CremiSegmentationDataset
from exampleCREMI import make_summary_plot, sample_seeds

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

    def _process_epoch(self, dataloader: DataLoader, is_training: bool, epoch_index: int):
        phase = "train" if is_training else "valid"
        self.model.train(is_training)
        total_loss = 0.0
        total_iou = 0.0
        loss_type = NHomogeneousBatchLoss(torch.nn.NLLLoss)

        logging.info(f"Starting {phase} step")
        with torch.set_grad_enabled(is_training):
            for i, batch in enumerate(tqdm(dataloader)):
                images = batch[0].to(self.device)
                segmentation = batch[1]
                subsample_mask = batch[2]

                # seed generation
                # TODO: Remake seed_sampling function for batches
                # mask_x, mask_y = subsample_mask.nonzero(as_tuple=True)
                # masked_target = images.squeeze()[mask_x, mask_y]
                sub_segmentation = (segmentation+1 * subsample_mask).squeeze()

                self.optimizer.zero_grad()

                diffusivities = self.model(images)

                # Diffusivities must be positive
                net_output = torch.sigmoid(diffusivities)

                # TODO: get seeds!
                # Random walker
                output = self.rw(net_output, sub_segmentation)
                # Loss and diffusivities update
                output_log = [torch.log(o).reshape((1,o.shape[1], -1)) for o in output]
                # output_log = []
                # target_selected = []
                # # for idx, o in enumerate(output):
                # #     mask_x, mask_y = subsample_mask[idx].nonzero(as_tuple=True)
                # #     output_log.append(torch.log(o)[:, :, mask_x, mask_y])
                # #     target_selected.append(segmentation[idx, :, mask_x, mask_y])
                    
                loss = loss_type(output_log, segmentation.reshape((segmentation.shape[0], segmentation.shape[1], -1)))

                if is_training:
                    loss.backward(retain_graph=True)
                    self.optimizer.step()

                # TODO: fix this!
                iou_score = 0.
                for idx, o in enumerate(output):
                    num_classes = segmentation[idx].unique().shape[0]
                    pred_masks = torch.argmax(o, dim=1).to(torch.float32)
                    iou_score = compute_iou(pred_masks.detach().cpu(), segmentation[idx].detach().cpu(), num_classes)

                total_loss += loss.item()
                total_iou += iou_score

                if i % 1000 == 999:
                    tb_x = epoch_index * len(dataloader) + i + 1
                    self.tb_writer.add_scalar(f'{phase}/loss', total_loss / (i + 1), tb_x)
                    self.tb_writer.add_scalar(f'{phase}/IoU', total_iou / (i + 1), tb_x)

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
    # Create datasets and dataloaders for training and validation
    raw_transforms = transforms.Compose([
        transforms.Normalize(mean=[0.5], std=[0.5]),
        transforms.FiveCrop(size=(100, 100)),
    ])
    target_transforms = transforms.Compose([
        transforms.FiveCrop(size=(100, 100)),
    ])

    # loading in dataset
    train_dataset = CremiSegmentationDataset("data/sample_A_20160501.hdf", transform=raw_transforms, target_transform=target_transforms, subsampling_ratio=0.1, testing=False)
    valid_dataset = CremiSegmentationDataset("data/sample_A_20160501.hdf", transform=raw_transforms, target_transform=target_transforms, subsampling_ratio=0.1, testing=True)

    # loader_args = dict(batch_size=args.batch_size, num_workers=os.cpu_count(), pin_memory=True)
    # train_dataloader = DataLoader(train_dataset, shuffle=True, **loader_args)
    # valid_dataloader = DataLoader(valid_dataset, shuffle=False, **loader_args)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=4)
    valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=4)

    print("Dataset loaded successfully")

    # Create model, load from state (is possible) and log model summary
    model = UNet(1, 32, 2)
    if args.load:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(args.load, map_location=device)
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    input_shape = train_dataset[0][0].shape
    logging.info("Model summary")
    logging.info(summary(model, input_size=(args.batch_size, *input_shape)))  # dimensions of padded images are fixed

    # Create optimizer and loss function
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    loss_fn = nn.BCEWithLogitsLoss()

    # Create options dict
    options = dict(
        max_epochs=args.max_epochs,
        patience=args.patience,
        min_delta=args.min_delta
    )

    # Create checkpoints folder
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

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
    parser.add_argument('--lr', dest='lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', dest='weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    parser.add_argument('--min-delta', dest='min_delta', type=float, default=0., help='Early stopping min delta')
    parser.add_argument('--load', type=str, default=False, help='Load model from a .pth file')

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    set_seeds(0)
    args = parse_args()

    # Log the hyperparameters
    logging.info(f"Max epochs: {args.max_epochs}")
    logging.info(f"Batch size: {args.batch_size}")
    logging.info(f"Learning rate: {args.lr}")
    logging.info(f"Weight decay: {args.weight_decay}")
    logging.info(f"Patience: {args.patience}")
    logging.info(f"Min delta: {args.min_delta}")
    logging.info(f"Load model path: {args.load}")

    main(args)
