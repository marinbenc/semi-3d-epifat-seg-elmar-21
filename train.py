import argparse
import json
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import datetime

import sys
sys.path.append('dataset_pericardium')

from dataset import PericardiumDataset as Dataset
from logger import Logger
from loss import DiceLoss
sys.path.append('models')
from unet_plain import UNet
from utils import log_images, dsc
from dice_metric import DiceMetric
from ignite.utils import setup_logger
from ignite.handlers import ModelCheckpoint

from ignite.contrib.handlers.tensorboard_logger import (
    GradsHistHandler,
    GradsScalarHandler,
    TensorboardLogger,
    WeightsHistHandler,
    WeightsScalarHandler,
    global_step_from_engine,
)

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, ConfusionMatrix, DiceCoefficient

def main(args):
    makedirs(args)
    snapshotargs(args)
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)

    loader_train, loader_valid = data_loaders(args)

    model = UNet(in_channels=Dataset.in_channels, out_channels=Dataset.out_channels, device=device)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    dsc_loss = DiceLoss()

    train_metrics = {
        "loss": Loss(dsc_loss),
        "dsc": DiceMetric(loader_train, device=device)
    }
    
    val_metrics = {
        "loss": Loss(dsc_loss),
        "dsc": DiceMetric(loader_valid, device=device)
    }

    trainer = create_supervised_trainer(model, optimizer, dsc_loss, device=device)
    trainer.logger = setup_logger("Trainer")

    train_evaluator = create_supervised_evaluator(model, metrics=train_metrics, device=device)
    train_evaluator.logger = setup_logger("Train Evaluator")
    validation_evaluator = create_supervised_evaluator(model, metrics=val_metrics, device=device)
    validation_evaluator.logger = setup_logger("Val Evaluator")

    @trainer.on(Events.EPOCH_COMPLETED)
    def compute_metrics(engine):
        train_evaluator.run(loader_train)
        validation_evaluator.run(loader_valid)
    
    log_dir = 'logs/' + datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    tb_logger = TensorboardLogger(log_dir=log_dir)

    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=100),
        tag="training",
        output_transform=lambda loss: {"batchloss": loss},
        metric_names="all",
    )

    for tag, evaluator in [("training", train_evaluator), ("validation", validation_evaluator)]:
        tb_logger.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag=tag,
            metric_names=["loss", "dsc"],
            global_step_transform=global_step_from_engine(trainer),
        )

    def score_function(engine):
        return engine.state.metrics["dsc"]

    model_checkpoint = ModelCheckpoint(
        log_dir,
        n_saved=2,
        filename_prefix="best",
        score_function=score_function,
        score_name="dsc",
        global_step_transform=global_step_from_engine(trainer),
        require_empty=False
    )
    validation_evaluator.add_event_handler(Events.COMPLETED, model_checkpoint, {"model": model})
    
    try:
        trainer.run(loader_train, max_epochs=120)
    except KeyboardInterrupt:
        tb_logger.close()

    tb_logger.close()

def data_loaders(args):
    dataset_train, dataset_valid = datasets(args)

    def worker_init(worker_id):
        np.random.seed(42 + worker_id)

    loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        worker_init_fn=worker_init,
    )
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.workers,
        worker_init_fn=worker_init,
    )

    return loader_train, loader_valid


def datasets(args):
    train = Dataset(
        inputs_dir=os.path.join(args.images, 'input'),
        labels_dir=os.path.join(args.images, 'label'),
        subset="train",
        image_size=args.image_size,
        #transform=transforms(scale=args.aug_scale, angle=args.aug_angle, flip_prob=0.5),
    )
    valid = Dataset(
        inputs_dir=os.path.join(args.images, 'input'),
        labels_dir=os.path.join(args.images, 'label'),
        subset="validation",
        image_size=args.image_size,
        random_sampling=False,
    )
    return train, valid

def makedirs(args):
    os.makedirs(args.weights, exist_ok=True)
    os.makedirs(args.logs, exist_ok=True)


def snapshotargs(args):
    args_file = os.path.join(args.logs, "args.json")
    with open(args_file, "w") as fp:
        json.dump(vars(args), fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training U-Net model"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="input batch size for training (default: 16)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="initial learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="device for training (default: cuda:0)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="number of workers for data loading (default: 4)",
    )
    parser.add_argument(
        "--vis-images",
        type=int,
        default=200,
        help="number of visualization images to save in log file (default: 200)",
    )
    parser.add_argument(
        "--vis-freq",
        type=int,
        default=10,
        help="frequency of saving images to log file (default: 10)",
    )
    parser.add_argument(
        "--weights", type=str, default="./weights", help="folder to save weights"
    )
    parser.add_argument(
        "--logs", type=str, default="./logs", help="folder to save logs"
    )
    parser.add_argument(
        "--images", type=str, default="./kaggle_3m", help="root folder with images"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="target input image size (default: 128)",
    )
    parser.add_argument(
        "--aug-scale",
        type=int,
        default=0.05,
        help="scale factor range for augmentation (default: 0.05)",
    )
    parser.add_argument(
        "--aug-angle",
        type=int,
        default=15,
        help="rotation angle range in degrees for augmentation (default: 15)",
    )
    args = parser.parse_args()
    main(args)