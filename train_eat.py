'''
Adapted from:
Mateusz Buda, Ashirbani Saha, Maciej A. Mazurowski,
Association of genomic subtypes of lower-grade gliomas with shape features automatically extracted by a deep learning algorithm,
Computers in Biology and Medicine, Volume 109, 2019, Pages 218-225, ISSN 0010-4825,
https://doi.org/10.1016/j.compbiomed.2019.05.002.
https://github.com/mateuszbuda/brain-segmentation-pytorch
'''

import argparse
import json
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import datetime
from sklearn.model_selection import KFold
import helpers as h
import sys
from ignite.utils import setup_logger
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers.tensorboard_logger import (
    TensorboardLogger,
    global_step_from_engine,
)
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, ConfusionMatrix, DiceCoefficient

from loss import DiceLoss
sys.path.append('models')
from unet_plain import UNet
from patients_dataset_eat import PatientsDataset as Dataset
from utils import dsc
from dice_metric import DiceMetric
from transform import transforms

def main(args):
    makedirs(args)
    snapshotargs(args)
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")

    all_patients = h.listdir(os.path.join(args.images, 'input'))
    all_patients.sort()

    kfold = KFold(n_splits=args.folds)
    folds = kfold.split(all_patients)

    fold_dscs = []

    for fold, (train_idxs, valid_idxs) in enumerate(folds):
        train_patients = list(np.array(all_patients)[train_idxs])
        valid_patients = list(np.array(all_patients)[valid_idxs])
        print(valid_patients, train_patients)

        best_dsc = train_fold(args, fold, device, train_patients, valid_patients)
        fold_dscs.append(best_dsc)
    
    mean_dsc = np.mean(fold_dscs)
    print(f'Mean CV DSC: {mean_dsc:.4f}')

def train_fold(args, fold, device, train_patients, valid_patients):
    '''
    Trains a single fold and returns the best DSC score.
    '''

    print(f'\n\n --- Fold {str(fold)} --- \n\n')

    loader_train, loader_valid = data_loaders(args, train_patients, valid_patients)

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

    best_dsc = 0

    @trainer.on(Events.EPOCH_COMPLETED)
    def compute_metrics(engine):
        nonlocal best_dsc
        train_evaluator.run(loader_train)
        validation_evaluator.run(loader_valid)
        curr_dsc = validation_evaluator.state.metrics['dsc']
        if curr_dsc > best_dsc:
            best_dsc = curr_dsc


    log_dir = f'logs/{datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}_fold{fold}'
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

    trainer.run(loader_train, max_epochs=args.epochs)
    tb_logger.close()

    return best_dsc

def data_loaders(args, patients_train, patients_valid):
    dataset_train, dataset_valid = datasets(args, patients_train, patients_valid)

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

def datasets(args, patients_train, patients_valid):
    train = Dataset(
        patient_names=patients_train,
        inputs_dir=os.path.join(args.images, 'input'),
        labels_dir=os.path.join(args.images, 'label'),
        peri_dir=os.path.join(args.images, 'peri'),
        image_size=args.image_size,
        transform=None,
    )
    valid = Dataset(
        patient_names=patients_valid,
        inputs_dir=os.path.join(args.images, 'input'),
        labels_dir=os.path.join(args.images, 'label'),
        peri_dir=os.path.join(args.images, 'peri'),
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
        "--folds",
        type=int,
        default=5,
        help="number of folds (default: 5)",
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
        "--workers",
        type=int,
        default=4,
        help="number of workers for data loading (default: 4)",
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