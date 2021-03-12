import sys
import os

import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2 as cv
from skimage.transform import resize
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

import helpers as h
from utils import dsc
sys.path.append('models')
from unet_plain import UNet
from patients_dataset_eat import PatientsDataset

def estimate_model_parameters(model, train_patients):
    all_xs = []
    all_ys = []
    all_predicted_ys = []

    for (x, y) in dataset:
        all_xs.append(x.squeeze(0).detach().cpu().numpy())
        all_ys.append(y.squeeze(0).detach().cpu().numpy()[:1])

        x = x.to('cuda')
        predicted_y = model(x.unsqueeze(0).detach())
        squeezed = predicted_y.squeeze(0).detach().cpu().numpy()[:1]
        all_predicted_ys.append(squeezed)

    tprs = []
    fprs = []

    for i in range(len(all_predicted_ys)):
        predicted_y = all_predicted_ys[i]

        predicted_y[predicted_y > 0.5] = 1
        predicted_y[predicted_y <= 0.5] = 0

        y = all_ys[i]
        y[y > 0.5] = 1
        y[y <= 0.5] = 0

        true_positives    = np.sum(np.logical_and(y == 1, predicted_y == 1))
        false_positives   = np.sum(np.logical_and(y == 1, predicted_y == 0))
        true_negatives    = np.sum(np.logical_and(y == 0, predicted_y == 0))
        false_negatives   = np.sum(np.logical_and(y == 0, predicted_y == 1))

        if true_positives == 0:
            tpr = 1
        else:
            tpr = true_positives / float(true_positives + false_negatives)
        tprs.append(tpr)

        if false_positives == 0:
            fpr = 0
        else:
            fpr = false_positives / float(false_positives + true_negatives)
        fprs.append(fpr)

    return np.mean(tprs), np.mean(fprs)

dataset_folder = 'datasets/eat/'
gt_eat_folder = 'datasets/eat/label'

folds = 2
run = 'logs/2021-03-11-13:54:19_fold0/'
models = h.listdir(run)
models.sort()
patients = h.listdir(gt_eat_folder)
patients.sort()

all_dscs = []
all_rs = []
all_adjusted_rs = []

kfold = KFold(n_splits=folds)
folds = kfold.split(patients)

for fold, (train_idxs, valid_idxs) in enumerate(folds):
    if fold > 0:
        break
    validation_patients = list(np.array(patients)[valid_idxs[:len(valid_idxs)//2]])
    dataset = PatientsDataset(
        patient_names=validation_patients,
        inputs_dir=os.path.join(dataset_folder, 'input'),
        labels_dir=os.path.join(dataset_folder, 'label'),
        peri_dir=os.path.join(dataset_folder, 'peri'),
        peri_as_input=False,
        peri_transform=None,
        image_size=128,
        random_sampling=False,
        verbose=False)

    fold_models_path = h.listdir(run)
    fold_models_path.sort()
    fold_model_path = run + '/' + fold_models_path[-2]
    print(f'validating model {fold_model_path}')
    print(f'fold {str(fold)}, patients {validation_patients}')

    model = smp.Unet(
        encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=2,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=2,                      # model output channels (number of classes in your dataset),
        activation="sigmoid"
    )
    model.to('cuda')
    model.load_state_dict(torch.load(fold_model_path))
    model.eval()

    tpr, fpr = estimate_model_parameters(model, list(np.array(patients)[valid_idxs[len(valid_idxs)//2:]]))

    all_pixel_counts = []
    all_predicted_pixel_counts = []
    all_adjusted_pixel_counts = []

    all_xs = []
    all_ys = []
    all_predicted_ys = []

    for (x, y) in dataset:
        all_xs.append(x.squeeze(0).detach().cpu().numpy())
        all_ys.append(y.squeeze(0).detach().cpu().numpy()[:1])

        x = x.to('cuda')
        predicted_y = model(x.unsqueeze(0).detach())
        squeezed = predicted_y.squeeze(0).detach().cpu().numpy()[:1]
        all_predicted_ys.append(squeezed)

    dscs = []

    for i in range(len(all_predicted_ys)):
        predicted_y = all_predicted_ys[i]

        # plt.imshow(all_ys[i].squeeze())
        # plt.show()
        # plt.imshow(predicted_y.squeeze())
        # plt.show()

        predicted_y[predicted_y > 0.5] = 1
        predicted_y[predicted_y <= 0.5] = 0

        y = all_ys[i]
        y[y > 0.5] = 1
        y[y <= 0.5] = 0

        positives = predicted_y.sum()
        total = predicted_y.shape[-1] * predicted_y.shape[-2]
        adjusted = (positives - fpr * total) / (tpr - fpr)

        all_pixel_counts.append(y.sum())
        all_predicted_pixel_counts.append(positives)
        all_adjusted_pixel_counts.append(adjusted)

        # plt.imshow(y.squeeze(0))
        # plt.show()
        # plt.imshow(predicted_y.squeeze(0))
        # plt.show()
        
        dscs.append(dsc(predicted_y, y))

    mean_dsc = np.mean(dscs)
    print(f'DSC fold {fold}: {mean_dsc:.4f}')
    all_dscs.append(mean_dsc)

    all_pixel_counts, all_predicted_pixel_counts, all_adjusted_pixel_counts = zip(*sorted(zip(all_pixel_counts, all_predicted_pixel_counts, all_adjusted_pixel_counts)))

    r, p = pearsonr(all_pixel_counts, all_predicted_pixel_counts)
    print(f'Pearson r fold {fold}: {r:.4f}, p = {p}')
    all_rs.append(r)

    r_adjusted, _ = pearsonr(all_pixel_counts, all_adjusted_pixel_counts)
    print(f'Adjusted r fold {fold}: {r_adjusted:.4f}, p = {p}')
    all_adjusted_rs.append(r_adjusted)

    dscs_sort = np.array(dscs).argsort()

print('\n   --- VALIDATION DONE')
print(f'    mean DSC: {np.mean(all_dscs):.4f}, std: {np.std(all_dscs):.4f}')
print(f'    mean   r: {np.mean(all_rs)}, std: {np.std(all_rs):.4f}')
print(f'    adjs   r: {np.mean(all_adjusted_rs)}, std: {np.std(all_adjusted_rs):.4f}')