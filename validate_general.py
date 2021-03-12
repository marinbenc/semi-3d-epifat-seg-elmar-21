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
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

import helpers as h
from utils import dsc
sys.path.append('models')
from unet_plain import UNet
from patients_dataset_eat import PatientsDataset

def get_dataset(patients):
    dataset = PatientsDataset(
        patient_names=patients,
        inputs_dir=os.path.join(dataset_folder, 'input'),
        labels_dir=os.path.join(dataset_folder, 'label'),
        peri_dir=os.path.join(dataset_folder, 'peri'),
        peri_as_input=False,
        peri_transform=None,
        image_size=128,
        random_sampling=False,
        verbose=False)
    return dataset

def get_predictions(model, patients):
    all_xs = []
    all_ys = []
    all_predicted_ys = []

    dataset = get_dataset(patients)

    for (x, y) in dataset:
        x = x.to('cuda')

        predicted_y = model(x.unsqueeze(0).detach())
        predicted_y = predicted_y.squeeze(0).detach().cpu().numpy()[:1]
        predicted_y[predicted_y > 0.5] = 1
        predicted_y[predicted_y <= 0.5] = 0
        all_predicted_ys.append(predicted_y)

        x = x.squeeze(0).detach().cpu().numpy()
        all_xs.append(x)

        y = y.squeeze(0).detach().cpu().numpy()[:1]
        y[y > 0.5] = 1
        y[y <= 0.5] = 0
        all_ys.append(y)
    
    return all_xs, all_ys, all_predicted_ys

def train_regression(segmentation_model, train_patients):
    all_xs, all_ys, all_predicted_ys = get_predictions(segmentation_model, train_patients)

    predicted_pixels = [np.sum(predicted_y) for predicted_y in all_predicted_ys]
    gt_pixels = [np.sum(y) for y in all_ys]

    predicted_pixels = np.array(predicted_pixels).reshape(-1, 1)
    gt_pixels = np.array(gt_pixels)

    regressor = RandomForestRegressor(n_estimators=10)
    regressor.fit(predicted_pixels, gt_pixels)

    #plt.scatter(predicted_pixels, gt_pixels,  color='black')
    #plt.show()

    return regressor

dataset_folder = 'datasets/eat/'
gt_eat_folder = 'datasets/eat/label'

folds = 4
regression_training_patient_count = 3

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
    regression_idxs = valid_idxs[:regression_training_patient_count]
    valid_idxs = valid_idxs[regression_training_patient_count:]

    validation_patients = list(np.array(patients)[valid_idxs])
    regression_patients = list(np.array(patients)[regression_idxs])

    fold_models_path = h.listdir(run)
    fold_models_path.sort()
    fold_model_path = run + '/' + fold_models_path[-2]

    print(f'model: {fold_model_path}')
    print(f'fold {str(fold)}, validation patients: {validation_patients}')

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

    print(f'Training regression on patients {regression_patients}...')
    regressor = train_regression(model, regression_patients)

    print('Validating...')

    all_xs, all_ys, all_predicted_ys = get_predictions(model, validation_patients)

    dscs = []
    all_pixel_counts = []
    all_predicted_pixel_counts = []
    all_adjusted_pixel_counts = []

    for i in range(len(all_predicted_ys)):
        predicted_y = all_predicted_ys[i]
        y = all_ys[i]

        positives = predicted_y.sum()

        adjusted = regressor.predict(np.array([positives]).reshape(-1, 1))[0]
        adjusted = 0 if adjusted < 0 else adjusted

        all_adjusted_pixel_counts.append(adjusted)
        all_pixel_counts.append(y.sum())
        all_predicted_pixel_counts.append(positives)
        dscs.append(dsc(predicted_y, y))

        # plt.imshow(y.squeeze(0))
        # plt.show()
        # plt.imshow(predicted_y.squeeze(0))
        # plt.show()
        
    mean_dsc = np.mean(dscs)
    print(f'DSC fold {fold}: {mean_dsc:.4f}')
    all_dscs.append(mean_dsc)

    all_pixel_counts, all_predicted_pixel_counts, all_adjusted_pixel_counts = zip(*sorted(zip(
        all_pixel_counts, 
        all_predicted_pixel_counts, 
        all_adjusted_pixel_counts)))

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