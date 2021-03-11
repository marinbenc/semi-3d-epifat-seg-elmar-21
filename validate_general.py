import sys
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2 as cv
from skimage.transform import resize
from sklearn.model_selection import KFold
from scipy.stats import pearsonr

import helpers as h
from utils import dsc
sys.path.append('models')
from unet_plain import UNet
from patients_dataset_eat import PatientsDataset

dataset_folder = 'datasets/eat/'
gt_eat_folder = 'datasets/eat/label'

folds = 20
run = 'logs/joint_training/2021-03-08-11:04:14_fold0/'
models = h.listdir(run)
models.sort()
patients = h.listdir(gt_eat_folder)
patients.sort()

all_dscs = []
all_rs = []

kfold = KFold(n_splits=folds)
folds = kfold.split(patients)

for fold, (train_idxs, valid_idxs) in enumerate(folds):
    validation_patients = list(np.array(patients)[valid_idxs])
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

    model = UNet(in_channels=2, out_channels=2, device='cuda')
    model.to('cuda')
    model.load_state_dict(torch.load(fold_model_path))
    model.eval()

    # all_pixel_counts = []
    # all_predicted_pixel_counts = []

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

        # all_pixel_counts.append(y[1].squeeze().item())
        # all_predicted_pixel_counts.append(predicted_y[1].squeeze().item())

    dscs = []
    for i in range(len(all_predicted_ys)):
        predicted_y = all_predicted_ys[i]
        predicted_y[predicted_y > 0.5] = 1
        predicted_y[predicted_y <= 0.5] = 0

        y = all_ys[i]
        y[y > 0.5] = 1
        y[y <= 0.5] = 0

        # plt.imshow(y.squeeze(0))
        # plt.show()
        # plt.imshow(predicted_y.squeeze(0))
        # plt.show()
        
        dscs.append(dsc(predicted_y, y))

    mean_dsc = np.mean(dscs)
    print(f'DSC fold {fold}: {mean_dsc:.4f}')
    all_dscs.append(mean_dsc)

    #all_pixel_counts, all_predicted_pixel_counts = zip(*sorted(zip(all_pixel_counts, all_predicted_pixel_counts)))

    #r, p = pearsonr(all_pixel_counts, all_predicted_pixel_counts)
    #print(f'Pearson r fold {fold}: {r:.4f}, p = {p}')
    #all_rs.append(r)

    dscs_sort = np.array(dscs).argsort()
    sorted_eats = np.array(all_ys)[dscs_sort]
    sorted_predicted_eats = np.array(all_predicted_ys)[dscs_sort]

print('\n   --- VALIDATION DONE')
print(f'    mean DSC: {np.mean(all_dscs):.4f}, std: {np.std(all_dscs):.4f}')
#print(f'    mean   r: {np.mean(all_rs):.4f}, std: {np.std(all_rs):.4f}')